use std::{fs::File, io::{BufWriter, Write, Error as IoError}};

use crate::{part::{Part, Primitives, Conserved}, equation_of_state::EquationOfState, riemann_solver::RiemannSolver};

#[derive(Clone, Copy)]
pub enum Boundary {
    Periodic,
    Reflective,
    Open,
    Vacuum,
}

pub struct Space {
    parts: Vec<Part>,
    boundary: Boundary,
    box_size: f64,
    num_parts: usize,
    eos: EquationOfState,
}

impl Space {

    pub fn parts(&self) -> &[Part] {
        &self.parts[1..self.num_parts+1]
    }

    pub fn parts_mut(&mut self) -> &mut [Part] {
        &mut self.parts[1..self.num_parts+1]
    }

    /// Constructs a space from given ic's (tuples of position, density, velocity and pressure) and 
    /// boundary conditions.
    pub fn from_ic(ic: &[(f64, f64, f64, f64)], boundary: Boundary, box_size: f64, gamma: f64) -> Self {
        // Initialize equation of state
        let eos = EquationOfState::Ideal { gamma };
        // Get our own mutable copy of the ICs
        let mut ic = ic.to_vec();
        // Wrap particles to be inside box:
        for properties in ic.iter_mut() {
            while properties.0 < 0. { properties.0 += box_size }
            while properties.0 >= box_size { properties.0 -= box_size }
        }

        // create vector of parts for space
        let mut parts = vec![Part::default(); ic.len() + 2];
        // Copy positions and primitives to particles
        for (idx, properties) in ic.iter().enumerate() {
            let part = &mut parts[idx + 1];
            part.x = properties.0;
            part.primitives = Primitives::new(properties.1, properties.2, properties.3);
        } 

        // create space
        let mut space = Space { parts, boundary, box_size, num_parts: ic.len(), eos };
        // sort particles 
        space.sort();
        // Set up the primitive variables of the boundary particles
        space.apply_boundary_condition();
        // Set up the conserved quantities
        space.first_init_parts();

        // return
        space
    }

    fn apply_boundary_condition(&mut self) {
        match self.boundary {
            Boundary::Periodic => {
                self.parts[0] = self.parts[self.num_parts].clone();
                self.parts[0].x -= self.box_size;
                self.parts[self.num_parts+1] = self.parts[1].clone();
                self.parts[self.num_parts+1].x += self.box_size;
            },
            _ => unimplemented!()
        }
    }

    fn apply_boundary_conserved(&mut self) {
        match self.boundary {
            Boundary::Periodic => {
                self.parts[0].conserved = self.parts[self.num_parts].conserved.clone();
                self.parts[0].volume = self.parts[self.num_parts].volume;
                self.parts[self.num_parts+1].conserved = self.parts[1].conserved.clone();
                self.parts[self.num_parts+1].volume = self.parts[1].volume;
            },
            _ => unimplemented!()
        }
    }

    /// Do the volume calculation for all the parts in the space
    pub fn volume_calculation(&mut self) {
        for i in 1..self.num_parts+1 {
            let x_left = self.parts[i-1].x;
            let x_right = self.parts[i+1].x;
            self.parts[i].volume = 0.5 * (x_right - x_left);
            debug_assert!(self.parts[i].volume >= 0.);
        }
    }

    pub fn convert_conserved_to_primitive(&mut self) {
        let eos = self.eos;
        for part in self.parts_mut() {
            part.convert_conserved_to_primitive(eos);
        }

        self.apply_boundary_condition();
    }

    /// Convert the primitive quantities to conserved quantities. This is only done when creating the space from ic's.
    fn first_init_parts(&mut self) {
        self.volume_calculation();

        let eos = self.eos;
        for part in self.parts_mut() {
            part.conserved = Conserved::from_primitives(&part.primitives, part.volume, eos);
        }

        self.apply_boundary_conserved();
    }

    /// Sort the parts in space according to their x coordinate
    pub fn sort(&mut self) {
        self.parts_mut().sort_by(|a, b| {a.x.partial_cmp(&b.x).unwrap()})
    }

    /// Do flux exchange between neighbouring particles
    pub fn flux_exchange(&mut self, solver: &RiemannSolver) {
        for i in 0..self.num_parts + 1 {
            let left = &self.parts[i];
            let right = &self.parts[i+1];
            let dt = left.dt.min(right.dt);
            let dx = 0.5 * (right.x - left.x);

            // Boost the primitives to the frame of reference of the interface:
            let v_face = 0.5 * (left.primitives.velocity() + right.primitives.velocity());
            let primitives_left = left.primitives.boost(-v_face);
            let primitives_right = right.primitives.boost(-v_face);
            let fluxes = solver.solve_for_flux(&primitives_left, &primitives_right, v_face, &self.eos);

            // TODO: gradient extrapolation

            self.parts[i].fluxes -= dt * fluxes;
            self.parts[i].gravity_mflux -= dx * fluxes.mass();
            self.parts[i+1].fluxes += dt * fluxes;
            self.parts[i+1].gravity_mflux += dx * fluxes.mass();
        }

        // Apply fluxes to non-ghost particles
        for part in self.parts_mut() {
            part.apply_flux()
        }
    }

    /// drift all particles foward in time for a full time step
    pub fn drift(&mut self) {
        for part in self.parts_mut() {
            part.drift();
        }

        // Handle particles that left the box.
        let box_size = self.box_size;
        let boundary = self.boundary;
        for part in self.parts_mut() {
            if part.x < 0. || part.x >= box_size {
                match boundary {
                    Boundary::Periodic => {
                        while part.x < 0. { part.x += box_size }
                        while part.x >= box_size { part.x -= box_size }
                    },
                    _ => todo!()
                }
            }
        }
    }

    /// Drift all particles forward in time for a half time step
    pub fn half_drift(&mut self) {
        unimplemented!();
    }

    /// Estimate the gradients for all particles
    pub fn gradient_estimate(&mut self) {
        // TODO
    }

    /// Calculate the next timestep for *all* particles
    pub fn timestep(&mut self, cfl_criterion: f64) -> f64 {
        let mut min_dt = f64::MAX;
        for part in self.parts.iter_mut() {
            min_dt = min_dt.min(part.timestep(cfl_criterion));
        }
        min_dt
    }

    /// Apply the first half kick to the particles
    pub fn kick1(&mut self) {
        // TODO
    }

    /// Apply the second half kick to the particles
    pub fn kick2(&mut self) {
        // TODO
    }

    /// Dump snapshot of space at the current time
    pub fn dump(&mut self, f: &mut BufWriter<File>) -> Result<(), IoError> {
        writeln!(f, "# x (m)\trho (kg m^-3)\tv (m s^-1)\tP (kg m^-1 s^-2)\ta (m s^-2)\tu (J / kg)")?;
        for part in self.parts() {
            writeln!(f, "{}\t{}\t{}\t{}\t{}\t{}", part.x, part.primitives.density(), part.primitives.velocity(), part.primitives.pressure(), part.a_grav, part.internal_energy())?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::{Space, Boundary};

    const IC: [(f64, f64, f64, f64); 4] = [(0.25, 1., 0.5, 1.), (0.65, 0.125, 0.5, 0.1), 
                                           (0.75, 0.125, 0.5, 0.1), (0.85, 0.125, 0.5, 0.1)];
    const BOX_SIZE: f64 = 1.;
    const GAMMA: f64 = 5. / 3.;

    #[test]
    fn test_init() {
        let space = Space::from_ic(&IC, Boundary::Periodic, BOX_SIZE, GAMMA);
        assert_eq!(space.parts.len(), 6);
        assert_eq!(space.num_parts, 4);
        // Check volumes
        assert_eq!(space.parts[0].volume, 0.25);
        assert_eq!(space.parts[1].volume, 0.4);
        assert_eq!(space.parts[2].volume, 0.25);
        assert_eq!(space.parts[3].volume, 0.1);
        assert_eq!(space.parts[4].volume, 0.25);
        assert_eq!(space.parts[6].volume, 0.4);
    }

    #[test]
    fn test_drift() {
        todo!();
    }


}