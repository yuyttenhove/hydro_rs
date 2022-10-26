use std::{
    fs::File,
    io::{BufWriter, Error as IoError, Write},
};

use crate::{
    engine::Engine,
    equation_of_state::EquationOfState,
    part::{Conserved, Part, Primitives},
    slope_limiters::{cell_wide_limiter, pairwise_limiter},
    timeline::{
        get_integer_time_end, make_integer_timestep, make_timestep, IntegerTime, MAX_NR_TIMESTEPS,
        NUM_TIME_BINS,
    },
};

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
        &self.parts[1..self.num_parts + 1]
    }

    pub fn parts_mut(&mut self) -> &mut [Part] {
        &mut self.parts[1..self.num_parts + 1]
    }

    /// Constructs a space from given ic's (tuples of position, density, velocity and pressure) and
    /// boundary conditions.
    pub fn from_ic(
        ic: &[(f64, f64, f64, f64)],
        boundary: Boundary,
        box_size: f64,
        gamma: f64,
    ) -> Self {
        // Initialize equation of state
        let eos = EquationOfState::Ideal { gamma };
        // Get our own mutable copy of the ICs
        let mut ic = ic.to_vec();
        // Wrap particles to be inside box:
        for properties in ic.iter_mut() {
            while properties.0 < 0. {
                properties.0 += box_size
            }
            while properties.0 >= box_size {
                properties.0 -= box_size
            }
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
        let mut space = Space {
            parts,
            boundary,
            box_size,
            num_parts: ic.len(),
            eos,
        };
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
                self.parts[self.num_parts + 1] = self.parts[1].clone();
                self.parts[self.num_parts + 1].x += self.box_size;
            }
            _ => unimplemented!(),
        }
    }

    fn apply_boundary_conserved(&mut self) {
        match self.boundary {
            Boundary::Periodic => {
                self.parts[0].conserved = self.parts[self.num_parts].conserved.clone();
                self.parts[0].volume = self.parts[self.num_parts].volume;
                self.parts[self.num_parts + 1].conserved = self.parts[1].conserved.clone();
                self.parts[self.num_parts + 1].volume = self.parts[1].volume;
            }
            _ => unimplemented!(),
        }
    }

    /// Do the volume calculation for all the active parts in the space
    pub fn volume_calculation(&mut self, engine: &Engine) {
        for i in 1..self.num_parts + 1 {
            let part = &self.parts[i];
            if !part.is_active(engine) {
                continue;
            }
            let x_left = self.parts[i - 1].x;
            let x_right = self.parts[i + 1].x;
            let part = &mut self.parts[i];
            part.volume = 0.5 * (x_right - x_left);
            debug_assert!(self.parts[i].volume >= 0.);
        }
    }

    pub fn convert_conserved_to_primitive(&mut self, engine: &Engine) {
        let eos = self.eos;
        for part in self.parts_mut() {
            if !part.is_active(engine) {
                continue;
            }
            part.convert_conserved_to_primitive(eos);
        }

        self.apply_boundary_condition();
    }

    /// Convert the primitive quantities to conserved quantities. This is only done when creating the space from ic's.
    fn first_init_parts(&mut self) {
        // Calculate the volume of *all* particles
        for i in 1..self.num_parts + 1 {
            let x_left = self.parts[i - 1].x;
            let x_right = self.parts[i + 1].x;
            let part = &mut self.parts[i];
            part.volume = 0.5 * (x_right - x_left);
            debug_assert!(self.parts[i].volume >= 0.);
        }

        // Calculate the conserved quantities
        let eos = self.eos;
        for part in self.parts_mut() {
            part.conserved = Conserved::from_primitives(&part.primitives, part.volume, eos);
        }

        self.apply_boundary_conserved();
    }

    /// Sort the parts in space according to their x coordinate
    pub fn sort(&mut self) {
        self.parts_mut()
            .sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap())
    }

    /// Do flux exchange between neighbouring particles
    pub fn flux_exchange(&mut self, engine: &Engine) {
        for i in 0..self.num_parts + 1 {
            let left = &self.parts[i];
            let right = &self.parts[i + 1];

            // anything to do here?
            let left_active = left.is_active(engine);
            let right_active = right.is_active(engine);
            if !left_active && !right_active {
                continue;
            }

            // At least one of the particles is active: Update the fluxes of both (symmetrically)
            let dt = left.dt.min(right.dt);
            let dx = 0.5 * (right.x - left.x);

            // Gradient extrapolation
            let mut primitives_left = pairwise_limiter(
                left.primitives,
                right.primitives,
                left.primitives + dx * left.gradients,
            );
            let mut primitives_right = pairwise_limiter(
                right.primitives,
                left.primitives,
                right.primitives - dx * right.gradients,
            );

            // Boost the primitives to the frame of reference of the interface:
            let v_face = 0.5 * (left.primitives.velocity() + right.primitives.velocity());
            primitives_left = primitives_left.boost(-v_face);
            primitives_right = primitives_right.boost(-v_face);
            let fluxes = engine.solver.solve_for_flux(
                &primitives_left,
                &primitives_right,
                v_face,
                &self.eos,
            );

            // update accumulated fluxes
            self.parts[i].fluxes -= dt * fluxes;
            self.parts[i].gravity_mflux -= dx * fluxes.mass();

            self.parts[i + 1].fluxes += dt * fluxes;
            self.parts[i + 1].gravity_mflux += dx * fluxes.mass();
        }
    }

    /// Apply the accumulated fluxes to all active particles
    pub fn flux_apply(&mut self, engine: &Engine) {
        for part in self.parts_mut() {
            if part.is_active(engine) {
                part.apply_flux();
            }
        }
    }

    /// drift all particles foward over the given timestep
    pub fn drift(&mut self, dt_drift: f64, dt_extrapolate: f64) {
        let eos = self.eos;
        for part in self.parts_mut() {
            part.drift(dt_drift, dt_extrapolate, &eos);
        }

        // Handle particles that left the box.
        let box_size = self.box_size;
        let boundary = self.boundary;
        for part in self.parts_mut() {
            if part.x < 0. || part.x >= box_size {
                match boundary {
                    Boundary::Periodic => {
                        while part.x < 0. {
                            part.x += box_size
                        }
                        while part.x >= box_size {
                            part.x -= box_size
                        }
                    }
                    _ => todo!(),
                }
            }
        }
    }

    /// Estimate the gradients for all particles
    pub fn gradient_estimate(&mut self, engine: &Engine) {
        for i in 1..self.num_parts + 1 {
            let part = &self.parts[i];
            if !part.is_active(engine) {
                continue;
            }

            self.parts[i].gradients = {
                let left = &self.parts[i - 1];
                let right = &self.parts[i + 1];
                let dx = right.x - left.x;
                let dx_inv = 1. / dx;
                let dx_left = 0.5 * (left.x - part.x);
                let dx_right = 0.5 * (right.x - part.x);

                let gradients = dx_inv * (right.primitives - left.primitives);

                cell_wide_limiter(
                    gradients,
                    part.primitives,
                    left.primitives,
                    right.primitives,
                    dx_left,
                    dx_right,
                )
            };
        }

        // Make sure the gradients are applied to the boundary particles
        self.apply_boundary_condition();
    }

    /// Calculate the next timestep for all active particles
    pub fn timestep(&mut self, engine: &Engine) -> IntegerTime {
        // Some useful variables
        let ti_current = engine.ti_current();
        let mut ti_end_min = MAX_NR_TIMESTEPS;

        for part in self.parts.iter_mut() {
            if part.is_active(engine) {
                // Compute new timestep
                let mut dt = part.timestep(engine.cfl_criterion(), &self.eos);
                dt = dt.min(engine.dt_max());
                assert!(
                    dt > engine.dt_min(),
                    "Error: particle requested dt ({}) below dt_min ({})",
                    dt,
                    engine.dt_min()
                );
                let dti = make_integer_timestep(
                    dt,
                    part.timebin,
                    /*TODO*/ NUM_TIME_BINS,
                    ti_current,
                    engine.time_base_inv(),
                );
                part.dt = make_timestep(dti, engine.time_base());
                part.set_timebin(dti);

                ti_end_min = ti_end_min.min(ti_current + dti);
            } else {
                ti_end_min = ti_end_min.min(get_integer_time_end(ti_current, part.timebin));
            }
        }

        debug_assert!(
            ti_end_min > ti_current,
            "Next sync point before current time!"
        );
        ti_end_min
    }

    /// Apply the first half kick (gravity) to the particles
    pub fn kick1(&mut self, engine: &Engine) {
        // TODO
    }

    /// Apply the second half kick (gravity) to the particles
    pub fn kick2(&mut self, engine: &Engine) {
        // TODO
    }

    /// Dump snapshot of space at the current time
    pub fn dump(&mut self, f: &mut BufWriter<File>) -> Result<(), IoError> {
        writeln!(
            f,
            "# x (m)\trho (kg m^-3)\tv (m s^-1)\tP (kg m^-1 s^-2)\ta (m s^-2)\tu (J / kg)\tS"
        )?;
        for part in self.parts() {
            let internal_energy = part.internal_energy();
            let density = part.primitives.density();
            let entropy = self
                .eos
                .gas_entropy_from_internal_energy(internal_energy, density);
            writeln!(
                f,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}",
                part.x,
                density,
                part.primitives.velocity(),
                part.primitives.pressure(),
                part.a_grav,
                internal_energy,
                entropy
            )?;
        }

        Ok(())
    }

    pub fn self_check(&self) {
        for part in self.parts() {
            debug_assert!(part.x.is_finite());
            debug_assert!(part.primitives.density().is_finite());
            debug_assert!(part.primitives.velocity().is_finite());
            debug_assert!(part.primitives.pressure().is_finite());
            debug_assert!(part.conserved.mass().is_finite());
            debug_assert!(part.conserved.momentum().is_finite());
            debug_assert!(part.conserved.energy().is_finite());
            debug_assert!(part.fluxes.mass().is_finite());
            debug_assert!(part.fluxes.momentum().is_finite());
            debug_assert!(part.fluxes.energy().is_finite());
            debug_assert!(part.conserved.mass() >= 0.);
            debug_assert!(part.conserved.energy() >= 0.);
        }
    }
}

#[cfg(test)]
mod test {
    use assert_approx_eq::assert_approx_eq;

    use super::{Boundary, Space};

    const IC: [(f64, f64, f64, f64); 4] = [
        (0.25, 1., 0.5, 1.),
        (0.65, 0.125, 0.5, 0.1),
        (0.75, 0.125, 0.5, 0.1),
        (0.85, 0.125, 0.5, 0.1),
    ];
    const BOX_SIZE: f64 = 1.;
    const GAMMA: f64 = 5. / 3.;

    #[test]
    fn test_init() {
        let space = Space::from_ic(&IC, Boundary::Periodic, BOX_SIZE, GAMMA);
        assert_eq!(space.parts.len(), 6);
        assert_eq!(space.num_parts, 4);
        // Check volumes
        assert_approx_eq!(space.parts[0].volume, 0.25, space.parts[0].volume * 1e-8);
        assert_approx_eq!(space.parts[1].volume, 0.4, space.parts[1].volume * 1e-8);
        assert_approx_eq!(space.parts[2].volume, 0.25, space.parts[2].volume * 1e-8);
        assert_approx_eq!(space.parts[3].volume, 0.1, space.parts[3].volume * 1e-8);
        assert_approx_eq!(space.parts[4].volume, 0.25, space.parts[4].volume * 1e-8);
        assert_approx_eq!(space.parts[5].volume, 0.4, space.parts[5].volume * 1e-8);
    }
}
