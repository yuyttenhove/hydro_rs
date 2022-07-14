use std::process::id;

use crate::{part::{Part, Primitives, Conserved}, equation_of_state::EquationOfState, riemann_solver::RiemannSolver};

pub enum Boundary {
    periodic,
    reflective,
    open,
    vacuum,
}

pub struct Space {
    parts: Vec<Part>,
    boundary: Boundary,
    box_size: f64,
    num_parts: usize,
}

impl Space {
    /// Constructs a space from given ic's (tuples of position, density, velocity and pressure) and 
    /// boundary conditions.
    pub fn from_ic(ic: &[(f64, f64, f64, f64)], boundary: Boundary, box_size: f64, eos: &EquationOfState) -> Self {
        // Get our own mutable copy of the ICs
        let mut ic = ic.to_vec();
        // Wrap particles to be inside box:
        for properties in ic.iter_mut() {
            while properties.0 < 0. { properties.0 += box_size }
            while properties.0 >= box_size { properties.0 -= box_size }
        }
        // Sort by x coordinate 
        ic.sort_by(|a, b| {a.0.partial_cmp(&b.0).unwrap()});

        // create vector of parts for space
        let mut parts = vec![Part::default(); ic.len() + 2];

        // Copy positions and primitives to particles
        for (idx, properties) in ic.iter().enumerate() {
            let part = &mut parts[idx + 1];
            part.x = properties.0;
            part.primitives = Primitives::new(properties.1, properties.2, properties.3);
        } 

        // create space
        let mut space = Space { parts, boundary, box_size, num_parts: ic.len() };
        
        // Set up the primitive variables of the boundary particles
        space.apply_boundary_primitives();
        // Set up the conserved quantities
        space.first_init_parts(eos);
        // Set up the conserved quantities for the boundary particles 
        space.apply_boundary_conserved();

        // return
        space
    }

    fn apply_boundary_primitives(&mut self) {
        match self.boundary {
            Boundary::periodic => {
                self.parts[0] = self.parts[self.num_parts - 1].clone();
                self.parts[0].x -= self.box_size;
                self.parts[self.num_parts] = self.parts[1].clone();
                self.parts[self.num_parts].x += self.box_size;
            },
            _ => todo!()
        }
    }

    fn apply_boundary_conserved(&mut self) {
        match self.boundary {
            Boundary::periodic => {
                self.parts[0].conserved = self.parts[self.num_parts - 1].conserved.clone();
                self.parts[0].volume = self.parts[self.num_parts - 1].volume;
                self.parts[self.num_parts].conserved = self.parts[1].conserved.clone();
                self.parts[self.num_parts].volume = self.parts[1].volume;
            },
            _ => todo!()
        }
    }

    /// Do the volume calculation for all the parts in the space
    pub fn update_volumes(&mut self) {
        for i in 1..self.num_parts+1 {
            let x_left = self.parts[i-1].x;
            let x_right = self.parts[i+1].x;
            self.parts[i].volume = 0.5 * (x_right - x_left);
        }
    }

    /// Convert the primitive quantities to conserved quantities. This is only done when creating the space from ic's.
    fn first_init_parts(&mut self, eos: &EquationOfState) {
        self.update_volumes();

        for part in self.parts.iter_mut() {
            part.conserved = Conserved::from_primitives(&part.primitives, part.volume, eos);
        }
    }

    /// Sort the parts in space according to their x coordinate
    pub fn sort(&mut self) {
        self.parts[1..self.num_parts+1].sort_by(|a, b| {a.x.partial_cmp(&b.x).unwrap()})
    }

    /// Do flux exchange between neighbouring particles
    pub fn flux_exchange(&mut self, solver: &RiemannSolver) {
        for i in 0..self.num_parts + 1 {
            let left = &self.parts[i];
            let right = &self.parts[i+1];
            let dt = left.dt.min(right.dt);
            let dx = right.x - left.x;
            let fluxes = solver.solve_for_flux(&left.primitives, &right.primitives);

            // TODO: gradient extrapolation

            self.parts[i].fluxes -= dt * fluxes;
            self.parts[i].gravity_mflux -= dx * fluxes.mass();
            self.parts[i+1].fluxes += dt * fluxes;
            self.parts[i+1].gravity_mflux += dx * fluxes.mass();
        }
    }

    /// drift all particles foward in time
    pub fn drift(&mut self) {
        for part in self.parts[1..self.num_parts+1].iter_mut() {
            part.x += part.primitives.velocity() * part.dt
            // TODO: primitive extrapolation using gradients
        }
    }

    /// Estimate the gradients for all particles
    pub fn gradient_estimate(&mut self) {
        todo!()
    }

    /// Calculate the next timestep for all particles
    pub fn timestep(&mut self, cfl_criterion: f64) {
        todo!()
    }

    /// Apply the first half kick to the particles
    pub fn kick1(&mut self) {
        todo!()
    }

    /// Apply the second half kick to the particles
    pub fn kick2(&mut self) {
        todo!()
    }
}