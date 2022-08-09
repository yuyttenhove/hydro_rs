use crate::equation_of_state::EquationOfState;
pub use crate::physical_quantities::{Primitives, Conserved};

#[derive(Default, Debug, Clone)]
pub struct Part {
    pub primitives: Primitives,
    pub gradients: Primitives,
    pub conserved: Conserved,
    pub fluxes: Conserved,
    pub gravity_mflux: f64,

    pub volume: f64,
    pub x: f64,
    pub dt: f64,

    pub a_grav: f64,
}

impl Part {
    pub fn timestep(&mut self, cfl_criterion: f64) -> f64 {
        self.dt = 1e-4;
        self.dt
    }

    pub fn drift(&mut self, eos: &EquationOfState) {
        self.x += self.primitives.velocity() * self.dt;

        // Extrapolate primitives in time
        if let EquationOfState::Ideal { gamma } = eos {
            let half_dt = 0.5 * self.dt;
            let rho = self.primitives.density();
            let rho_inv = 1. / rho;
            let v = self.primitives.velocity();
            let p = self.primitives.pressure();
            self.primitives -= half_dt * Primitives::new(
                rho * self.gradients.velocity() + v * self.gradients.density(), 
                v * self.gradients.velocity() + rho_inv * self.gradients.pressure(), 
                gamma * p * self.gradients.velocity() + v * self.gradients.pressure()
            )
        } else {
            unimplemented!()
        }
    }

    pub fn apply_flux(&mut self) {
        self.conserved += self.fluxes;
        self.fluxes = Conserved::vacuum();
    }

    pub fn convert_conserved_to_primitive(&mut self, eos: EquationOfState) {
        self.primitives = Primitives::from_conserved(&self.conserved, self.volume, eos)
    }

    pub fn internal_energy(&self) -> f64 {
        (self.conserved.energy() - 0.5 * self.conserved.momentum() * self.primitives.velocity()) / self.conserved.mass()
    }
}