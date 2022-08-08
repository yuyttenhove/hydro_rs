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

    pub fn drift(&mut self) {
        self.x += self.primitives.velocity() * self.dt;
        // TODO: primitive extrapolation using gradients
    }

    pub fn apply_flux(&mut self) {
        self.conserved += self.fluxes;
        self.fluxes = Conserved::vacuum();
    }

    pub fn convert_conserved_to_primitive(&mut self, eos: EquationOfState) {
        self.primitives = Primitives::from_conserved(&self.conserved, self.volume, eos)
    }
}