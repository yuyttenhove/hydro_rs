use crate::{equation_of_state::EquationOfState, timeline::*, engine::Engine};
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
    pub timebin: Timebin,
    pub dt: f64,

    pub a_grav: f64,
}

impl Part {
    pub fn timestep(&mut self, cfl_criterion: f64, eos: &EquationOfState) -> f64 {
        let v_max = eos.sound_speed(self.primitives.pressure(), 1. / self.primitives.density());

        if v_max > 0. {
            cfl_criterion * self.volume / v_max
        } else {
            f64::INFINITY
        }
    }

    pub fn drift(&mut self, dt: f64, eos: &EquationOfState) {
        self.x += self.primitives.velocity() * dt;

        // Extrapolate primitives in time
        if let EquationOfState::Ideal { gamma } = eos {
            let half_dt = 0.5 * dt;
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

    pub fn is_active(&self, engine: &Engine) -> bool {
        return self.timebin <= get_max_active_bin(engine.ti_current())
    }

    pub fn set_timebin(&mut self, new_dti: IntegerTime) {
        self.timebin = get_time_bin(new_dti);
    }
}