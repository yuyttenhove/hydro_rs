pub use crate::physical_quantities::{Conserved, Primitives};
use crate::{engine::Engine, equation_of_state::EquationOfState, timeline::*};

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

    /// Drifts the particle forward in time over a time `dt_drift`.
    ///
    /// Also predicts the primitive quantities forward in time over a time dt_extrapolate using the Euler equations.
    pub fn drift(&mut self, dt_drift: f64, dt_extrapolate: f64, eos: &EquationOfState) {
        self.x += self.primitives.velocity() * dt_drift;

        debug_assert!(self.x.is_finite(), "Infinite x after drift!");

        // Extrapolate primitives in time
        if let EquationOfState::Ideal { gamma } = eos {
            let rho = self.primitives.density();
            if rho > 0. {
                let rho_inv = 1. / rho;
                let v = self.primitives.velocity();
                let p = self.primitives.pressure();
                self.primitives -= dt_extrapolate
                    * Primitives::new(
                    rho * self.gradients.velocity() + v * self.gradients.density(),
                    v * self.gradients.velocity() + rho_inv * self.gradients.pressure(),
                    gamma * p * self.gradients.velocity() + v * self.gradients.pressure(),
                )
            }
        } else {
            unimplemented!()
        }

        self.primitives.check_physical();
    }

    pub fn apply_flux(&mut self) {
        self.conserved += self.fluxes;
        self.fluxes = Conserved::vacuum();

        if self.conserved.mass() < 0. {
            eprintln!("Negative mass after applying fluxes");
            self.conserved = Conserved::vacuum();
        }
        if self.conserved.energy() < 0. {
            eprintln!("Negative energy after applying fluxes");
            self.conserved = Conserved::vacuum();
        }
    }

    pub fn convert_conserved_to_primitive(&mut self, eos: EquationOfState) {
        debug_assert!(self.conserved.mass() >= 0., "Encountered negative mass!");
        debug_assert!(
            self.conserved.energy() >= 0.,
            "Encountered negative energy!"
        );

        self.primitives = if self.conserved.mass() == 0. {
            debug_assert_eq!(
                self.conserved.energy(),
                0.,
                "Zero mass, indicating vacuum, but energy != 0!"
            );
            Primitives::vacuum()
        } else {
            Primitives::from_conserved(&self.conserved, self.volume, eos)
        };

        debug_assert!(self.primitives.density().is_finite(), "Infinite density detected!");
        debug_assert!(self.primitives.velocity().is_finite(), "Infinite velocity detected!");
        debug_assert!(self.primitives.pressure().is_finite(), "Infinite pressure detected!");
    }

    pub fn internal_energy(&self) -> f64 {
        (self.conserved.energy() - 0.5 * self.conserved.momentum() * self.primitives.velocity())
            / self.conserved.mass()
    }

    pub fn is_active(&self, engine: &Engine) -> bool {
        return self.timebin <= get_max_active_bin(engine.ti_current());
    }

    pub fn set_timebin(&mut self, new_dti: IntegerTime) {
        self.timebin = get_time_bin(new_dti);
    }
}
