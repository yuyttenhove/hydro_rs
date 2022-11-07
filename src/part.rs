pub use crate::physical_quantities::{Conserved, Primitives};
use crate::{
    engine::{Engine, ParticleMotion},
    equation_of_state::EquationOfState,
    timeline::*,
};

#[derive(Default, Debug, Clone)]
pub struct Part {
    pub primitives: Primitives,
    pub gradients: Primitives,
    pub extrapolations: Primitives,
    pub conserved: Conserved,
    pub fluxes: Conserved,
    pub gravity_mflux: f64,

    pub volume: f64,
    pub x: f64,
    pub centroid: f64,
    pub v: f64,
    pub timebin: Timebin,
    pub dt: f64,

    pub a_grav: f64,
}

impl Part {
    pub fn timestep(
        &mut self,
        cfl_criterion: f64,
        eos: &EquationOfState,
        particle_motion: &ParticleMotion,
    ) -> f64 {

        if self.conserved.mass() == 0. {
            // We have vacuum
            debug_assert_eq!(self.conserved.momentum(), 0.);
            debug_assert_eq!(self.conserved.energy(), 0.);
            self.v = 0.;
            return f64::INFINITY;
        }

        // Normal case
        debug_assert!(self.conserved.mass() > 0.);
        assert!((self.conserved.mass() / self.volume - self.primitives.density()).abs() <= self.primitives.density().abs() * 1e-8);
        assert!((self.conserved.momentum() / self.conserved.mass() - self.primitives.velocity()).abs() <= self.primitives.velocity().abs() * 1e-8);
        let internal_energy = self.conserved.energy() - 0.5 * self.conserved.momentum() * self.primitives.velocity();
        assert!((eos.gas_pressure_from_internal_energy(internal_energy, self.volume) - self.primitives.pressure()).abs() <= self.primitives.pressure().abs() * 1e-8);

        let mass_inv = 1. / self.conserved.mass();
        let fluid_v = self.conserved.momentum() * mass_inv;
        let sound_speed = eos.sound_speed(
            eos.gas_pressure_from_internal_energy(internal_energy, self.volume),
            self.volume * mass_inv,
        );
        // Set the velocity with which this particle will be drifted over the course of it's next timestep
        self.v = match particle_motion {
            ParticleMotion::FIXED => 0.,
            ParticleMotion::FLUID => fluid_v,
            ParticleMotion::STEER => {
                let d = self.centroid - self.x;
                let abs_d = d.abs();
                let r = 0.5 * self.volume;
                let eta = 0.25;
                let eta_r = eta * r;
                let xi = 1.0;
                if abs_d > 0.9 * eta_r {
                    let mut fac = xi * sound_speed / abs_d;
                    if abs_d < 1.1 * eta_r {
                        fac *= 5. * (abs_d - 0.9 * eta_r) / eta_r;
                    }
                    fluid_v + fac * d
                } else {
                    fluid_v
                }
            }
        };
        assert!(self.v.is_finite(), "Invalid value for v!");

        // determine the size of this particle's next timestep
        let v_rel = (self.v - fluid_v).abs();
        let v_max = v_rel + sound_speed;

        if v_max > 0. {
            0.5 * cfl_criterion * self.volume / v_max
        } else {
            f64::INFINITY
        }
    }

    /// Drifts the particle forward in time over a time `dt_drift`.
    ///
    /// Also predicts the primitive quantities forward in time over a time dt_extrapolate using the Euler equations.
    pub fn drift(&mut self, dt_drift: f64, dt_extrapolate: f64, eos: &EquationOfState) {
        self.x += self.v * dt_drift;

        debug_assert!(self.x.is_finite(), "Infinite x after drift!");

        // Extrapolate primitives in time
        if let EquationOfState::Ideal { gamma } = eos {
            let rho = self.primitives.density();
            if rho > 0. {
                let rho_inv = 1. / rho;
                let v = self.primitives.velocity();
                let p = self.primitives.pressure();
                self.extrapolations -= dt_extrapolate
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

        // if self.primitives.density() < 1e-6 {
        //     self.primitives = Primitives::new(
        //         self.primitives.density(),
        //         self.primitives.velocity() * self.primitives.density() * 1e6,
        //         self.primitives.pressure(),
        //     );
        //     self.conserved = Conserved::from_primitives(&self.primitives, self.volume, eos);
        // }

        assert!(self.primitives.density() >= 0.);
        assert!(self.primitives.pressure() >= 0.);

        debug_assert!(
            self.primitives.density().is_finite(),
            "Infinite density detected!"
        );
        debug_assert!(
            self.primitives.velocity().is_finite(),
            "Infinite velocity detected!"
        );
        debug_assert!(
            self.primitives.pressure().is_finite(),
            "Infinite pressure detected!"
        );
    }

    pub fn timestep_limit(&mut self, new_bin: Timebin, engine: &Engine) {
        self.timebin = new_bin;

        // TODO: Rewind kick1 if necessary
        // TODO: Reapply kick1 if necessary
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
