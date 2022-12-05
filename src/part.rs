use crate::physical_quantities::{Conserved, Primitives, Vec3f64};
#[cfg(any(dimensionality = "2D", dimensionality = "3D"))]
use crate::spherical::ALPHA;
use crate::{
    engine::{Engine, ParticleMotion},
    equation_of_state::EquationOfState,
    time_integration::Runner,
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
    pub wakeup: Timebin,
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
        // assert!(self.conserved.mass() > 0.);
        // assert!((self.conserved.mass() / self.volume - self.primitives.density()).abs() <= self.primitives.density().abs() * 2e-2);
        // assert!((self.conserved.momentum() / self.conserved.mass() - self.primitives.velocity()).abs() <= self.primitives.velocity().abs() * 2e-2);
        let m_inv = 1. / self.conserved.mass();
        let internal_energy = (self.conserved.energy()
            - 0.5 * self.conserved.momentum() * self.primitives.velocity())
            * m_inv;
        // assert!(internal_energy > 0.);
        // assert!((eos.gas_pressure_from_internal_energy(internal_energy, self.volume) - self.primitives.pressure()).abs() <= self.primitives.pressure().abs() * 2e-2);

        let fluid_v = self.conserved.momentum() * m_inv;
        let sound_speed = eos.sound_speed(
            eos.gas_pressure_from_internal_energy(internal_energy, self.primitives.density()),
            self.physical_volume() * m_inv,
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

        if self.conserved.mass() < 0. {
            eprintln!("Negative mass after applying fluxes");
            self.conserved = Conserved::vacuum();
        }
        if self.conserved.energy() < 0. {
            eprintln!("Negative energy after applying fluxes");
            self.conserved = Conserved::vacuum();
        }
    }

    pub fn reset_fluxes(&mut self) {
        self.fluxes = Conserved::vacuum();
        self.gravity_mflux = 0.;
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
            Primitives::from_conserved(&self.conserved, self.physical_volume(), eos)
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

    pub fn timestep_limit(&mut self, engine: &Engine) {
        // Anything to do here?
        if self.wakeup >= self.timebin {
            return;
        }

        if self.is_active(engine) {
            // Particle was active/starting anyway, so just updating the timestep/timebin suffices.
            self.timebin = self.wakeup;
            self.dt = make_timestep(get_integer_timestep(self.timebin), engine.time_base());
        } else {
            // Wake this particle up
            let ti_current = engine.ti_current();
            let ti_end_old = get_integer_time_end(ti_current, self.timebin);
            debug_assert_ne!(ti_end_old, ti_current);
            // Substract the remainder of this particles old timestep from its dt
            self.dt -= make_timestep(ti_end_old - ti_current, engine.time_base());
            // Update the timebin
            self.timebin = self.wakeup;
            let dti_new = get_integer_timestep(self.timebin);
            let ti_end = get_integer_time_end(ti_current, self.timebin);
            // Add the remainder of the new timestep to the particle's dt
            self.dt += if ti_end == ti_current {
                make_timestep(dti_new, engine.time_base())
                // Part is now active/starting, so the remaining KICK1 will be applied automatically
            } else {
                make_timestep(ti_end - ti_current, engine.time_base())
                // TODO: reapply second part of KICK1 if neccessary
            };

            // TODO this might still not be correct for particles that have a longer timestep then this particles new timestep, but shorter than this particles old timestep
        }

        // TODO: Rewind kick1 if necessary
        // TODO: Reapply kick1 if necessary
    }

    /// Add the spherical source terms.
    ///
    /// See Toro, 2009, chapter 17.
    /// We use a second order Runge-Kutta step and apply an operator splitting method
    /// to couple the source term to the hydro step.
    #[cfg(any(dimensionality = "2D", dimensionality = "3D"))]
    pub fn add_spherical_source_term(&mut self, eos: &EquationOfState) {
        if self.conserved.mass() == 0. {
            return;
        }

        let r = self.centroid;
        let r_inv = ALPHA / r;
        let vol = self.physical_volume();
        let vol_inv = 1. / vol;
        let u = (vol_inv * self.conserved).values();

        let u0_inv = 1. / u.0;
        let u1_2 = u.1 * u.1;
        let internal_energy = u.2 - 0.5 * u1_2 * u0_inv; // rho e
        let p1 = eos.gas_pressure_from_internal_energy(internal_energy * u0_inv, u.0);
        let k1 = -self.dt * r_inv * Vec3f64(u.1, u1_2 * u0_inv, u.1 * u0_inv * (u.2 + p1));

        let u_prime = u + k1;
        let u_prime0_inv = 1. / u_prime.0;
        let u_prime1_2 = u_prime.1 * u_prime.1;
        let internal_energy = u_prime.2 - 0.5 * u_prime1_2 * u_prime0_inv;
        let p2 = eos.gas_pressure_from_internal_energy(internal_energy, u_prime.0);
        let k2 = -self.dt
            * r_inv
            * Vec3f64(
                u_prime.1,
                u_prime1_2 * u_prime0_inv,
                u_prime.1 * u_prime0_inv * (u_prime.2 + p2),
            );

        let u = vol * (u + 0.5 * (k1 + k2));

        self.conserved = Conserved::new(u.0, u.1, u.2);
    }

    pub fn grav_kick(&mut self) {
        let mass = self.conserved.mass();
        let momentum = self.conserved.momentum();
        let grav_kick_factor = 0.5 * self.dt * self.a_grav;
        self.conserved += Conserved::new(
            0.,
            grav_kick_factor * mass,
            grav_kick_factor * (momentum - self.gravity_mflux),
        );
    }

    pub fn internal_energy(&self) -> f64 {
        (self.conserved.energy() - 0.5 * self.conserved.momentum() * self.primitives.velocity())
            / self.conserved.mass()
    }

    pub fn is_ending(&self, engine: &Engine) -> bool {
        return self.timebin <= get_max_active_bin(engine.ti_current());
    }

    fn is_halfway(&self, engine: &Engine) -> bool {
        let dti = engine.ti_current() - engine.ti_old();
        return !self.is_ending(engine)
            && self.timebin <= get_max_active_bin(engine.ti_old() + 2 * dti);
    }

    pub fn is_active_flux(&self, engine: &Engine) -> bool {
        match engine.runner() {
            Runner::OptimalOrderHalfDrift | Runner::DefaultHalfDrift => self.is_halfway(engine),
            _ => self.is_ending(engine),
        }
    }

    pub fn is_active_primitive_calculation(&self, engine: &Engine) -> bool {
        match engine.runner() {
            Runner::DefaultHalfDrift => self.is_halfway(engine),
            _ => self.is_ending(engine),
        }
    }

    pub fn is_active(&self, engine: &Engine) -> bool {
        self.is_ending(engine)
    }

    pub fn set_timebin(&mut self, new_dti: IntegerTime) {
        self.timebin = get_time_bin(new_dti);
    }

    pub fn physical_volume(&self) -> f64 {
        let x_left = self.centroid - 0.5 * self.volume;
        let x_right = self.centroid + 0.5 * self.volume;
        if cfg!(dimensionality = "2D") {
            std::f64::consts::PI * (x_right * x_right - x_left * x_left)
        } else if cfg!(dimensionality = "3D") {
            4. * std::f64::consts::FRAC_PI_3 * (x_right.powi(3) - x_left.powi(3))
        } else {
            self.volume
        }
    }

    pub fn half_physical_volume(&self) -> f64 {
        let x_left = self.centroid - 0.5 * self.volume;
        let r = self.centroid;
        if cfg!(dimensionality = "2D") {
            std::f64::consts::PI * (r * r - x_left * x_left)
        } else if cfg!(dimensionality = "3D") {
            4. * std::f64::consts::FRAC_PI_3 * (r.powi(3) - x_left.powi(3))
        } else {
            self.volume
        }
    }

    pub fn reflect(&mut self, around: f64) -> &mut Self {
        self.x += 2. * (around - self.x);
        self.centroid += 2. * (around - self.centroid);
        self.v *= -1.;

        self
    }

    pub fn reflect_quantities(&mut self) -> &mut Self {
        self.primitives = Primitives::new(
            self.primitives.density(),
            -self.primitives.velocity(),
            self.primitives.pressure(),
        );
        self.gradients = Primitives::new(
            -self.gradients.density(),
            self.gradients.velocity(),
            -self.gradients.pressure(),
        );
        self.conserved = Conserved::new(
            self.conserved.mass(),
            -self.conserved.momentum(),
            self.conserved.energy(),
        );

        self
    }
}
