use glam::{DMat3, DVec3};

use crate::{
    engine::{Engine, ParticleMotion},
    equation_of_state::EquationOfState,
    time_integration::Runner,
    timeline::*, utils::{HydroDimension, HydroDimension::*},
};
use crate::{
    physical_quantities::{Conserved, Primitives, StateGradients, StateVector},
    slope_limiters::cell_wide_limiter,
};

#[derive(Debug, Clone, Copy)]
pub struct LimiterData {
    pub min: Primitives,
    pub max: Primitives,
    pub e_min: Primitives,
    pub e_max: Primitives,
}

impl Default for LimiterData {
    fn default() -> Self {
        Self {
            min: StateVector::splat(f64::INFINITY).into(),
            max: StateVector::splat(f64::NEG_INFINITY).into(),
            e_min: StateVector::splat(f64::INFINITY).into(),
            e_max: StateVector::splat(f64::NEG_INFINITY).into(),
        }
    }
}

#[derive(Default, Debug, Clone)]
pub struct Part {
    pub primitives: Primitives,
    pub gradients: StateGradients,
    pub extrapolations: Primitives,
    pub conserved: Conserved,
    pub fluxes: Conserved,
    pub gravity_mflux: DVec3,

    pub limiter: LimiterData,
    pub matrix_wls: DMat3,

    pub volume: f64,
    pub x: DVec3,
    pub centroid: DVec3,
    pub v: DVec3,
    pub max_signal_velocity: f64,
    pub timebin: Timebin,
    pub wakeup: Timebin,
    pub dt: f64,

    pub a_grav: DVec3,
}

impl Part {
    pub fn timestep(
        &mut self,
        cfl_criterion: f64,
        eos: &EquationOfState,
        particle_motion: &ParticleMotion,
        dimensionality: HydroDimension,
    ) -> f64 {
        if self.conserved.mass() == 0. {
            // We have vacuum
            debug_assert_eq!(self.conserved.momentum().length(), 0.);
            debug_assert_eq!(self.conserved.energy(), 0.);
            self.v = DVec3::ZERO;
            return f64::INFINITY;
        }

        // Normal case
        let m_inv = 1. / self.conserved.mass();
        let internal_energy = (self.conserved.energy()
            - 0.5 * self.conserved.momentum().dot(self.primitives.velocity()))
            * m_inv;

        let fluid_v = self.conserved.momentum() * m_inv;
        let sound_speed = eos.sound_speed(
            eos.gas_pressure_from_internal_energy(internal_energy, self.primitives.density()),
            self.volume() * m_inv,
        );
        // Set the velocity with which this particle will be drifted over the course of it's next timestep
        let radius = self.radius(dimensionality);
        self.v = match particle_motion {
            ParticleMotion::FIXED => DVec3::ZERO,
            ParticleMotion::FLUID => fluid_v,
            ParticleMotion::STEER => {
                let d = self.centroid - self.x;
                let abs_d = d.length();
                let r = radius;
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
        let v_rel = (self.v - fluid_v).length();
        let v_max = self.max_signal_velocity.max(v_rel + sound_speed);

        if v_max > 0. {
            cfl_criterion * radius / v_max
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
                let div_v = self.gradients[1].x + self.gradients[2].y + self.gradients[3].z;
                self.extrapolations -= dt_extrapolate
                    * Primitives::new(
                        rho * div_v + v.dot(self.gradients[0]),
                        v * div_v + rho_inv * self.gradients[4],
                        gamma * p * div_v + v.dot(self.gradients[4]),
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
        self.gravity_mflux = DVec3::ZERO;
    }

    pub fn first_init(&mut self, eos: &EquationOfState) {
        self.primitives = Primitives::from_conserved(&self.conserved, self.volume(), eos);
    }

    pub fn from_ic(x: DVec3, mass: f64, velocity: DVec3, internal_energy: f64) -> Self {
        Self {
            x,
            conserved: Conserved::new(
                mass,
                mass * velocity,
                mass * internal_energy + 0.5 * mass * velocity.length_squared(),
            ),
            ..Self::default()
        }
    }

    pub fn convert_conserved_to_primitive(&mut self, eos: &EquationOfState) {
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
            Primitives::from_conserved(&self.conserved, self.volume(), eos)
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

        if engine.with_gravity {
            todo!()
            // TODO: Rewind kick1 if necessary
            // TODO: Reapply kick1 if necessary
        }
    }

    pub fn gradient_estimate(
        &mut self,
        primitives_other: &Primitives,
        w: f64,
        ds: DVec3,
        matrix_wls: DMat3,
    ) {
        fn gradient_est_single_quantity(q_l: f64, q_r: f64, w: f64, ds: DVec3, grad: &mut DVec3) {
            *grad += w * (q_r - q_l) * ds;
        }

        gradient_est_single_quantity(
            self.density(),
            primitives_other.density(),
            w,
            ds,
            &mut self.gradients[0],
        );
        gradient_est_single_quantity(
            self.velocity().x,
            primitives_other.velocity().x,
            w,
            ds,
            &mut self.gradients[1],
        );
        gradient_est_single_quantity(
            self.velocity().y,
            primitives_other.velocity().y,
            w,
            ds,
            &mut self.gradients[2],
        );
        gradient_est_single_quantity(
            self.velocity().z,
            primitives_other.velocity().z,
            w,
            ds,
            &mut self.gradients[3],
        );
        gradient_est_single_quantity(
            self.pressure(),
            primitives_other.pressure(),
            w,
            ds,
            &mut self.gradients[4],
        );

        self.matrix_wls += matrix_wls;
    }

    pub fn gradient_finalize(&mut self) {
        fn gradient_finalize_single_quantity(grad: &mut DVec3, mat_wls_inv: &DMat3) {
            *grad = mat_wls_inv.mul_vec3(*grad);
        }

        self.matrix_wls = self.matrix_wls.inverse();
        gradient_finalize_single_quantity(&mut self.gradients[0], &self.matrix_wls);
        gradient_finalize_single_quantity(&mut self.gradients[1], &self.matrix_wls);
        gradient_finalize_single_quantity(&mut self.gradients[2], &self.matrix_wls);
        gradient_finalize_single_quantity(&mut self.gradients[3], &self.matrix_wls);
        gradient_finalize_single_quantity(&mut self.gradients[4], &self.matrix_wls);
    }

    pub fn gradient_limiter_collect(
        &mut self,
        primitives_other: &Primitives,
        extrapolations: &Primitives,
    ) {
        self.limiter.min = self.limiter.min.pairwise_min(primitives_other);
        self.limiter.max = self.limiter.max.pairwise_max(primitives_other);
        self.limiter.e_min = self.limiter.min.pairwise_min(extrapolations);
        self.limiter.e_max = self.limiter.max.pairwise_max(extrapolations);
    }

    pub fn gradient_limit(&mut self) {
        cell_wide_limiter(
            &self.limiter.min,
            &self.limiter.max,
            &self.limiter.e_min,
            &self.limiter.e_max,
            &mut self.gradients,
        );
    }

    pub(crate) fn reset_gradients(&mut self) {
        self.gradients = StateGradients::zeros();
        self.limiter = LimiterData::default();
        self.extrapolations = Primitives::vacuum();
    }

    fn density(&mut self) -> f64 {
        self.primitives.density()
    }

    fn velocity(&mut self) -> DVec3 {
        self.primitives.velocity()
    }

    fn pressure(&mut self) -> f64 {
        self.primitives.pressure()
    }

    pub fn grav_kick(&mut self) {
        let mass = self.conserved.mass();
        let momentum = self.conserved.momentum();
        let grav_kick_factor = 0.5 * self.dt * self.a_grav;
        self.conserved += Conserved::new(
            0.,
            grav_kick_factor * mass,
            grav_kick_factor.dot(momentum - self.gravity_mflux),
        );
    }

    pub fn internal_energy(&self) -> f64 {
        (self.conserved.energy() - 0.5 * self.conserved.momentum().dot(self.primitives.velocity()))
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
        self.wakeup = self.timebin;
    }

    pub fn volume(&self) -> f64 {
            self.volume
    }

    pub fn reflect(&self, around: DVec3, normal: DVec3) -> Self {
        let mut reflected = self.clone();
        let dx = around - self.x;
        reflected.x += 2. * (around - self.x).dot(normal) * normal;
        reflected.centroid += 2. * (around - self.centroid).dot(normal) * normal;
        reflected.v -= 2. * reflected.v.dot(normal) * normal;

        reflected
    }

    pub fn reflect_quantities(&mut self, normal: DVec3) -> &mut Self {
        // Reflect fluid velocity component along normal
        let mut v = self.primitives.velocity();
        v -= 2. * v.dot(normal) * normal;
        self.primitives = Primitives::new(self.primitives.density(), v, self.primitives.pressure());

        self
    }

    pub fn reflect_gradients(&mut self, normal: DVec3) -> &mut Self {
        // Reflect gradients along normal
        let mut grad_reflected = self.gradients;
        grad_reflected[0] = grad_reflected[0] - 2. * grad_reflected[0].dot(normal) * normal;
        grad_reflected[4] = grad_reflected[4] - 2. * grad_reflected[0].dot(normal) * normal;
        // For the velocity: first account for sign change due to reflection of velocity:
        let (dv_dx, dv_dy, dv_dz) = (
            DVec3 {
                x: grad_reflected[1].x,
                y: grad_reflected[2].x,
                z: grad_reflected[3].x,
            },
            DVec3 {
                x: grad_reflected[1].y,
                y: grad_reflected[2].y,
                z: grad_reflected[3].y,
            },
            DVec3 {
                x: grad_reflected[1].z,
                y: grad_reflected[2].z,
                z: grad_reflected[3].z,
            },
        );
        let dv_dot_n = DVec3 {
            x: dv_dx.dot(normal),
            y: dv_dy.dot(normal),
            z: dv_dz.dot(normal),
        };
        grad_reflected[1] = grad_reflected[1] - 2. * dv_dot_n * normal.x;
        grad_reflected[2] = grad_reflected[2] - 2. * dv_dot_n * normal.y;
        grad_reflected[3] = grad_reflected[3] - 2. * dv_dot_n * normal.z;
        // Now flip the gradients:
        grad_reflected[1] = grad_reflected[1] - 2. * grad_reflected[1].dot(normal) * normal;
        grad_reflected[2] = grad_reflected[2] - 2. * grad_reflected[2].dot(normal) * normal;
        grad_reflected[3] = grad_reflected[3] - 2. * grad_reflected[3].dot(normal) * normal;

        self.gradients = grad_reflected;

        self
    }

    pub fn loc(&self) -> DVec3 {
        self.x
    }

    pub fn set_centroid(&mut self, centroid: DVec3) {
        self.centroid = centroid;
    }

    fn radius(&self, dimensionality: HydroDimension) -> f64 {
        match dimensionality {
            HydroDimension1D => 0.5 * self.volume(),
            HydroDimension2D => (std::f64::consts::FRAC_1_PI * self.volume()).sqrt(),
            HydroDimension3D => (0.25 * 3. * std::f64::consts::FRAC_1_PI * self.volume()).powf(1. / 3.),
        }
    }
}
