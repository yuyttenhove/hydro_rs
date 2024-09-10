use glam::DVec3;
use meshless_voronoi::VoronoiCell;

use crate::engine::TimestepInfo;
use crate::gas_law::GasLaw;
use crate::physical_quantities::{Conserved, Gradients, Primitive, State};
use crate::utils::{box_reflect, box_wrap, contains};
use crate::{engine::ParticleMotion, flux::FluxInfo, timeline::*, Dimensionality};

#[derive(Default, Debug, Clone)]
pub struct Particle {
    pub primitives: State<Primitive>,
    pub gradients: Gradients<Primitive>,
    pub gradients_centroid: DVec3,
    pub extrapolations: State<Primitive>,
    pub conserved: State<Conserved>,
    pub fluxes: State<Conserved>,
    pub gravity_mflux: DVec3,

    pub volume: f64,
    pub face_connections_offset: usize,
    pub face_count: usize,
    pub search_radius: f64,
    pub cell_id: usize,
    pub loc: DVec3,
    pub centroid: DVec3,
    // Extrapolated particle velocity
    pub v: DVec3,
    // relative fluid velocity at last full timestep
    v_rel: DVec3,
    max_a_over_r: f64,
    pub max_signal_velocity: f64,
    pub timebin: Timebin,
    pub dt: f64,
    // Volume derivative
    pub dvdt: f64,

    pub a_grav: DVec3,
}

impl Particle {
    pub fn timestep(
        &mut self,
        cfl_criterion: f64,
        particle_motion: &ParticleMotion,
        eos: &GasLaw,
        dimensionality: Dimensionality,
    ) -> f64 {
        if self.conserved.mass() == 0. {
            // We have vacuum
            debug_assert_eq!(self.conserved.momentum().length(), 0.);
            debug_assert_eq!(self.conserved.energy(), 0.);
            self.v = DVec3::ZERO;
            return f64::INFINITY;
        }

        let m_inv = 1. / self.conserved.mass();
        // Fluid velocity at generator (instead of centroid)
        let d_v = self
            .gradients
            .dot(self.loc - self.gradients_centroid)
            .velocity();
        let fluid_v = self.conserved.momentum() * m_inv + d_v;
        let sound_speed = eos.sound_speed(
            eos.gas_pressure_from_internal_energy(
                self.internal_energy(),
                self.primitives.density(),
            ),
            self.volume() * m_inv,
        );
        self.set_particle_velocity(fluid_v, sound_speed, particle_motion, dimensionality);

        // determine the size of this particle's next timestep
        let v_max = self
            .max_signal_velocity
            .max(self.v_rel.length() + sound_speed);
        self.max_signal_velocity = 0.;

        if v_max > 0. {
            cfl_criterion * self.radius(dimensionality) / v_max
        } else {
            f64::INFINITY
        }
    }

    /// We need to kick the particle velocity (not fluid velocity!) for half the timestep
    pub fn hydro_kick1(
        &mut self,
        particle_motion: &ParticleMotion,
        dimensionality: Dimensionality,
    ) {
        match particle_motion {
            ParticleMotion::Fixed => (),
            _ => {
                let rho = self.primitives.density();
                if rho > 0. {
                    let kick_fac = 0.5 * self.dt / rho;
                    for i in 0..dimensionality.into() {
                        self.v[i] -= kick_fac * self.gradients[4][i];
                    }
                }
            }
        };
    }

    /// Drifts the particle forward in time over a time `dt_drift`.
    ///
    /// Also predicts the primitive quantities forward in time over a time dt_extrapolate using the Euler equations.
    pub fn drift(
        &mut self,
        dt_drift: f64,
        dt_extrapolate: f64,
        eos: &GasLaw,
        dimensionality: Dimensionality,
    ) {
        for i in 0..dimensionality.into() {
            self.loc[i] += self.v[i] * dt_drift;
            self.centroid[i] += self.v[i] * dt_drift;
            self.gradients_centroid[i] += self.v[i] * dt_drift;
        }

        debug_assert!(self.loc.is_finite(), "Infinite x after drift!");

        // Extrapolate primitives in time
        self.extrapolations += self.time_extrapolations(dt_extrapolate, eos);

        self.primitives.check_physical();

        // Extrapolate volume in time
        let volume = self.volume + self.dvdt * self.dt;
        self.volume = (volume).clamp(0.5 * self.volume, 2. * self.volume);
    }

    pub fn time_extrapolations(&self, dt: f64, eos: &GasLaw) -> State<Primitive> {
        let rho = self.primitives.density();
        if rho > 0. {
            let rho_inv = 1. / rho;
            // Use fluid velocity in comoving frame!
            let p = self.primitives.pressure();
            let div_v = self.gradients.div_v();
            -dt * State::<Primitive>::new(
                rho * div_v + self.v_rel.dot(self.gradients[0]),
                self.v_rel * div_v + rho_inv * self.gradients[4],
                eos.gamma().gamma() * p * div_v + self.v_rel.dot(self.gradients[4]),
            )
        } else {
            State::vacuum()
        }
    }

    pub fn update_geometry(&mut self, voronoi_cell: &VoronoiCell) {
        self.volume = voronoi_cell.volume();
        self.face_connections_offset = voronoi_cell.face_connections_offset();
        self.face_count = voronoi_cell.face_count();
        self.search_radius = 1.5 * voronoi_cell.safety_radius();
        self.set_centroid(voronoi_cell.centroid());
        debug_assert!(self.volume >= 0.);
    }

    pub fn update_fluxes_left(&mut self, flux_info: &FluxInfo, part_is_active: bool) {
        self.fluxes -= flux_info.fluxes;
        self.gravity_mflux -= flux_info.mflux;
        if part_is_active {
            self.max_signal_velocity = flux_info.v_max.max(self.max_signal_velocity);
            self.max_a_over_r = self.max_a_over_r.max(flux_info.a_over_r);
        }
    }

    pub fn update_fluxes_right(&mut self, flux_info: &FluxInfo, part_is_active: bool) {
        self.fluxes += flux_info.fluxes;
        self.gravity_mflux -= flux_info.mflux;
        if part_is_active {
            self.max_signal_velocity = flux_info.v_max.max(self.max_signal_velocity);
            self.max_a_over_r = self.max_a_over_r.max(flux_info.a_over_r);
        }
    }

    pub fn apply_flux(&mut self) {
        self.conserved += self.fluxes;
        debug_assert!(self.conserved.mass().is_finite());
        debug_assert!(self.conserved.momentum().is_finite());
        debug_assert!(self.conserved.energy().is_finite());

        if self.conserved.mass() < 0. {
            eprintln!("Negative mass after applying fluxes");
            self.conserved = State::vacuum();
        }
        if self.conserved.energy() < 0. {
            eprintln!("Negative energy after applying fluxes");
            self.conserved = State::vacuum();
        }
        self.reset_fluxes();
    }

    fn reset_fluxes(&mut self) {
        self.fluxes = State::vacuum();
        self.gravity_mflux = DVec3::ZERO;
    }

    pub fn first_init(&mut self, eos: &GasLaw) {
        self.primitives = State::from_conserved(&self.conserved, self.volume(), eos);
        debug_assert!(self.primitives.density().is_finite());
        debug_assert!(self.primitives.velocity().is_finite());
        debug_assert!(self.primitives.pressure().is_finite());
    }

    pub fn from_ic(x: DVec3, mass: f64, velocity: DVec3, internal_energy: f64) -> Self {
        Self {
            loc: x,
            conserved: State::<Conserved>::new(
                mass,
                mass * velocity,
                mass * internal_energy + 0.5 * mass * velocity.length_squared(),
            ),
            ..Self::default()
        }
    }

    pub fn convert_conserved_to_primitive(&mut self, eos: &GasLaw) {
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
            State::vacuum()
        } else {
            State::from_conserved(&self.conserved, self.volume(), eos)
        };

        // if self.primitives.density() < 1e-10 {
        //     self.primitives = Primitives::new(
        //         self.primitives.density(),
        //         self.primitives.velocity() * self.primitives.density() * 1e10,
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

    pub fn timestep_limit(&mut self, wakeup: i8, timestep_info: &TimestepInfo) {
        // Anything to do here?
        if wakeup >= self.timebin {
            return;
        }

        if timestep_info.bin_is_starting(self.timebin) {
            // Particle was active/starting anyway, so just updating the timestep/timebin suffices.
            self.timebin = wakeup;
            self.dt = timestep_info.dt_from_bin(self.timebin);
        } else {
            // Wake this particle up
            let ti_end_old = timestep_info.get_integer_time_end(self.timebin);
            let ti_current = timestep_info.ti_current;
            debug_assert_ne!(ti_end_old, ti_current);
            // Substract the remainder of this particles old timestep from its dt
            self.dt -= timestep_info.dt_from_dti(ti_end_old - ti_current);
            // Update the timebin
            self.timebin = wakeup;
            let dti_new = get_integer_timestep(self.timebin);
            let ti_end = timestep_info.get_integer_time_end(self.timebin);
            // Add the remainder of the new timestep to the particle's dt
            self.dt += if ti_end == ti_current {
                timestep_info.dt_from_dti(dti_new)
                // Part is now active/starting, so the remaining KICK1 will be applied automatically
            } else {
                timestep_info.dt_from_dti(ti_end - ti_current)
                // TODO: reapply second part of KICK1 if neccessary
            };

            // TODO this might still not be correct for particles that have a longer timestep then this particles new timestep, but shorter than this particles old timestep
        }

        // TODO: Handle gravity (rewind kick1 and reapply)
    }

    pub(crate) fn reset_gradients(&mut self) {
        self.gradients = Gradients::zeros();
        self.extrapolations = State::vacuum();
    }

    pub fn grav_kick(&mut self) {
        let mass = self.conserved.mass();
        let momentum = self.conserved.momentum();
        let grav_kick_factor = 0.5 * self.dt * self.a_grav;
        self.conserved += State::<Conserved>::new(
            0.,
            grav_kick_factor * mass,
            grav_kick_factor.dot(momentum - self.gravity_mflux),
        );
    }

    pub fn internal_energy(&self) -> f64 {
        (self.conserved.energy() - 0.5 * self.conserved.momentum().dot(self.primitives.velocity()))
            / self.conserved.mass()
    }

    pub fn set_timestep(&mut self, dt: f64, new_dti: IntegerTime) {
        self.dt = dt;
        self.timebin = get_time_bin(new_dti);
    }

    pub fn volume(&self) -> f64 {
        self.volume
    }

    fn set_particle_velocity(
        &mut self,
        fluid_v: DVec3,
        sound_speed: f64,
        particle_motion: &ParticleMotion,
        dimensionality: Dimensionality,
    ) {
        self.v = match particle_motion {
            ParticleMotion::Fixed => DVec3::ZERO,
            ParticleMotion::Fluid => fluid_v,
            ParticleMotion::Steer => {
                let d = self.centroid - self.loc;
                let abs_d = d.length();
                let eta = 0.25;
                let eta_r = eta * self.radius(dimensionality);
                let xi = 0.99;
                if abs_d > 0.9 * eta_r {
                    let mut fac = xi * sound_speed / abs_d;
                    if abs_d < 1.1 * eta_r {
                        fac *= 5. * (abs_d - 0.9 * eta_r) / eta_r;
                    }
                    debug_assert!(fac * abs_d < sound_speed);
                    fluid_v + fac * d
                } else {
                    fluid_v
                }
            }
            ParticleMotion::SteerPakmor => {
                let alpha = 2. / usize::from(dimensionality) as f64 * self.max_a_over_r;
                let beta = 2.25;
                let eta = 0f64.max(0.5f64.min(2. / beta * (alpha - 0.75 * beta)));
                let d = (self.centroid - self.loc).normalize();
                let v_reg = self.radius(dimensionality) * self.gradients.curl_v().length();
                let v_reg = eta * sound_speed.max(v_reg) * d;
                fluid_v + v_reg
            }
        };
        // Set the velocity with which this particle will be drifted over the course of it's next timestep
        assert!(self.v.is_finite(), "Invalid value for v!");
        self.v_rel = fluid_v - self.v;
        debug_assert!(self.v_rel.length() < sound_speed);

        // Reset max_a_over_r (not needed any more)
        self.max_a_over_r = 0.;
    }

    pub fn reflect(&self, around: DVec3, normal: DVec3) -> Self {
        let mut reflected = self.clone();
        reflected.loc += 2. * (around - self.loc).dot(normal) * normal;
        reflected.centroid += 2. * (around - self.centroid).dot(normal) * normal;
        reflected.gradients_centroid +=
            2. * (around - self.gradients_centroid).dot(normal) * normal;
        reflected.v -= 2. * reflected.v.dot(normal) * normal;

        reflected
    }

    pub fn reflect_quantities(mut self, normal: DVec3) -> Self {
        self.primitives = self.primitives.reflect(normal);
        self
    }

    pub fn reflect_gradients(mut self, normal: DVec3) -> Self {
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
        self.loc
    }

    pub fn set_centroid(&mut self, centroid: DVec3) {
        self.centroid = centroid;
    }

    pub fn radius(&self, dimensionality: Dimensionality) -> f64 {
        match dimensionality {
            Dimensionality::OneD => 0.5 * self.volume(),
            Dimensionality::TwoD => (std::f64::consts::FRAC_1_PI * self.volume()).sqrt(),
            Dimensionality::ThreeD => {
                (0.25 * 3. * std::f64::consts::FRAC_1_PI * self.volume()).powf(1. / 3.)
            }
        }
    }

    pub fn box_wrap(&mut self, box_size: DVec3, dimensionality: Dimensionality) {
        let pos_old = self.loc;
        box_wrap(box_size, &mut self.loc, dimensionality.into());
        let shift = self.loc - pos_old;
        self.gradients_centroid += shift;
        self.centroid += shift;
    }

    pub fn box_reflect(&mut self, box_size: DVec3, dimensionality: Dimensionality) {
        let pos_old = self.loc;
        box_reflect(box_size, &mut self.loc, dimensionality.into());
        debug_assert!(contains(box_size, self.loc, dimensionality.into()));
        let shift = self.loc - pos_old;
        if shift.length_squared() > 0. {
            let normal = shift * shift.length_recip();
            *self = self
                .clone()
                .reflect_quantities(normal)
                .reflect_gradients(normal);
        }
    }
}
