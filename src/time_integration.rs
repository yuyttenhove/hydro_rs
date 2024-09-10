use crate::{
    engine::{Engine, TimestepInfo},
    part::Particle,
    riemann_solver::RiemannFluxSolver,
    space::Space,
    timeline::IntegerTime,
};

pub enum Iact {
    Volume,
    Primitive,
    Gradient,
    Flux,
    ApplyFlux,
}

#[derive(Debug)]
pub enum Runner {
    Default,
    OptimalOrder,
    TwoGradient,
    Pakmor,
    PakmorExtrapolate,
    VolumeBackExtrapolate,
    TwoVolumeHalfDrift,
    OptimalOrderHalfDrift,
    DefaultHalfDrift,
    MeshlessGradientHalfDrift,
    FluxExtrapolateHalfDrift,
}

impl Runner {
    pub fn label(&self) -> &str {
        match self {
            Runner::Default => "default",
            Runner::OptimalOrder => "optimal",
            Runner::TwoGradient => "two_gradient",
            Runner::Pakmor => "pakmor",
            Runner::PakmorExtrapolate => "pakmor_end",
            Runner::VolumeBackExtrapolate => "volume_back_extrapolate",
            Runner::TwoVolumeHalfDrift => "two_volume_half",
            Runner::OptimalOrderHalfDrift => "optimal_half",
            Runner::DefaultHalfDrift => "default_half",
            Runner::MeshlessGradientHalfDrift => "meshless_gradient_half",
            Runner::FluxExtrapolateHalfDrift => "flux_extrapolate_half",
        }
    }

    pub fn drift(&self, dti: IntegerTime, timestep_info: &TimestepInfo, space: &mut Space) {
        let dt = timestep_info.dt_from_dti(dti);
        match self {
            Runner::FluxExtrapolateHalfDrift => space.drift(dt, 0.),
            // Runner::VolumeBackExtrapolate => space.drift(dt, 0.5 * dt, engine),
            _ => space.drift(dt, dt),
        }
    }

    pub fn step<R: RiemannFluxSolver>(&self, engine: &Engine<R>, space: &mut Space) -> IntegerTime {
        let ti_next;
        match self {
            Runner::Default => {
                space.volume_calculation(self, &engine.timestep_info);
                space.convert_conserved_to_primitive(self, &engine.timestep_info);
                space.gradient_estimate(self, &engine.timestep_info);
                space.flux_exchange(self, &engine.timestep_info, &engine.riemann_solver);
                space.apply_flux(self, &engine.timestep_info);
                space.kick2(&engine.timestep_info);
                ti_next = space.timestep(engine);
                space.timestep_limiter(&engine.timestep_info); // Note: this can never decrease ti_next
                space.kick1(&engine.timestep_info, &engine.particle_motion);
            }
            Runner::OptimalOrder => {
                // space.regrid();
                space.volume_calculation(self, &engine.timestep_info);
                space.flux_exchange(self, &engine.timestep_info, &engine.riemann_solver);
                space.apply_flux(self, &engine.timestep_info);
                space.gravity(&engine.gravity_solver);
                space.kick2(&engine.timestep_info);
                space.convert_conserved_to_primitive(self, &engine.timestep_info);
                space.gradient_estimate(self, &engine.timestep_info);
                // space.meshless_gradient_estimate(self, &engine.timestep_info);
                ti_next = space.timestep(engine);
                space.timestep_limiter(&engine.timestep_info); // Note: this can never decrease ti_next
                space.kick1(&engine.timestep_info, &engine.particle_motion);
            }
            Runner::TwoGradient => {
                space.volume_calculation(self, &engine.timestep_info);
                space.gradient_estimate(self, &engine.timestep_info);
                space.flux_exchange(self, &engine.timestep_info, &engine.riemann_solver);
                space.apply_flux(self, &engine.timestep_info);
                space.convert_conserved_to_primitive(self, &engine.timestep_info);
                space.gradient_estimate(self, &engine.timestep_info);
                space.gravity(&engine.gravity_solver);
                space.kick2(&engine.timestep_info);
                ti_next = space.timestep(engine);
                space.timestep_limiter(&engine.timestep_info); // Note: this can never decrease ti_next
                space.kick1(&engine.timestep_info, &engine.particle_motion);
            }
            Runner::Pakmor => {
                space.volume_calculation(self, &engine.timestep_info);
                space.gradient_estimate(self, &engine.timestep_info);
                // First half flux
                space.half_flux_exchange(self, &engine.timestep_info, &engine.riemann_solver);
                space.apply_flux(self, &engine.timestep_info);
                space.gravity(&engine.gravity_solver);
                space.kick2(&engine.timestep_info);
                space.convert_conserved_to_primitive(self, &engine.timestep_info);
                space.gradient_estimate(self, &engine.timestep_info);
                ti_next = space.timestep(engine);
                // space.timestep_limiter(self, &engine.timestep_info); // Note: this can never decrease ti_next
                space.kick1(&engine.timestep_info, &engine.particle_motion);
                // second half flux
                space.half_flux_exchange(self, &engine.timestep_info, &engine.riemann_solver);
            }
            Runner::PakmorExtrapolate => {
                space.volume_calculation(self, &engine.timestep_info);
                space.flux_exchange_pakmor_single(
                    self,
                    &engine.timestep_info,
                    &engine.riemann_solver,
                );
                space.apply_flux(self, &engine.timestep_info);
                space.gravity(&engine.gravity_solver);
                space.kick2(&engine.timestep_info);
                space.convert_conserved_to_primitive(self, &engine.timestep_info);
                space.gradient_estimate(self, &engine.timestep_info);
                ti_next = space.timestep(engine);
                space.timestep_limiter(&engine.timestep_info); // Note: this can never decrease ti_next
                space.kick1(&engine.timestep_info, &engine.particle_motion);
            }
            Runner::VolumeBackExtrapolate => {
                space.regrid();
                space.volume_calculation_back_extrapolate(self, &engine.timestep_info);
                // space.volume_derivative_estimate(self, &engine.timestep_info);
                // Todo: update primitives using new volumes?
                // Todo: flux extrapolate (godunov) to half timestep? Or keep gradient extrapolation?
                // Todo: Recompute spatial gradients at half timestep in back extrapolated coordinates?
                // Todo: Do flux calculation in back extrapolated coordinates as well
                // space.convert_conserved_to_primitive(self, &engine.timestep_info);
                // space.apply_time_extrapolations(self, &engine.timestep_info);
                // space.gradient_estimate(self, &engine.timestep_info);
                // space.flux_exchange_no_back_extrapolation(self, &engine.timestep_info, &engine.riemann_solver);
                space.flux_exchange(self, &engine.timestep_info, &engine.riemann_solver);
                space.apply_flux(self, &engine.timestep_info);
                space.drift_centroids_to_current_time(self, &engine.timestep_info);
                space.gravity(&engine.gravity_solver);
                space.kick2(&engine.timestep_info);
                space.convert_conserved_to_primitive(self, &engine.timestep_info);
                space.meshless_gradient_estimate(self, &engine.timestep_info);
                // space.gradient_estimate(self, &engine.timestep_info);
                ti_next = space.timestep(engine);
                space.timestep_limiter(&engine.timestep_info); // Note: this can never decrease ti_next
                space.kick1(&engine.timestep_info, &engine.particle_motion);
            }
            _ => {
                panic!("{self:?} should not be run with full steps!")
            }
        }
        space.prepare();
        space.self_check();
        ti_next
    }

    pub fn half_step1<R: RiemannFluxSolver>(&self, engine: &Engine<R>, space: &mut Space) {
        match self {
            Runner::TwoVolumeHalfDrift => {
                space.volume_calculation(self, &engine.timestep_info);
                space.flux_exchange_no_back_extrapolation(
                    self,
                    &engine.timestep_info,
                    &engine.riemann_solver,
                );
            }
            Runner::OptimalOrderHalfDrift => {
                space.volume_calculation(self, &engine.timestep_info);
                space.flux_exchange_no_back_extrapolation(
                    self,
                    &engine.timestep_info,
                    &engine.riemann_solver,
                );
                space.apply_flux(self, &engine.timestep_info);
                space.convert_conserved_to_primitive(self, &engine.timestep_info);
                space.gradient_estimate(self, &engine.timestep_info);
            }
            Runner::DefaultHalfDrift => {
                space.volume_calculation(self, &engine.timestep_info);
                space.convert_conserved_to_primitive(self, &engine.timestep_info);
                space.gradient_estimate(self, &engine.timestep_info);
                space.flux_exchange_no_back_extrapolation(
                    self,
                    &engine.timestep_info,
                    &engine.riemann_solver,
                );
            }
            Runner::MeshlessGradientHalfDrift => {
                space.volume_calculation(self, &engine.timestep_info);
                space.volume_derivative_estimate(self, &engine.timestep_info);
                space.apply_time_extrapolations(self, &engine.timestep_info);
                space.gradient_estimate(self, &engine.timestep_info);
                space.flux_exchange_no_back_extrapolation(
                    self,
                    &engine.timestep_info,
                    &engine.riemann_solver,
                );
            }
            Runner::FluxExtrapolateHalfDrift => {
                space.volume_calculation(self, &engine.timestep_info);
                space.extrapolate_flux(self, &engine.timestep_info, &engine.riemann_solver);
                space.convert_conserved_to_primitive(self, &engine.timestep_info);
                space.gradient_estimate(self, &engine.timestep_info);
                space.flux_exchange_no_back_extrapolation(
                    self,
                    &engine.timestep_info,
                    &engine.riemann_solver,
                );
            }
            _ => {
                panic!("{self:?} should not be run with half steps!")
            }
        }
    }

    pub fn half_step2<R: RiemannFluxSolver>(
        &self,
        engine: &Engine<R>,
        space: &mut Space,
    ) -> IntegerTime {
        let ti_next;
        match self {
            Runner::TwoVolumeHalfDrift => {
                space.volume_calculation(self, &engine.timestep_info);
                space.apply_flux(self, &engine.timestep_info);
                space.kick2(&engine.timestep_info);
                space.convert_conserved_to_primitive(self, &engine.timestep_info);
                space.gradient_estimate(self, &engine.timestep_info);
                ti_next = space.timestep(engine);
                // space.timestep_limiter(self, &engine.timestep_info); // Note: this can never decrease ti_next
                space.kick1(&engine.timestep_info, &engine.particle_motion);
            }
            Runner::OptimalOrderHalfDrift => {
                space.kick2(&engine.timestep_info);
                ti_next = space.timestep(engine);
                // space.timestep_limiter(self, &engine.timestep_info); // Note: this can never decrease ti_next
                space.kick1(&engine.timestep_info, &engine.particle_motion);
            }
            Runner::DefaultHalfDrift => {
                space.apply_flux(self, &engine.timestep_info);
                space.kick2(&engine.timestep_info);
                ti_next = space.timestep(engine);
                // space.timestep_limiter(self, &engine.timestep_info); // Note: this can never decrease ti_next
                space.kick1(&engine.timestep_info, &engine.particle_motion);
            }
            Runner::MeshlessGradientHalfDrift => {
                space.regrid();
                space.apply_flux(self, &engine.timestep_info);
                space.kick2(&engine.timestep_info);
                space.convert_conserved_to_primitive(self, &engine.timestep_info);
                space.meshless_gradient_estimate(self, &engine.timestep_info);
                ti_next = space.timestep(engine);
                // space.meshless_timestep_limiter(self, &engine.timestep_info); // Note: this can never decrease ti_next
                space.kick1(&engine.timestep_info, &engine.particle_motion);
            }
            Runner::FluxExtrapolateHalfDrift => {
                space.regrid();
                space.apply_flux(self, &engine.timestep_info);
                space.kick2(&engine.timestep_info);
                space.convert_conserved_to_primitive(self, &engine.timestep_info);
                space.meshless_gradient_estimate(self, &engine.timestep_info);
                ti_next = space.timestep(engine);
                // space.meshless_timestep_limiter(self, &engine.timestep_info); // Note: this can never decrease ti_next
                space.kick1(&engine.timestep_info, &engine.particle_motion);
            }
            _ => {
                panic!("{self:?} should not be run with half steps!")
            }
        }
        space.prepare();
        space.self_check();
        ti_next
    }

    pub fn use_half_step(&self) -> bool {
        match self {
            Runner::DefaultHalfDrift
            | Runner::OptimalOrderHalfDrift
            | Runner::TwoVolumeHalfDrift
            | Runner::MeshlessGradientHalfDrift
            | Runner::FluxExtrapolateHalfDrift => true,
            _ => false,
        }
    }

    pub fn part_is_active(
        &self,
        part: &Particle,
        iact: Iact,
        timestep_info: &TimestepInfo,
    ) -> bool {
        match self {
            Runner::Default
            | Runner::OptimalOrder
            | Runner::TwoGradient
            | Runner::Pakmor
            | Runner::PakmorExtrapolate
            | Runner::VolumeBackExtrapolate => timestep_info.bin_is_ending(part.timebin),
            Runner::OptimalOrderHalfDrift => timestep_info.bin_is_halfway(part.timebin),
            Runner::DefaultHalfDrift => match iact {
                Iact::ApplyFlux => timestep_info.bin_is_ending(part.timebin),
                _ => timestep_info.bin_is_halfway(part.timebin),
            },
            Runner::TwoVolumeHalfDrift => match iact {
                Iact::Flux => timestep_info.bin_is_halfway(part.timebin),
                Iact::Volume => {
                    timestep_info.bin_is_ending(part.timebin)
                        || timestep_info.bin_is_halfway(part.timebin)
                }
                _ => timestep_info.bin_is_ending(part.timebin),
            },
            Runner::MeshlessGradientHalfDrift => match iact {
                Iact::Flux | Iact::Volume => timestep_info.bin_is_halfway(part.timebin),
                Iact::Gradient => {
                    timestep_info.bin_is_halfway(part.timebin)
                        || timestep_info.bin_is_ending(part.timebin)
                }
                _ => timestep_info.bin_is_ending(part.timebin),
            },
            Runner::FluxExtrapolateHalfDrift => match iact {
                Iact::Flux | Iact::Volume => timestep_info.bin_is_halfway(part.timebin),
                _ => timestep_info.bin_is_ending(part.timebin),
            },
        }
    }
}
