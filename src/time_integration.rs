use crate::{engine::Engine, part::Particle, space::Space, timeline::IntegerTime};

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

    pub fn drift(&self, dti: IntegerTime, engine: &Engine, space: &mut Space) {
        let dt = engine.dt(dti);
        match self {
            Runner::FluxExtrapolateHalfDrift => space.drift(dt, 0., engine),
            // Runner::VolumeBackExtrapolate => space.drift(dt, 0.5 * dt, engine),
            _ => space.drift(dt, dt, engine),
        }
    }

    pub fn step(&self, engine: &Engine, space: &mut Space) -> IntegerTime {
        let ti_next;
        match self {
            Runner::Default => {
                space.volume_calculation(engine);
                space.convert_conserved_to_primitive(engine);
                space.gradient_estimate(engine);
                space.flux_exchange(engine);
                space.apply_flux(engine);
                space.kick2(engine);
                ti_next = space.timestep(engine);
                space.timestep_limiter(engine); // Note: this can never decrease ti_next
                space.kick1(engine);
            }
            Runner::OptimalOrder => {
                // space.regrid();
                space.volume_calculation(engine);
                space.flux_exchange(engine);
                space.apply_flux(engine);
                space.gravity(engine);
                space.kick2(engine);
                space.convert_conserved_to_primitive(engine);
                space.gradient_estimate(engine);
                // space.meshless_gradient_estimate(engine);
                ti_next = space.timestep(engine);
                space.timestep_limiter(engine); // Note: this can never decrease ti_next
                space.kick1(engine);
            }
            Runner::TwoGradient => {
                space.volume_calculation(engine);
                space.gradient_estimate(engine);
                space.flux_exchange(engine);
                space.apply_flux(engine);
                space.convert_conserved_to_primitive(engine);
                space.gradient_estimate(engine);
                space.gravity(engine);
                space.kick2(engine);
                ti_next = space.timestep(engine);
                space.timestep_limiter(engine); // Note: this can never decrease ti_next
                space.kick1(engine);
            }
            Runner::Pakmor => {
                space.volume_calculation(engine);
                space.gradient_estimate(engine);
                // First half flux
                space.half_flux_exchange(engine);
                space.apply_flux(engine);
                space.gravity(engine);
                space.kick2(engine);
                space.convert_conserved_to_primitive(engine);
                space.gradient_estimate(engine);
                ti_next = space.timestep(engine);
                // space.timestep_limiter(engine); // Note: this can never decrease ti_next
                space.kick1(engine);
                // second half flux
                space.half_flux_exchange(engine);
            }
            Runner::PakmorExtrapolate => {
                space.volume_calculation(engine);
                space.flux_exchange_pakmor_single(engine);
                space.apply_flux(engine);
                space.gravity(engine);
                space.kick2(engine);
                space.convert_conserved_to_primitive(engine);
                space.gradient_estimate(engine);
                ti_next = space.timestep(engine);
                space.timestep_limiter(engine); // Note: this can never decrease ti_next
                space.kick1(engine);
            }
            Runner::VolumeBackExtrapolate => {
                space.regrid();
                space.volume_calculation_back_extrapolate(engine);
                // space.volume_derivative_estimate(engine);
                // Todo: update primitives using new volumes?
                // Todo: flux extrapolate (godunov) to half timestep? Or keep gradient extrapolation?
                // Todo: Recompute spatial gradients at half timestep in back extrapolated coordinates?
                // Todo: Do flux calculation in back extrapolated coordinates as well
                // space.convert_conserved_to_primitive(engine);
                // space.apply_time_extrapolations(engine);
                // space.gradient_estimate(engine);
                // space.flux_exchange_no_back_extrapolation(engine);
                space.flux_exchange(engine);
                space.apply_flux(engine);
                space.drift_centroids_to_current_time(engine);
                space.gravity(engine);
                space.kick2(engine);
                space.convert_conserved_to_primitive(engine);
                space.meshless_gradient_estimate(engine);
                // space.gradient_estimate(engine);
                ti_next = space.timestep(engine);
                space.timestep_limiter(engine); // Note: this can never decrease ti_next
                space.kick1(engine);
            }
            _ => {
                panic!("{self:?} should not be run with full steps!")
            }
        }
        space.prepare(engine);
        space.self_check();
        ti_next
    }

    pub fn half_step1(&self, engine: &Engine, space: &mut Space) {
        match self {
            Runner::TwoVolumeHalfDrift => {
                space.volume_calculation(engine);
                space.flux_exchange_no_back_extrapolation(engine);
            }
            Runner::OptimalOrderHalfDrift => {
                space.volume_calculation(engine);
                space.flux_exchange_no_back_extrapolation(engine);
                space.apply_flux(engine);
                space.convert_conserved_to_primitive(engine);
                space.gradient_estimate(engine);
            }
            Runner::DefaultHalfDrift => {
                space.volume_calculation(engine);
                space.convert_conserved_to_primitive(engine);
                space.gradient_estimate(engine);
                space.flux_exchange_no_back_extrapolation(engine);
            }
            Runner::MeshlessGradientHalfDrift => {
                space.volume_calculation(engine);
                space.volume_derivative_estimate(engine);
                space.apply_time_extrapolations(engine);
                space.gradient_estimate(engine);
                space.flux_exchange_no_back_extrapolation(engine);
            }
            Runner::FluxExtrapolateHalfDrift => {
                space.volume_calculation(engine);
                space.extrapolate_flux(engine);
                space.convert_conserved_to_primitive(engine);
                space.gradient_estimate(engine);
                space.flux_exchange_no_back_extrapolation(engine);
            }
            _ => {
                panic!("{self:?} should not be run with half steps!")
            }
        }
    }

    pub fn half_step2(&self, engine: &Engine, space: &mut Space) -> IntegerTime {
        let ti_next;
        match self {
            Runner::TwoVolumeHalfDrift => {
                space.volume_calculation(engine);
                space.apply_flux(engine);
                space.kick2(engine);
                space.convert_conserved_to_primitive(engine);
                space.gradient_estimate(engine);
                ti_next = space.timestep(engine);
                // space.timestep_limiter(engine); // Note: this can never decrease ti_next
                space.kick1(engine);
            }
            Runner::OptimalOrderHalfDrift => {
                space.kick2(engine);
                ti_next = space.timestep(engine);
                // space.timestep_limiter(engine); // Note: this can never decrease ti_next
                space.kick1(engine);
            }
            Runner::DefaultHalfDrift => {
                space.apply_flux(engine);
                space.kick2(engine);
                ti_next = space.timestep(engine);
                // space.timestep_limiter(engine); // Note: this can never decrease ti_next
                space.kick1(engine);
            }
            Runner::MeshlessGradientHalfDrift => {
                space.regrid();
                space.apply_flux(engine);
                space.kick2(engine);
                space.convert_conserved_to_primitive(engine);
                space.meshless_gradient_estimate(engine);
                ti_next = space.timestep(engine);
                // space.meshless_timestep_limiter(engine); // Note: this can never decrease ti_next
                space.kick1(engine);
            }
            Runner::FluxExtrapolateHalfDrift => {
                space.regrid();
                space.apply_flux(engine);
                space.kick2(engine);
                space.convert_conserved_to_primitive(engine);
                space.meshless_gradient_estimate(engine);
                ti_next = space.timestep(engine);
                // space.meshless_timestep_limiter(engine); // Note: this can never decrease ti_next
                space.kick1(engine);
            }
            _ => {
                panic!("{self:?} should not be run with half steps!")
            }
        }
        space.prepare(engine);
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

    pub fn part_is_active(&self, part: &Particle, iact: Iact, engine: &Engine) -> bool {
        match self {
            Runner::Default
            | Runner::OptimalOrder
            | Runner::TwoGradient
            | Runner::Pakmor
            | Runner::PakmorExtrapolate
            | Runner::VolumeBackExtrapolate => part.is_ending(engine),
            Runner::OptimalOrderHalfDrift => part.is_halfway(engine),
            Runner::DefaultHalfDrift => match iact {
                Iact::ApplyFlux => part.is_ending(engine),
                _ => part.is_halfway(engine),
            },
            Runner::TwoVolumeHalfDrift => match iact {
                Iact::Flux => part.is_halfway(engine),
                Iact::Volume => part.is_ending(engine) || part.is_halfway(engine),
                _ => part.is_ending(engine),
            },
            Runner::MeshlessGradientHalfDrift => match iact {
                Iact::Flux | Iact::Volume => part.is_halfway(engine),
                Iact::Gradient => part.is_halfway(engine) || part.is_ending(engine),
                _ => part.is_ending(engine),
            },
            Runner::FluxExtrapolateHalfDrift => match iact {
                Iact::Flux | Iact::Volume => part.is_halfway(engine),
                _ => part.is_ending(engine),
            },
        }
    }
}
