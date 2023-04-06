use crate::{
    engine::Engine, errors::ConfigError, part::Particle, space::Space, timeline::IntegerTime,
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
    TwoGradientHalfDrift,
    OptimalOrderHalfDrift,
    DefaultHalfDrift,
    MeshlessGradientHalfDrift,
}

impl Runner {
    pub fn new(kind: &str) -> Result<Runner, ConfigError> {
        match kind {
            "Default" => Ok(Runner::Default),
            "OptimalOrder" => Ok(Runner::OptimalOrder),
            "TwoGradient" => Ok(Runner::TwoGradient),
            "TwoGradientHalfDrift" => Ok(Runner::TwoGradientHalfDrift),
            "OptimalOrderHalfDrift" => Ok(Runner::OptimalOrderHalfDrift),
            "DefaultHalfDrift" => Ok(Runner::DefaultHalfDrift),
            "MeshlessGradientHalfDrift" => Ok(Runner::MeshlessGradientHalfDrift),
            _ => Err(ConfigError::UnknownRunner(kind.to_string())),
        }
    }

    pub fn step(&self, engine: &Engine, space: &mut Space) -> IntegerTime {
        let ti_next;
        match self {
            Runner::Default => {
                let dt = engine.dt();
                space.drift(dt, 0.5 * dt);
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
                let dt = engine.dt();
                space.drift(dt, 0.5 * dt);
                space.volume_calculation(engine);
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
            Runner::TwoGradient => {
                let dt = engine.dt();
                space.drift(dt, 0.5 * dt);
                space.volume_calculation(engine);
                // space.convert_conserved_to_primitive(engine);
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
            Runner::TwoGradientHalfDrift => {
                let dt = engine.dt();
                space.drift(dt, 0.5 * dt);
                space.volume_calculation(engine);
                // space.convert_conserved_to_primitive(engine);
                space.gradient_estimate(engine);
                space.flux_exchange(engine);
                space.apply_flux(engine);
                space.convert_conserved_to_primitive(engine);
                space.gradient_estimate(engine);
            }
            Runner::OptimalOrderHalfDrift => {
                let dt = engine.dt();
                space.drift(dt, 0.5 * dt);
                space.volume_calculation(engine);
                space.flux_exchange(engine);
                space.apply_flux(engine);
                space.convert_conserved_to_primitive(engine);
                space.gradient_estimate(engine);
            }
            Runner::DefaultHalfDrift => {
                let dt = engine.dt();
                space.drift(dt, 0.5 * dt);
                space.volume_calculation(engine);
                space.convert_conserved_to_primitive(engine);
                space.gradient_estimate(engine);
                space.flux_exchange(engine);
            }
            Runner::MeshlessGradientHalfDrift => {
                let dt = engine.dt();
                space.drift(dt, dt);
                space.volume_calculation(engine);
                space.flux_exchange(engine);
            }
            _ => {
                panic!("{self:?} should not be run with half steps!")
            }
        }
    }

    pub fn half_step2(&self, engine: &Engine, space: &mut Space) -> IntegerTime {
        let ti_next;
        match self {
            Runner::TwoGradientHalfDrift => {
                let dt = engine.dt();
                space.drift(0.5 * dt, 0.25 * dt);
                space.kick2(engine);
                ti_next = space.timestep(engine);
                space.timestep_limiter(engine); // Note: this can never decrease ti_next
                space.kick1(engine);
            }
            Runner::OptimalOrderHalfDrift => {
                let dt = engine.dt();
                space.drift(0.5 * dt, 0.25 * dt);
                space.kick2(engine);
                ti_next = space.timestep(engine);
                space.timestep_limiter(engine); // Note: this can never decrease ti_next
                space.kick1(engine);
            }
            Runner::DefaultHalfDrift => {
                let dt = engine.dt();
                space.drift(0.5 * dt, 0.25 * dt);
                space.apply_flux(engine);
                space.kick2(engine);
                ti_next = space.timestep(engine);
                space.timestep_limiter(engine); // Note: this can never decrease ti_next
                space.kick1(engine);
            }
            Runner::MeshlessGradientHalfDrift => {
                let dt = engine.dt();
                space.drift(0.5 * dt, 0.5 * dt);
                space.regrid();
                space.apply_flux(engine);
                space.kick2(engine);
                space.convert_conserved_to_primitive(engine);
                space.meshless_gradient_estimate(engine);
                ti_next = space.timestep(engine);
                space.meshless_timestep_limiter(engine); // Note: this can never decrease ti_next
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
            | Runner::TwoGradientHalfDrift
            | Runner::MeshlessGradientHalfDrift => true,
            _ => false,
        }
    }

    pub fn part_is_active(&self, part: &Particle, iact: Iact, engine: &Engine) -> bool {
        match self {
            Runner::Default | Runner::OptimalOrder | Runner::TwoGradient => part.is_ending(engine),
            Runner::OptimalOrderHalfDrift | Runner::TwoGradientHalfDrift => part.is_halfway(engine),
            Runner::DefaultHalfDrift => match iact {
                Iact::ApplyFlux => part.is_ending(engine),
                _ => part.is_halfway(engine),
            },
            Runner::MeshlessGradientHalfDrift => {
                match iact {
                    Iact::Flux | Iact::Volume => part.is_halfway(engine),
                    _ => part.is_ending(engine),
                }
            }
        }
    }
}
