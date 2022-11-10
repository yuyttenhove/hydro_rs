use crate::{engine::Engine, errors::ConfigError, space::Space, timeline::IntegerTime};

#[derive(Debug)]
pub enum Runner {
    Default,
    OptimalOrder,
    OptimalOrderHalfDrift,
    DefaultHalfDrift,
}

impl Runner {
    pub fn new(kind: &str) -> Result<Runner, ConfigError> {
        match kind {
            "Default" => Ok(Runner::Default),
            "OptimalOrder" => Ok(Runner::OptimalOrder),
            "OptimalOrderHalfDrift" => Ok(Runner::OptimalOrderHalfDrift),
            "DefaultHalfDrift" => Ok(Runner::DefaultHalfDrift),
            _ => Err(ConfigError::UnknownRunner(kind.to_string())),
        }
    }

    pub fn step(&self, engine: &Engine, space: &mut Space) -> IntegerTime {
        match self {
            Runner::Default => {
                let dt = engine.dt();
                space.drift(dt, 0.5 * dt);
                space.sort();
                space.volume_calculation(engine);
                space.convert_conserved_to_primitive(engine);
                space.gradient_estimate(engine);
                space.flux_exchange(engine);
                space.apply_flux(engine);
                space.kick2(engine);
                let ti_next = space.timestep(engine);
                space.timestep_limiter(engine); // Note: this can never decrease ti_next
                space.kick1(engine);
                space.self_check();
                ti_next
            }
            Runner::OptimalOrder => {
                let dt = engine.dt();
                space.drift(dt, 0.5 * dt);
                space.sort();
                space.volume_calculation(engine);
                space.flux_exchange(engine);
                space.apply_flux(engine);
                space.convert_conserved_to_primitive(engine);
                space.gradient_estimate(engine);
                space.kick2(engine);
                let ti_next = space.timestep(engine);
                space.timestep_limiter(engine); // Note: this can never decrease ti_next
                space.kick1(engine);
                space.self_check();
                ti_next
            }
            Runner::OptimalOrderHalfDrift | Runner::DefaultHalfDrift => {
                panic!("{self:?} should not be run with full steps!")
            }
        }
    }

    pub fn half_step1(&self, engine: &Engine, space: &mut Space) {
        match self {
            Runner::Default | Runner::OptimalOrder => {
                panic!("{self:?} should not be run with half steps!")
            }
            Runner::OptimalOrderHalfDrift => {
                let dt = engine.dt();
                space.drift(dt, 0.5 * dt);
                space.sort();
                space.volume_calculation(engine);
                space.flux_exchange(engine);
            }
            Runner::DefaultHalfDrift => {
                let dt = engine.dt();
                space.drift(dt, 0.5 * dt);
                space.sort();
                space.volume_calculation(engine);
                space.convert_conserved_to_primitive(engine);
                space.gradient_estimate(engine);
                space.flux_exchange(engine);
            }
        }
    }

    pub fn half_step2(&self, engine: &Engine, space: &mut Space) -> IntegerTime {
        match self {
            Runner::Default | Runner::OptimalOrder => {
                panic!("{self:?} should not be run with half steps!")
            }
            Runner::OptimalOrderHalfDrift => {
                let dt = 0.5 * engine.dt();
                space.drift(dt, 0.5 * dt);
                space.apply_flux(engine);
                space.convert_conserved_to_primitive(engine);
                space.gradient_estimate(engine);
                space.kick2(engine);
                let ti_next = space.timestep(engine);
                space.timestep_limiter(engine); // Note: this can never decrease ti_next
                space.kick1(engine);
                space.self_check();
                ti_next
            }
            Runner::DefaultHalfDrift => {
                let dt = 0.5 * engine.dt();
                space.drift(dt, 0.5 * dt);
                space.apply_flux(engine);
                space.kick2(engine);
                let ti_next = space.timestep(engine);
                space.timestep_limiter(engine); // Note: this can never decrease ti_next
                space.kick1(engine);
                space.self_check();
                ti_next
            }
        }
    }

    pub fn use_half_step(&self) -> bool {
        match self {
            Runner::DefaultHalfDrift | Runner::OptimalOrderHalfDrift => true,
            _ => false,
        }
    }
}
