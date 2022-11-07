use crate::{engine::Engine, space::Space, timeline::IntegerTime, errors::ConfigError};

pub enum Runner {
    Default,
    OptimalOrder,
    OptimalOrderHalfDrift,
}

impl Runner {

    pub fn new(kind: &str) -> Result<Runner, ConfigError> {
        match kind {
            "Default" => Ok(Runner::Default),
            "OptimalOrder" => Ok(Runner::OptimalOrder),
            "OptimalOrderHalfDrift" => Ok(Runner::OptimalOrderHalfDrift),
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
                space.timestep_limiter(engine);  // Note: this can never decrease ti_next
                space.kick1(engine);
                space.self_check();
                ti_next
            },
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
                space.timestep_limiter(engine);  // Note: this can never decrease ti_next
                space.kick1(engine);
                space.self_check();
                ti_next
            },
            Runner::OptimalOrderHalfDrift => {
                let dt = 0.5 * engine.dt();
                space.drift(dt, dt);
                space.sort();
                space.volume_calculation(engine);
                space.flux_exchange(engine);
                space.drift(dt, dt);
                space.apply_flux(engine);
                space.convert_conserved_to_primitive(engine);
                space.gradient_estimate(engine);
                space.kick2(engine);
                let ti_next = space.timestep(engine);
                space.timestep_limiter(engine);  // Note: this can never decrease ti_next
                space.kick1(engine);
                space.self_check();
                ti_next
            },
        }
    }
}
