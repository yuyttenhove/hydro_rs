use crate::{engine::Engine, errors::ConfigError, space::Space, timeline::{IntegerTime, make_timestep}};

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
        let ti_next;
        match self {
            Runner::Default => {
                let dt = engine.dt();
                space.drift(dt, 0.5 * dt);
                space.sort();
                space.volume_calculation(engine);
                space.convert_conserved_to_primitive(engine);
                space.gradient_estimate(engine);
                space.apply_boundary_condition();
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
                space.sort();
                space.volume_calculation(engine);
                space.flux_exchange(engine);
                space.apply_flux(engine);
                #[cfg(any(dimensionality="2D", dimensionality="3D"))]
                space.add_spherical_source_term(engine);
                space.convert_conserved_to_primitive(engine);
                space.gradient_estimate(engine);
                space.self_gravity(engine);
                space.kick2(engine);
                ti_next = space.timestep(engine);
                space.timestep_limiter(engine); // Note: this can never decrease ti_next
                space.kick1(engine);
            }
            Runner::OptimalOrderHalfDrift | Runner::DefaultHalfDrift => {
                panic!("{self:?} should not be run with full steps!")
            }
        }
        space.prepare(engine);
        space.self_check();
        ti_next
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
                space.apply_boundary_condition();
                space.flux_exchange(engine);
            }
        }
    }

    pub fn half_step2(&self, engine: &Engine, space: &mut Space) -> IntegerTime {
        let ti_next;
        match self {
            Runner::Default | Runner::OptimalOrder => {
                panic!("{self:?} should not be run with half steps!")
            }
            Runner::OptimalOrderHalfDrift => {
                let dt = engine.dt();
                space.drift(0.5 * dt, 0.25 * dt);
                space.apply_flux(engine);
                space.convert_conserved_to_primitive(engine);
                space.gradient_estimate(engine);
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
        }
        space.prepare(engine);
        space.self_check();
        ti_next
    }

    pub fn use_half_step(&self) -> bool {
        match self {
            Runner::DefaultHalfDrift | Runner::OptimalOrderHalfDrift => true,
            _ => false,
        }
    }
}
