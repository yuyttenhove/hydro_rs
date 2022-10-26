use crate::{engine::Engine, space::Space, timeline::IntegerTime};

pub enum Runner {
    Default,
    OptimalOrder,
    OptimalOrderHalfDrift,
}

impl Runner {
    pub fn step(&self, engine: &Engine, space: &mut Space) -> IntegerTime {
        match self {
            Runner::Default => {
                let dt = engine.dt();
                space.kick1(engine);
                space.drift(dt, 0.5 * dt);
                space.sort();
                space.volume_calculation(engine);
                space.convert_conserved_to_primitive(engine);
                space.gradient_estimate(engine);
                space.flux_exchange(engine);
                space.flux_apply(engine);
                space.kick2(engine);
                space.timestep(engine)
            },
            Runner::OptimalOrder => {
                let dt = engine.dt();
                space.kick1(engine);
                space.drift(dt, 0.5 * dt);
                space.sort();
                space.volume_calculation(engine);
                space.flux_exchange(engine);
                space.flux_apply(engine);
                space.convert_conserved_to_primitive(engine);
                space.gradient_estimate(engine);
                space.kick2(engine);
                space.timestep(engine)
            },
            Runner::OptimalOrderHalfDrift => {
                let dt = 0.5 * engine.dt();
                space.kick1(engine);
                space.drift(dt, dt);
                space.sort();
                space.volume_calculation(engine);
                space.flux_exchange(engine);
                space.drift(dt, dt);
                space.flux_apply(engine);
                space.convert_conserved_to_primitive(engine);
                space.gradient_estimate(engine);
                space.kick2(engine);
                let ti_next = space.timestep(engine);
                space.self_check();
                ti_next
            },
        }
    }
}
