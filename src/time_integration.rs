use crate::{engine::Engine, space::Space, timeline::IntegerTime};

pub trait Runner {
    fn step(&self, engine: &Engine, space: &mut Space) -> IntegerTime;
}

pub struct DefaultRunner;

impl Runner for DefaultRunner {
    fn step(&self, engine: &Engine, space: &mut Space) -> IntegerTime {
        
        space.kick1(engine);

        space.drift(engine);

        space.sort();

        space.volume_calculation(engine);

        space.convert_conserved_to_primitive(engine);

        space.gradient_estimate(engine);

        space.flux_exchange(engine);

        space.flux_apply(engine);

        space.kick2(engine);

        space.timestep(engine)
    }
}


pub struct OptimalRunner;

impl Runner for OptimalRunner {
    fn step(&self, engine: &Engine, space: &mut Space) -> IntegerTime {

        space.kick1(engine);

        space.drift(engine);

        space.sort();

        space.volume_calculation(engine);

        space.flux_exchange(engine);

        space.flux_apply(engine);

        space.convert_conserved_to_primitive(engine);

        space.gradient_estimate(engine);

        space.kick2(engine);

        space.timestep(engine)
    }
}