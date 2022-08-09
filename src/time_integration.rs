use crate::{engine::Engine, space::Space};

pub trait Runner {
    fn step(&self, engine: &mut Engine, space: &mut Space) -> f64;
}

pub struct DefaultRunner {
    cfl_criterion: f64,
}

impl DefaultRunner {
    pub fn new(cfl_criterion: f64) -> Self {
        Self { cfl_criterion }
    }
}

impl Runner for DefaultRunner {
    fn step(&self, engine: &mut Engine, space: &mut Space) -> f64 {
        
        // Calculate end of timestep for all active particles
        let timestep = space.timestep(self.cfl_criterion);

        space.kick1();

        space.drift();

        space.sort();

        space.volume_calculation();

        space.convert_conserved_to_primitive();

        space.gradient_estimate();

        space.flux_exchange(&engine.solver);

        space.kick2();

        timestep
    }
}


pub struct OptimalRunner {
    cfl_criterion: f64,
}

impl OptimalRunner {
    pub fn new(cfl_criterion: f64) -> Self {
        Self { cfl_criterion }
    }
}

impl Runner for OptimalRunner {
    fn step(&self, engine: &mut Engine, space: &mut Space) -> f64 {
        
        let timestep = space.timestep(self.cfl_criterion);

        space.kick1();

        space.drift();

        space.sort();

        space.volume_calculation();

        space.flux_exchange(&engine.solver);

        space.convert_conserved_to_primitive();

        space.gradient_estimate();

        space.kick2();

        timestep
        
    }
}