use crate::{finite_volume_solver::FiniteVolumeSolver, gravity::GravitySolver, timeline::IntegerTime, ParticleMotion, Space, TimestepInfo};

pub mod hydrodynamics;

#[allow(unused)]
pub trait Runner {
    fn use_half_step(&self) -> bool;

    fn label(&self) -> String;

    fn step(&self, space: &mut Space, fv_solver: &Box<dyn FiniteVolumeSolver>, gravity_solver: &Option<Box<dyn GravitySolver>>, timestep_info: &TimestepInfo, sync_all: bool, particle_motion: ParticleMotion) -> IntegerTime {
        unreachable!("This runner should be run with half steps!");
    }

    fn half_step1(&self, space: &mut Space, fv_solver: &Box<dyn FiniteVolumeSolver>, gravity_solver: &Option<Box<dyn GravitySolver>>, timestep_info: &TimestepInfo, sync_all: bool, particle_motion: ParticleMotion) -> IntegerTime {
        unreachable!("This runner should be run with full steps!");
    }

    fn half_step2(&self, space: &mut Space, fv_solver: &Box<dyn FiniteVolumeSolver>, gravity_solver: &Option<Box<dyn GravitySolver>>, timestep_info: &TimestepInfo, sync_all: bool, particle_motion: ParticleMotion) -> IntegerTime {
        unreachable!("This runner should be run with full steps!");
    }
}