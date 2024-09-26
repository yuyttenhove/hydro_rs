use crate::{finite_volume_solver::FiniteVolumeSolver, gravity::GravitySolver, timeline::IntegerTime, ParticleMotion, Space, TimestepInfo};

use super::{apply_fluxes, gradient_apply, gradient_estimate, gradient_limit, kick1, kick2, timestep_limiter, timestep_sync, timesteps_apply};
use crate::runner::Runner;
use rayon::prelude::*;

pub struct OptimalOrderRunner;


impl Runner for OptimalOrderRunner {
    fn use_half_step(&self) -> bool {
        false
    }

    fn step(&self, space: &mut Space, fv_solver: &Box<dyn FiniteVolumeSolver>, gravity_solver: &Option<Box<dyn GravitySolver>>, timestep_info: &TimestepInfo, sync_all: bool, particle_motion: ParticleMotion) -> IntegerTime {
        // Get the mask of active parts
        let part_is_active: Vec<bool> = space.parts.iter().map(|part| { timestep_info.bin_is_ending(part.timebin) }).collect();
        
        // Extrapolate forward in time
        let dt = timestep_info.dt_from_dti(timestep_info.ti_current - timestep_info.ti_old);
        fv_solver.predict(space.parts_mut(), dt);

        // Update Voronoi grid
        match particle_motion {
            ParticleMotion::Fixed => (),
            _ => space.volume_calculation(&part_is_active),
        }

        // Fluxes
        let fluxes = fv_solver.compute_fluxes(space.faces(), space.parts(), space.boundary());
        apply_fluxes(space, &fluxes, &part_is_active);

        // TODO: gravity

        kick2(space, &part_is_active);

        fv_solver.convert_conserved_to_primitive(space.parts_mut(), &part_is_active);
        
        // Compute, limit and apply gradients
        let mut gradients = gradient_estimate(space, &part_is_active);
        gradient_limit(space, &mut gradients); 
        gradient_apply(space, &gradients);

        // Timestep
        let dimensionality = space.dimensionality();
        let mut timesteps = fv_solver.compute_timesteps(space.parts_mut(), &part_is_active, particle_motion, dimensionality);
        if let Some(gravity_solver) = gravity_solver {
            let timesteps_grav = gravity_solver.compute_timesteps(space.parts()); 
            timesteps.par_iter_mut().zip(timesteps_grav.par_iter()).for_each(|(dt_fv, dt_grav)| *dt_fv = dt_fv.min(*dt_grav));
        }
        let ti_next = timestep_info.ti_current + timesteps_apply(space, &timesteps, &part_is_active, timestep_info);
        // NOTE: this can never *decrease* ti_next...
        if sync_all {
            timestep_sync(space, timestep_info);
        } else {
            timestep_limiter(space, &part_is_active, timestep_info);
        }

        kick1(space, &part_is_active, particle_motion);

        ti_next
    }
    
    fn label(&self) -> String {
        "optimal".to_string()
    }
}