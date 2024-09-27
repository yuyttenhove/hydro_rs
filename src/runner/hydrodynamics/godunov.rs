use crate::{ParticleMotion, Runner};

use super::{apply_fluxes, kick1, kick2, reset_extrapolations, timestep_limiter, timestep_sync, timesteps_apply};

use rayon::prelude::*;

pub struct GodunovHydroRunner;

impl Runner for GodunovHydroRunner {
    fn use_half_step(&self) -> bool {
        false
    }

    fn step(&self, space: &mut crate::Space, fv_solver: &Box<dyn crate::finite_volume_solver::FiniteVolumeSolver>, gravity_solver: &Option<Box<dyn crate::gravity::GravitySolver>>, timestep_info: &crate::TimestepInfo, sync_all: bool, particle_motion: crate::ParticleMotion) -> crate::timeline::IntegerTime {
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
        let fluxes = fv_solver.compute_fluxes(space.faces(), space.parts(), &part_is_active, space.boundary());
        apply_fluxes(space, &fluxes, &part_is_active);

        // TODO: gravity

        kick2(space, &part_is_active);

        fv_solver.convert_conserved_to_primitive(space.parts_mut(), &part_is_active);

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

        reset_extrapolations(space, &part_is_active);

        kick1(space, &part_is_active, particle_motion);

        ti_next
    }

    fn label(&self) -> String {
        "godunov".to_string()
    }
}