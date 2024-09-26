use crate::{riemann_solver::RiemannFluxSolver, timeline::IntegerTime, ParticleMotion, TimestepInfo};

use super::{apply_fluxes, compute_fluxes, convert_conserved_to_primitive, extrapolate_state, gradient_apply, gradient_estimate, gradient_limit, kick1, kick2, timestep, timestep_hydro, timestep_limiter, timestep_sync, HydroSolver};
use crate::runner::Runner;

pub struct OptimalOrderRunner<R: RiemannFluxSolver> {
    hydro_solver: HydroSolver<R>,
}

impl<R: RiemannFluxSolver> OptimalOrderRunner<R> {
    pub fn new(hydro_solver: HydroSolver<R>) -> Self {
        Self {
            hydro_solver
        }
    }
}

impl<R: RiemannFluxSolver> Runner for OptimalOrderRunner<R> {
    fn use_half_step(&self) -> bool {
        false
    }

    fn step(&self, space: &mut crate::Space, timestep_info: &TimestepInfo, sync_all: bool, particle_motion: ParticleMotion) -> IntegerTime {
        // Get the mask of active parts
        let part_is_active: Vec<bool> = space.parts.iter().map(|part| { timestep_info.bin_is_ending(part.timebin) }).collect();
        
        // Extrapolate forward in time
        let dt = timestep_info.dt_from_dti(timestep_info.ti_current - timestep_info.ti_old);
        extrapolate_state(space, dt, &self.hydro_solver.gas_law);

        // Update Voronoi grid
        match particle_motion {
            ParticleMotion::Fixed => (),
            _ => space.volume_calculation(&part_is_active),
        }

        // Fluxes
        let fluxes = compute_fluxes(space, &self.hydro_solver.gas_law, &self.hydro_solver.riemann_solver);
        apply_fluxes(space, &fluxes, &part_is_active);

        // TODO: gravity

        kick2(space, &part_is_active);

        convert_conserved_to_primitive(space, &part_is_active, &self.hydro_solver.gas_law);
        
        // Compute, limit and apply gradients
        let mut gradients = gradient_estimate(space, &part_is_active);
        gradient_limit(space, &mut gradients); 
        gradient_apply(space, &gradients);

        // Timestep
        let timesteps = timestep_hydro(space, &part_is_active, timestep_info, self.hydro_solver.cfl, particle_motion, &self.hydro_solver.gas_law);
        let ti_next = timestep_info.ti_current + timestep(space, &timesteps, timestep_info);
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