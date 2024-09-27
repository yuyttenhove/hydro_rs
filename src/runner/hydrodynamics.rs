use glam::DVec3;
use meshless_voronoi::VoronoiFace;
use rayon::prelude::*;

use crate::{flux::FluxInfo, gradients::{GradientData, LimiterData}, physical_quantities::{Gradients, Primitive}, timeline::{make_integer_timestep, make_timestep, IntegerTime, MAX_NR_TIMESTEPS, NUM_TIME_BINS, TIME_BIN_NEIGHBOUR_MAX_DELTA_BIN}, ParticleMotion, Space, TimestepInfo};

mod optimal_order;
mod godunov;

pub use optimal_order::OptimalOrderRunner;
pub use godunov::GodunovHydroRunner;

fn apply_fluxes(space: &mut Space, fluxes: &[FluxInfo], part_is_active: &[bool]) {
    let faces = &space.voronoi_faces;
    let cell_face_connections = &space.voronoi_cell_face_connections;
    space.parts.par_iter_mut().enumerate().for_each(|(part_idx, part)|{
        let face_idx: &[usize] = {
            let start = part.face_connections_offset;
            let end = start + part.face_count;
            &cell_face_connections[start..end]
        };
        let active = part_is_active[part_idx];
        for &idx in face_idx {
            let face = &faces[idx];
            let flux = &fluxes[idx];
            if face.left() == part_idx {
                part.update_fluxes_left(flux, active);
            } else {
                assert!(face.right().is_some());
                assert_eq!(face.right().expect("Right is not None"), part_idx);
                part.update_fluxes_right(flux, active);
            }
        }

        if active {
            part.apply_flux();
        }
    });
}

fn kick1(space: &mut Space, part_is_active: &[bool], motion: ParticleMotion) {
    let dimensionality = space.dimensionality();
    space.parts_mut().par_iter_mut().enumerate().for_each(|(part_idx, part)| {
        if part_is_active[part_idx] {
            part.hydro_kick1(&motion, dimensionality);
                part.grav_kick();
        }
    });
}

fn kick2(space: &mut Space, part_is_active: &[bool]) {
    space.parts_mut().par_iter_mut().enumerate().for_each(|(part_idx, part)| {
        if part_is_active[part_idx] {
            part.grav_kick();
        }
    });
}

fn get_other(face: &VoronoiFace, part_idx: usize) -> Option<usize> {
    if part_idx == face.left() {
        face.right()
    } else {
        Some(face.left())
    }
}

fn gradient_estimate(space: &Space, part_is_active: &[bool]) -> Vec<Option<Gradients<Primitive>>> {
    let faces = space.faces();
    let cell_face_connections = space.cell_face_connections();
    space.parts().par_iter().enumerate().map(|(part_idx, part)| {
        if !part_is_active[part_idx] { return None }

        let centroid = part.centroid;
        let face_idx: &[usize] = {
            let start = part.face_connections_offset;
            let end = start + part.face_count;
            &cell_face_connections[start..end]
        };

        let mut gradient_data = GradientData::init(space.dimensionality());
        for &idx in face_idx {
            let face = &faces[idx];
            let area = face.area();
            let shift = face.shift().unwrap_or(DVec3::ZERO);
            let shift = if part_idx == face.left() {shift} else { -shift };
            match get_other(face, part_idx) {
                Some(other_idx) => {
                    let other = & space.parts()[other_idx];
                    let ds = other.centroid + shift - centroid;
                    gradient_data.collect(&part.primitives, &other.primitives, area / ds.length_squared(), ds)
                },
                None => {
                    let other = &space.get_boundary_part(part, face);
                    let ds = other.centroid + shift - centroid;
                    gradient_data.collect(&part.primitives, &other.primitives, area / ds.length_squared(), ds)
                }
            };
        }

        Some(gradient_data.finalize())
    }).collect()
}

fn gradient_limit(space: &Space, gradients: &mut [Option<Gradients<Primitive>>]) {
    let faces = space.faces();
    let cell_face_connections = space.cell_face_connections();
    space.parts().par_iter().enumerate().zip(gradients.par_iter_mut()).for_each(|((part_idx, part), gradients)| {
        if let Some(gradients) = gradients {

            let centroid = part.centroid;
            let face_idx: &[usize] = {
                let start = part.face_connections_offset;
                let end = start + part.face_count;
                &cell_face_connections[start..end]
            };
            
            let mut limiter = LimiterData::init(&part.primitives);
            for &idx in face_idx {
                let face = &faces[idx];
                let shift = face.shift().unwrap_or(DVec3::ZERO);
                let shift = if part_idx == face.left() {shift} else { -shift };
                let extrapolated = gradients.dot(face.centroid() - centroid - shift);
                let other_primitives = match get_other(face, part_idx) {
                    Some(other_idx) => space.parts()[other_idx].primitives,
                    None => space.get_boundary_part(part, face).primitives,
                };
                limiter.collect(&other_primitives, &extrapolated);
            }

            limiter.limit(gradients, &part.primitives);
            debug_assert!(gradients.is_finite());
        }
    });
}

fn gradient_apply(space: &mut Space, gradients: &[Option<Gradients<Primitive>>]) {
    space.parts_mut().par_iter_mut().zip(gradients.par_iter()).for_each(|(part, gradients)| {
        if let Some(gradients) = gradients {
            part.gradients = *gradients;
            part.gradients_centroid = part.centroid;
        }
    });
}

fn timesteps_apply(space: &mut Space, timesteps: &[f64], part_is_active: &[bool], timestep_info: &TimestepInfo) -> IntegerTime {
    space.parts_mut().par_iter_mut().zip(timesteps.par_iter()).zip(part_is_active.par_iter()).map(|((part, dt), active)| {
        if !active { return MAX_NR_TIMESTEPS; }
        let dti = make_integer_timestep(
            *dt,
            part.timebin,
            /*TODO*/ NUM_TIME_BINS,
            timestep_info.ti_current,
            timestep_info.time_base_inv,
        );
        part.set_timestep(make_timestep(dti, timestep_info.time_base), dti);
        dti
    }).min().expect("At least one particle must be active")
} 

fn timestep_limiter(space: &mut Space, part_is_active: &[bool], timestep_info: &TimestepInfo) {
    let faces = space.faces();
    let cell_face_connections = space.cell_face_connections();
    let parts = space.parts();
    
    let wakeups: Vec<_> = parts.par_iter().enumerate().map(|(part_idx, part)| {
        let face_idx: &[usize] = {
            let start = part.face_connections_offset;
            let end = start + part.face_count;
            &cell_face_connections[start..end]
        };
        let mut wakeup = part.timebin;
        for &idx in face_idx {
            if let Some(other_idx) = get_other(&faces[idx], part_idx) {
                if !part_is_active[other_idx] { continue; }
                wakeup = wakeup.min(parts[other_idx].timebin + TIME_BIN_NEIGHBOUR_MAX_DELTA_BIN);
            }
        }
        wakeup
    }).collect();

    space.parts_mut().par_iter_mut().zip(wakeups.par_iter()).for_each(|(part, wakeup)| {
        part.timestep_limit(*wakeup, timestep_info);
    });
}

fn timestep_sync(space: &mut Space, timestep_info: &TimestepInfo) {
    let min_timebin = space.parts().par_iter().map(|part| part.timebin).min().expect("Parts cannot be empty");
    space.parts_mut().par_iter_mut().for_each(|part| {
        part.timestep_limit(min_timebin, timestep_info);
    });
}

fn reset_extrapolations(space: &mut Space, part_is_active: &[bool]) {
    space.parts_mut().par_iter_mut().zip(part_is_active).for_each(|(part, active)| if *active {part.reset_extrapolations()});
}