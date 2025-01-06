use glam::DVec3;
use meshless_voronoi::VoronoiFace;
use rayon::prelude::*;

use crate::{
    finite_volume_solver::FluxInfo,
    gradients::{GradientData, LimiterData},
    physical_quantities::{Gradients, Primitive},
    timeline::{
        make_integer_timestep, make_timestep, IntegerTime, MAX_NR_TIMESTEPS, NUM_TIME_BINS,
        TIME_BIN_NEIGHBOUR_MAX_DELTA_BIN,
    },
    ParticleMotion, Space, TimestepInfo,
};

mod optimal_order;

use crate::finite_volume_solver::{FiniteVolumeSolver, FluxLimiterData};
use crate::physical_quantities::State;
use crate::riemann_solver::{RiemannStarSolver, RiemannWafFluxSolver};
pub use optimal_order::OptimalOrderRunner;

fn apply_fluxes(space: &mut Space, fluxes: &[FluxInfo], part_is_active: &[bool]) {
    let faces = &space.voronoi_faces;
    let cell_face_connections = &space.voronoi_cell_face_connections;
    space
        .parts
        .par_iter_mut()
        .enumerate()
        .for_each(|(part_idx, part)| {
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
    space
        .parts_mut()
        .par_iter_mut()
        .enumerate()
        .for_each(|(part_idx, part)| {
            if part_is_active[part_idx] {
                part.hydro_kick1(&motion, dimensionality);
                part.grav_kick();
            }
        });
}

fn kick2(space: &mut Space, part_is_active: &[bool]) {
    space
        .parts_mut()
        .par_iter_mut()
        .enumerate()
        .for_each(|(part_idx, part)| {
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
    space
        .parts()
        .par_iter()
        .enumerate()
        .map(|(part_idx, part)| {
            if !part_is_active[part_idx] {
                return None;
            }

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
                let shift = if part_idx == face.left() {
                    shift
                } else {
                    -shift
                };
                match get_other(face, part_idx) {
                    Some(other_idx) => {
                        let other = &space.parts()[other_idx];
                        let ds = other.centroid + shift - centroid;
                        gradient_data.collect(
                            &part.primitives,
                            &other.primitives,
                            area / ds.length_squared(),
                            ds,
                        )
                    }
                    None => {
                        let other = &space.get_boundary_part(part, face);
                        let ds = other.centroid + shift - centroid;
                        gradient_data.collect(
                            &part.primitives,
                            &other.primitives,
                            area / ds.length_squared(),
                            ds,
                        )
                    }
                };
            }

            Some(gradient_data.finalize())
        })
        .collect()
}

/// 1D slope limiters a la Toro 2009
fn slope_limiter(space: &Space, gradients: &mut [Option<Gradients<Primitive>>]) {
    // Compute flow parameter for each particle
    let faces = space.faces();
    let cell_face_connections = space.cell_face_connections();

    let flow_r: Vec<_> = space
        .parts
        .iter()
        .enumerate()
        .zip(gradients.iter())
        .map(|((part_idx, part), grad)| {
            if let Some(grad) = grad {
                let face_idx: &[usize] = {
                    let start = part.face_connections_offset;
                    let end = start + part.face_count;
                    &cell_face_connections[start..end]
                };
                debug_assert!(face_idx.len() == 2);
                let mut ngb_states = [State::vacuum(); 2];
                for &idx in face_idx {
                    let face = &faces[idx];
                    let mut centroid = face.centroid();
                    if part_idx != face.left() {
                        if let Some(shift) = face.shift() {
                            centroid += shift;
                        }
                    }
                    let other = match get_other(face, part_idx) {
                        Some(other_idx) => space.parts()[other_idx].primitives,
                        None => space.get_boundary_part(part, face).primitives,
                    };
                    if centroid.x < part.loc.x {
                        ngb_states[0] = other;
                    } else {
                        ngb_states[1] = other;
                    }
                }
                let mut ratios = [DVec3::ZERO; 5];
                for i in 0..5 {
                    let slope_prev = part.primitives[i] - ngb_states[0][i];
                    let slope_next = ngb_states[1][i] - part.primitives[i];
                    if slope_next != 0. {
                        ratios[i] = DVec3::new(slope_prev / slope_next, slope_prev, slope_next);
                    };
                }
                Some(ratios)
            } else {
                None
            }
        })
        .collect();

    // Now apply slope limiters
    flow_r
        .iter()
        .enumerate()
        .zip(gradients.iter_mut())
        .for_each(|((i, limiter_info), grad)| {
            if let Some(grad) = grad {
                let limiter_info = limiter_info.expect("cannot be none for Some gradients");
                let dx = space.parts[i].volume;
                for i in 0..5 {
                    let r = limiter_info[i].x;
                    let xi_l = 2. / (1. + r);
                    let xi_r = 2. * r / (1. + r);
                    // vanleer limiter
                    // let xi = if r < 0. { 0. } else { (2. * r / (1. + r)).min(xi_l).min(xi_r) };
                    // minbee
                    // let xi = if r < 0. { 0. } else { r.min(1.).min(xi_l).min(xi_r) };
                    // superbee
                    // let xi = if r < 0. {
                    //     0.
                    // } else if r < 1. {
                    //     1f64.min(2. * r)
                    // } else {
                    //     r.min(1.).min(xi_l).min(xi_r)
                    // };
                    // grad[i] *= xi;
                    // Compute limited slopes directly
                    let slope_prev = limiter_info[i].y;
                    let slope_next = limiter_info[i].z;
                    // Minbee
                    let beta = 1.;
                    // Superbee
                    let beta = 2.;
                    // limited slope
                    grad[i] = if slope_next > 0. {
                        0f64.max(slope_next.min(beta * slope_prev))
                            .max(slope_prev.min(beta * slope_next))
                    } else {
                        0f64.min(slope_next.max(beta * slope_prev))
                            .min(slope_prev.max(beta * slope_next))
                    } * DVec3::X
                        / dx;
                }
            }
        })
}

fn gradient_limit(space: &Space, gradients: &mut [Option<Gradients<Primitive>>]) {
    let faces = space.faces();
    let cell_face_connections = space.cell_face_connections();
    space
        .parts()
        .par_iter()
        .enumerate()
        .zip(gradients.par_iter_mut())
        .for_each(|((part_idx, part), gradients)| {
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
                    let shift = if part_idx == face.left() {
                        shift
                    } else {
                        -shift
                    };
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
    space
        .parts_mut()
        .par_iter_mut()
        .zip(gradients.par_iter())
        .for_each(|(part, gradients)| {
            if let Some(gradients) = gradients {
                part.gradients = *gradients;
                part.gradients_centroid = part.centroid;
            }
        });
}

fn apply_flux_limiter(
    space: &mut Space,
    flux_limiters: &[FluxLimiterData],
    part_is_active: &[bool],
) {
    let faces = &space.voronoi_faces;
    let cell_face_connections = &space.voronoi_cell_face_connections;
    space
        .parts
        .par_iter_mut()
        .enumerate()
        .for_each(|(part_idx, part)| {
            let active = part_is_active[part_idx];
            if !active {
                return;
            }
            part.flux_limiter = FluxLimiterData::zero();
            let face_idx: &[usize] = {
                let start = part.face_connections_offset;
                let end = start + part.face_count;
                &cell_face_connections[start..end]
            };
            for &idx in face_idx {
                let face = &faces[idx];
                if face.left() == part_idx {
                    part.flux_limiter.combine(flux_limiters[idx]);
                } else {
                    part.flux_limiter.combine(-flux_limiters[idx]);
                }
            }
        });
}

fn timesteps_apply(
    space: &mut Space,
    timesteps: &[f64],
    part_is_active: &[bool],
    timestep_info: &TimestepInfo,
) -> IntegerTime {
    space
        .parts_mut()
        .par_iter_mut()
        .zip(timesteps.par_iter())
        .zip(part_is_active.par_iter())
        .map(|((part, dt), active)| {
            if !active {
                return MAX_NR_TIMESTEPS;
            }
            if *dt < timestep_info.dt_min {
                panic!("Particle wants timestep smaller than dt_min!");
            }
            let dti = make_integer_timestep(
                dt.min(timestep_info.dt_max),
                part.timebin,
                /*TODO*/ NUM_TIME_BINS,
                timestep_info.ti_current,
                timestep_info.time_base_inv,
            );
            part.set_timestep(make_timestep(dti, timestep_info.time_base), dti);
            dti
        })
        .min()
        .expect("At least one particle must be active")
}

fn timestep_limiter(space: &mut Space, part_is_active: &[bool], timestep_info: &TimestepInfo) {
    let faces = space.faces();
    let cell_face_connections = space.cell_face_connections();
    let parts = space.parts();

    let wakeups: Vec<_> = parts
        .par_iter()
        .enumerate()
        .map(|(part_idx, part)| {
            let face_idx: &[usize] = {
                let start = part.face_connections_offset;
                let end = start + part.face_count;
                &cell_face_connections[start..end]
            };
            let mut wakeup = part.timebin;
            for &idx in face_idx {
                if let Some(other_idx) = get_other(&faces[idx], part_idx) {
                    if !part_is_active[other_idx] {
                        continue;
                    }
                    wakeup =
                        wakeup.min(parts[other_idx].timebin + TIME_BIN_NEIGHBOUR_MAX_DELTA_BIN);
                }
            }
            wakeup
        })
        .collect();

    space
        .parts_mut()
        .par_iter_mut()
        .zip(wakeups.par_iter())
        .for_each(|(part, wakeup)| {
            part.timestep_limit(*wakeup, timestep_info);
        });
}

fn timestep_sync(space: &mut Space, timestep_info: &TimestepInfo) {
    let min_timebin = space
        .parts()
        .par_iter()
        .map(|part| part.timebin)
        .min()
        .expect("Parts cannot be empty");
    space.parts_mut().par_iter_mut().for_each(|part| {
        part.timestep_limit(min_timebin, timestep_info);
    });
}

fn reset_extrapolations(space: &mut Space, part_is_active: &[bool]) {
    space
        .parts_mut()
        .par_iter_mut()
        .zip(part_is_active)
        .for_each(|(part, active)| {
            if *active {
                part.reset_extrapolations()
            }
        });
}
