use glam::DVec3;
use meshless_voronoi::VoronoiFace;
use std::ops::Neg;

use crate::{
    gas_law::GasLaw,
    part::Particle,
    physical_quantities::{Conserved, State},
    Boundary, Dimensionality, ParticleMotion,
};

use rayon::prelude::*;

mod godunov_fvs;
mod muscl_fvs;
mod waf_fvs;

use crate::gradients::LimiterData;
use crate::physical_quantities::{Gradients, Primitive};
use crate::utils::{get_boundary_part, get_other};
pub use godunov_fvs::GodunovFvs;
pub use muscl_fvs::MusclFvs;
pub use waf_fvs::WafFvs;

pub trait FiniteVolumeSolver: Sync {
    fn predict(&self, particles: &mut [Particle], dt: f64);

    fn compute_fluxes(
        &self,
        faces: &[VoronoiFace],
        particles: &[Particle],
        part_is_active: &[bool],
        boundary: Boundary,
    ) -> Vec<FluxInfo>;

    fn convert_conserved_to_primitive(&self, particles: &mut [Particle], part_is_active: &[bool]) {
        particles
            .par_iter_mut()
            .enumerate()
            .for_each(|(part_idx, part)| {
                if part_is_active[part_idx] {
                    part.convert_conserved_to_primitive(self.eos());
                }
            });
    }

    fn compute_timesteps(
        &self,
        particles: &mut [Particle],
        part_is_active: &[bool],
        particle_motion: ParticleMotion,
        dimensionality: Dimensionality,
    ) -> Vec<f64> {
        particles
            .par_iter_mut()
            .enumerate()
            .map(|(part_idx, part)| {
                if !part_is_active[part_idx] {
                    return std::f64::INFINITY;
                }
                // Compute new hydro timestep
                part.timestep(self.cfl(), particle_motion, self.eos(), dimensionality)
            })
            .collect()
    }

    fn eos(&self) -> &GasLaw;

    fn cfl(&self) -> f64;

    fn do_gradients(&self) -> bool {
        false
    }

    fn do_gradients_limit(&self) -> bool {
        false
    }

    fn gradient_limit(
        &self,
        gradients: &mut [Option<Gradients<Primitive>>],
        particles: &[Particle],
        part_is_active: &[bool],
        faces: &[VoronoiFace],
        cell_face_connections: &[usize],
        boundary: Boundary,
    ) {
        // Only use this function when it is implemented properly!
        unimplemented!("Shouldn't call this function!")
    }

    fn do_flux_limit(&self) -> bool {
        false
    }

    fn flux_limit_collect(
        &self,
        faces: &[VoronoiFace],
        particles: &[Particle],
        part_is_active: &[bool],
        boundary: Boundary,
    ) -> Vec<FluxLimiterData> {
        unimplemented!("Shouldn't call this function!")
    }
}

pub enum GradientLimiter {
    None,
    /// General gradient limiter as in Springel 2010, Hopkins 2015
    Gradients,
    /// 1D slope limiter as in Toro 2009
    Slopes(SlopeLimiterFunction),
}

impl GradientLimiter {
    fn apply(
        &self,
        gradients: &mut [Option<Gradients<Primitive>>],
        particles: &[Particle],
        part_is_active: &[bool],
        faces: &[VoronoiFace],
        cell_face_connections: &[usize],
        boundary: Boundary,
    ) {
        match self {
            GradientLimiter::None => {} // Nothing to do here
            GradientLimiter::Gradients => {
                particles
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
                                let extrapolated =
                                    gradients.dot(face.centroid() - centroid - shift);
                                let other_primitives = match get_other(face, part_idx) {
                                    Some(other_idx) => particles[other_idx].primitives,
                                    None => get_boundary_part(boundary, part, face).primitives,
                                };
                                limiter.collect(&other_primitives, &extrapolated);
                            }

                            limiter.limit(gradients, &part.primitives);
                            debug_assert!(gradients.is_finite());
                        }
                    });
            }
            GradientLimiter::Slopes(limiter_function) => {
                let flow_r: Vec<_> = particles
                    .par_iter()
                    .enumerate()
                    .zip(gradients.par_iter())
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
                                    Some(other_idx) => particles[other_idx].primitives,
                                    None => get_boundary_part(boundary, part, face).primitives,
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
                                    ratios[i] =
                                        DVec3::new(slope_prev / slope_next, slope_prev, slope_next);
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
                    .par_iter()
                    .enumerate()
                    .zip(gradients.par_iter_mut())
                    .for_each(|((i, limiter_info), grad)| {
                        if let Some(grad) = grad {
                            let limiter_info =
                                limiter_info.expect("cannot be none for Some gradients");
                            let dx = particles[i].volume;
                            let dx_inv = 1. / dx;
                            for j in 0..5 {
                                let (r, slope_left, slope_right) =
                                    (limiter_info[j].x, limiter_info[j].y, limiter_info[j].z);
                                let slope = grad[j].dot(dx * DVec3::X);
                                grad[j] = limiter_function.limit(r, slope, slope_left, slope_right)
                                    * dx_inv
                                    * DVec3::X;
                            }
                        }
                    })
            }
        }
    }
}

pub enum SlopeLimiterFunction {
    Direct { beta: f64 },
    VanLeer,
    SuperBee,
    MinBee,
}

impl SlopeLimiterFunction {
    fn xi_l(r: f64) -> f64 {
        2. / (1. + r)
    }
    fn xi_r(r: f64) -> f64 {
        2. * r / (1. + r)
    }

    fn minbee(r: f64) -> f64 {
        if r < 0. {
            0.
        } else {
            r.min(1.).min(Self::xi_l(r)).min(Self::xi_r(r))
        }
    }

    fn superbee(r: f64) -> f64 {
        if r < 0. {
            0.
        } else if r < 1. {
            1f64.min(2. * r)
        } else {
            r.min(1.).min(Self::xi_l(r)).min(Self::xi_r(r))
        }
    }

    fn van_leer(r: f64) -> f64 {
        if r < 0. {
            0.
        } else {
            (2. * r / (1. + r)).min(Self::xi_l(r)).min(Self::xi_r(r))
        }
    }

    fn direct(beta: f64, slope_left: f64, slope_right: f64) -> f64 {
        if slope_right > 0. {
            0f64.max(slope_right.min(beta * slope_left))
                .max(slope_left.min(beta * slope_right))
        } else {
            0f64.min(slope_right.max(beta * slope_left))
                .min(slope_left.max(beta * slope_right))
        }
    }

    fn limit(&self, r: f64, slope: f64, slope_left: f64, slope_right: f64) -> DVec3 {
        let limited_slope = match self {
            SlopeLimiterFunction::Direct { beta } => Self::direct(*beta, slope_left, slope_right),
            SlopeLimiterFunction::VanLeer => Self::van_leer(r) * slope,
            SlopeLimiterFunction::SuperBee => Self::superbee(r) * slope,
            SlopeLimiterFunction::MinBee => Self::minbee(r) * slope,
        };
        limited_slope * DVec3::X
    }
}

pub struct FluxInfo {
    pub fluxes: State<Conserved>,
    pub mflux: DVec3,
    pub v_max: f64,
    pub a_over_r: f64,
}
impl FluxInfo {
    pub fn zero() -> Self {
        Self {
            fluxes: State::vacuum(),
            mflux: DVec3::ZERO,
            v_max: 0.,
            a_over_r: 0.,
        }
    }
}

#[derive(Default, Debug, Copy, Clone)]
pub struct FluxLimiterData {
    pub jumps: DVec3,
    pub weight: f64,
}

impl FluxLimiterData {
    pub fn zero() -> Self {
        Self {
            jumps: DVec3::ZERO,
            weight: 0.,
        }
    }

    pub fn init(jumps: DVec3, r: f64) -> Self {
        let w = f64::exp(-r);
        Self {
            jumps: w * jumps,
            weight: w,
        }
    }

    pub fn collect(&mut self, jumps: DVec3, r: f64) {
        let w = f64::exp(-r);
        self.jumps += w * jumps;
        self.weight += w;
    }

    pub fn combine(&mut self, other: FluxLimiterData) {
        self.jumps += other.jumps;
        self.weight += other.weight;
    }

    pub fn apply(&self, jumps: DVec3, r: f64) -> DVec3 {
        let w = f64::exp(-r);
        let jumps = self.jumps - w * jumps;
        let w = self.weight - w;
        if w > 0. {
            jumps / w
        } else {
            DVec3::ZERO
        }
    }
}

impl Neg for FluxLimiterData {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            jumps: -self.jumps,
            weight: self.weight,
        }
    }
}

pub enum FluxLimiterFunction {
    None,
    MinBee,
    VanLeer,
    MC,
    SuperBee,
}

impl FluxLimiterFunction {
    pub fn limit(&self, r: f64) -> f64 {
        match self {
            FluxLimiterFunction::None => 1.,
            FluxLimiterFunction::MinBee => r.min(1.).max(0.),
            FluxLimiterFunction::VanLeer => ((r + r.abs()) / (1. + r.abs())).max(0.),
            FluxLimiterFunction::MC => ((1. + r) / 2.).min(2. * r).min(2.).max(0.),
            FluxLimiterFunction::SuperBee => r.min(2.).max((2. * r).min(1.)).max(0.),
        }
    }
}
