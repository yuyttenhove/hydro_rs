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

use crate::physical_quantities::Primitive;
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

    fn do_flux_limit(&self) -> bool {
        false
    }

    fn flux_limit_faces(
        &self,
        faces: &[VoronoiFace],
        particles: &[Particle],
        part_is_active: &[bool],
        boundary: Boundary,
    ) -> Vec<FluxLimiterData> {
        unimplemented!("Shouldn't call this function!")
    }

    fn flux_limiter_collect(
        &self,
        left: &State<Primitive>,
        right: &State<Primitive>,
        ds: DVec3,
        normal: DVec3,
        limiter_data: &mut FluxLimiterData,
    ) {
        unimplemented!("Shouldn't call this function!")
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
    MinBee,
    VanLeer,
    MC,
    SuperBee,
}

impl FluxLimiterFunction {
    pub fn limit(&self, r: f64) -> f64 {
        match self {
            FluxLimiterFunction::MinBee => r.min(1.).max(0.),
            FluxLimiterFunction::VanLeer => ((r + r.abs()) / (1. + r.abs())).max(0.),
            FluxLimiterFunction::MC => ((1. + r) / 2.).min(2. * r).min(2.).max(0.),
            FluxLimiterFunction::SuperBee => r.min(2.).max((2. * r).min(1.)).max(0.),
        }
    }
}
