use glam::DVec3;
use meshless_voronoi::VoronoiFace;

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

pub use godunov_fvs::GodunovFvs;
pub use muscl_fvs::MusclFvs;
pub use waf_fvs::WafFvs;

pub trait FiniteVolumeSolver: Sync {
    fn predict(&self, particles: &mut [Particle], dt: f64) {
        particles.par_iter_mut().for_each(|part| {
            part.extrapolate_state(dt, self.eos());
        });
    }

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
