use crate::{flux::{flux_exchange, flux_exchange_boundary, FluxInfo}, gas_law::GasLaw, part::Particle, riemann_solver::RiemannFluxSolver, Boundary, Dimensionality, ParticleMotion};

use super::FiniteVolumeSolver;

use rayon::prelude::*;

pub struct EulerEqnsFvs<R: RiemannFluxSolver> {
    riemann_solver: R,
    cfl: f64,
    gas_law: GasLaw,
}

impl<R: RiemannFluxSolver> EulerEqnsFvs<R> {
    pub fn new(riemann_solver: R, cfl: f64, gas_law: GasLaw) -> Self {
        Self { riemann_solver, cfl, gas_law }
    }
}

impl<R: RiemannFluxSolver> FiniteVolumeSolver for EulerEqnsFvs<R> {
    fn predict(&self, particles: &mut [Particle], dt: f64) {
        particles.par_iter_mut().for_each(|part| {
            part.extrapolate_state(dt, &self.gas_law);
        });
    }

    fn compute_fluxes(&self, faces: &[meshless_voronoi::VoronoiFace], particles: &[Particle], boundary: Boundary) -> Vec<FluxInfo> {
        faces.par_iter().map(|face|{
            let left = &particles[face.left()];
            match face.right() {
                Some(right_idx) => {
                    let right = &particles[right_idx];
                    let dt = left.dt.min(right.dt);
                    flux_exchange(left, right, dt, face, 0.5, &self.gas_law, &self.riemann_solver)
                }
                None => flux_exchange_boundary(left, face, boundary, 0.5, &self.gas_law, &self.riemann_solver)
            }
        }).collect()
    }

    fn compute_timesteps(&self, particles: &mut [Particle], part_is_active: &[bool], particle_motion: ParticleMotion, dimensionality: Dimensionality) -> Vec<f64> {
        particles.par_iter_mut().enumerate().map(|(part_idx, part)| {
            if !part_is_active[part_idx] { return std::f64::INFINITY; }
            // Compute new hydro timestep
            part.timestep(
                self.cfl,
                particle_motion,
                &self.gas_law,
                dimensionality,
            )
        }).collect()
    }
    
    fn convert_conserved_to_primitive(&self, particles: &mut [Particle], part_is_active: &[bool]) {
        particles.par_iter_mut().enumerate().for_each(|(part_idx, part)| {
            if part_is_active[part_idx] {
                part.convert_conserved_to_primitive(&self.gas_law);
            }
        });
    }
}