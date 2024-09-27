use meshless_voronoi::VoronoiFace;

use crate::{flux::FluxInfo, part::Particle, Boundary, Dimensionality, ParticleMotion};

mod euler_eqns_fvs;

pub use euler_eqns_fvs::EulerEqnsFvs;

pub trait FiniteVolumeSolver {
    fn predict(&self, particles: &mut [Particle], dt: f64);

    fn compute_fluxes(&self, faces: &[VoronoiFace], particles: &[Particle], part_is_active: &[bool], boundary: Boundary) -> Vec<FluxInfo>;

    fn convert_conserved_to_primitive(&self, particles: &mut [Particle], part_is_active: &[bool]);

    fn compute_timesteps(&self, particles: &mut [Particle], part_is_active: &[bool], particle_motion: ParticleMotion, dimesionality: Dimensionality) -> Vec<f64>;
}