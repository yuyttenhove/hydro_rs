use glam::DVec3;
use rayon::prelude::*;

use crate::{part::Particle, Dimensionality};

pub trait GravitySolver {
    fn compute_accelerations(&self, particles: &mut [Particle]);

    fn compute_timesteps(&self, particles: &[Particle]) -> Vec<f64>;
}

pub struct SelfGravity {
    softening_length: f64,
}

impl SelfGravity {
    pub fn new(softening_length: f64) -> Self {
        Self { softening_length }
    }
}

impl GravitySolver for SelfGravity {
    fn compute_accelerations(&self, particles: &mut [Particle]) {
        let num_particles = particles.len();

        let accelerations: Vec<_> = (0..num_particles)
            .into_par_iter()
            .map(|i| {
                let pi = &particles[i];
                let mut acceleration = DVec3::ZERO;
                for j in (i + 1)..num_particles {
                    let pj = &particles[j];
                    let r = (pj.loc - pi.loc).length() + self.softening_length;
                    let dir = pj.loc - pi.loc;
                    let a = pj.conserved.mass() * dir / (r * r * r);
                    acceleration += a;
                }
                acceleration
            })
            .collect();

        particles.par_iter_mut().zip(accelerations).for_each(|(p, a)| p.a_grav = a);
    }

    fn compute_timesteps(&self, particles: &[Particle]) -> Vec<f64> {
        particles.par_iter().map(|particle|{
            let a2 = particle.a_grav.length_squared();
                if a2 == 0. {
                    return f64::INFINITY;
                }
                (self.softening_length / a2.sqrt()).sqrt()
        }).collect()
    }
}

pub struct ExternalPotentialGravity {
    potential: Potential,
}

impl ExternalPotentialGravity {
    pub fn new(potential: Potential) -> Self {
        Self { potential }
    }
}

impl GravitySolver for ExternalPotentialGravity {
    fn compute_accelerations(&self, particles: &mut [Particle]) {
        let accelerations = self.potential.accelerations(particles);
        particles.par_iter_mut().zip(accelerations).for_each(|(p, a)| p.a_grav = a);
    }

    fn compute_timesteps(&self, particles: &[Particle]) -> Vec<f64> {
        particles.par_iter().map(|particle| self.potential.get_timestep(particle)).collect()
    }
}

pub enum Potential {
    Constant { acceleration: DVec3 },
    Keplerian(KeplerianPotential),
}

impl Potential {
    fn accelerations(&self, particles: &[Particle]) -> Vec<DVec3> {
        match self {
            Self::Constant { acceleration } => vec![*acceleration; particles.len()],
            Self::Keplerian(potential) => potential.accelerations(particles),
        }
    }

    fn get_timestep(&self, particle: &Particle) -> f64 {
        match self {
            Potential::Constant { acceleration: _ } => f64::INFINITY,
            Potential::Keplerian(potential) => potential.get_timestep(particle),
        }
    }
}

pub struct KeplerianPotential {
    position: DVec3,
    softening_length: f64,
    eta: f64,
}

impl KeplerianPotential {
    pub fn new(position: DVec3, softening_length: f64) -> Self {
        Self {
            position,
            softening_length,
            eta: 0.5,
        }
    }

    fn accelerations(&self, particles: &[Particle]) -> Vec<DVec3> {
        particles
            .iter()
            .map(|part| self.acceleration(part.loc))
            .collect()
    }

    fn get_timestep(&self, particle: &Particle) -> f64 {
        // Make sure the particle does not travel its more than a fraction of its radius at the gravitational circular velocity. Only 2D simulations are supported
        self.eta * particle.radius(Dimensionality::TwoD) / self.circular_velocity(particle.loc)
    }

    fn acceleration(&self, position: DVec3) -> DVec3 {
        let dx = position - self.position;
        let r2 = dx.length_squared();
        let r = r2.sqrt();
        // soften if necessary
        let r2 = if r < self.softening_length {
            r2 + self.softening_length * self.softening_length
        } else {
            r2
        };
        -dx / (r * r2)
    }

    fn circular_velocity(&self, position: DVec3) -> f64 {
        let dx = position - self.position;
        let r = dx.length();
        let r_softened = if r < self.softening_length {
            r + self.softening_length
        } else {
            r
        };
        (1. / r_softened).sqrt()
    }
}
