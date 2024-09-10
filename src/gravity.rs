use glam::DVec3;
use rayon::prelude::*;

use crate::{part::Particle, Dimensionality};

fn compute_self_gravity(particles: &[Particle], softening_length: f64) -> Vec<DVec3> {
    let num_particles = particles.len();

    (0..num_particles)
        .into_par_iter()
        .map(|i| {
            let pi = &particles[i];
            let mut acceleration = DVec3::ZERO;
            for j in (i + 1)..num_particles {
                let pj = &particles[j];
                let r = (pj.loc - pi.loc).length() + softening_length;
                let dir = pj.loc - pi.loc;
                let a = pj.conserved.mass() * dir / (r * r * r);
                acceleration += a;
            }
            acceleration
        })
        .collect()
}

pub enum GravitySolver {
    External(Potential),
    SelfGravity { softening_length: f64 },
}

impl GravitySolver {
    pub fn accelerations(&self, particles: &[Particle]) -> Vec<DVec3> {
        match self {
            Self::External(potential) => potential.accelerations(particles),
            Self::SelfGravity { softening_length } => {
                compute_self_gravity(particles, *softening_length)
            }
        }
    }

    pub fn get_timestep(&self, particle: &Particle) -> f64 {
        match self {
            Self::External(potential) => potential.get_timestep(particle),
            Self::SelfGravity { softening_length } => {
                let a2 = particle.a_grav.length_squared();
                if a2 == 0. {
                    return std::f64::INFINITY;
                }
                (softening_length / a2.sqrt()).sqrt()
            }
        }
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
            Potential::Constant { acceleration: _ } => std::f64::INFINITY,
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
