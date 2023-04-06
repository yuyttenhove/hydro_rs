use glam::DVec3;
use rayon::prelude::*;
use yaml_rust::Yaml;

use crate::{errors::ConfigError, part::Particle};

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
    pub fn init(cfg: &Yaml) -> Result<Option<Self>, ConfigError> {
        let kind = cfg["kind"].as_str().unwrap_or("none");
        match kind {
            "none" => Ok(None),
            "external" => Ok(Some(GravitySolver::External(Potential::init(
                &cfg["potential"],
            )?))),
            "self-gravity" => {
                let softening_length = cfg["softening-length"].as_f64().unwrap_or(0.);
                Ok(Some(GravitySolver::SelfGravity { softening_length }))
            }
            _ => Err(ConfigError::UnknownGravity(kind.to_string())),
        }
    }

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

macro_rules! parse_dvec3 {
    ($acceleration:expr, $key:expr) => {
        match $acceleration {
            Yaml::Array(v) => Ok(DVec3 {
                x: v[0].as_f64().unwrap(),
                y: v[1].as_f64().unwrap(),
                z: v[2].as_f64().unwrap(),
            }),
            Yaml::BadValue => Err(ConfigError::MissingParameter(format!($key))),
            _ => Err(ConfigError::InvalidArrayFormat($acceleration.clone())),
        }
    };
}

impl Potential {
    fn accelerations(&self, particles: &[Particle]) -> Vec<DVec3> {
        match self {
            Self::Constant { acceleration } => vec![*acceleration; particles.len()],
            Self::Keplerian(potential) => potential.accelerations(particles),
        }
    }

    fn init(cfg: &Yaml) -> Result<Self, ConfigError> {
        let kind = cfg["kind"].as_str().ok_or(ConfigError::MissingParameter(
            "gravity:potential:kind".to_string(),
        ))?;

        match kind {
            "constant" => Ok(Self::Constant {
                acceleration: parse_dvec3!(&cfg["acceleration"], "gravity:potential:acceleration")?,
            }),
            "keplerian" => Ok(Self::Keplerian(KeplerianPotential::new(
                parse_dvec3!(&cfg["position"], "gravity:potential:position")?,
                cfg["softening-length"].as_f64().unwrap_or(0.),
            ))),
            _ => Err(ConfigError::UnknownGravity(format!(
                "gravity:external:kind:{:}",
                kind
            ))),
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
}

impl KeplerianPotential {
    fn new(position: DVec3, softening_length: f64) -> Self {
        Self {
            position,
            softening_length,
        }
    }

    fn accelerations(&self, particles: &[Particle]) -> Vec<DVec3> {
        particles
            .iter()
            .map(|part| {
                let r = part.loc - self.position;
                -r * (r.length_squared() + self.softening_length * self.softening_length).powf(-1.5)
            })
            .collect()
    }

    fn get_timestep(&self, particle: &Particle) -> f64 {
        let dx = particle.loc - self.position;
        let eta = 0.1 * (dx.length_squared() + self.softening_length.powi(2)).sqrt();
        (eta / particle.a_grav.length()).sqrt()
    }
}
