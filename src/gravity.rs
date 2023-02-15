use glam::DVec3;
use rayon::prelude::*;
use yaml_rust::Yaml;

use crate::{errors::ConfigError, part::Part};

fn compute_self_gravity(particles: &[Part], softening_length: f64) -> Vec<DVec3> {
    let num_particles = particles.len();

    (0..num_particles)
        .into_par_iter()
        .map(|i| {
            let pi = &particles[i];
            let mut acceleration = DVec3::ZERO;
            for j in (i + 1)..num_particles {
                let pj = &particles[j];
                let r = (pj.x - pi.x).length() + softening_length;
                let dir = pj.x - pi.x;
                let a = pj.conserved.mass() * dir / (r * r * r);
                acceleration += a;
            }
            acceleration
        })
        .collect()
}

pub enum GravitySolver {
    ConstantAcceleration(DVec3),
    SelfGravity { softening_length: f64 },
}

impl GravitySolver {
    pub fn init(cfg: &Yaml) -> Result<Option<Self>, ConfigError> {
        let kind = cfg["kind"].as_str().unwrap_or("none");
        match kind {
            "none" => Ok(None),
            "constant" => {
                let acceleration = &cfg["acceleration"];
                let acceleration = match acceleration {
                    Yaml::Array(v) => Ok(DVec3 {
                        x: v[0].as_f64().unwrap(),
                        y: v[1].as_f64().unwrap(),
                        z: v[2].as_f64().unwrap(),
                    }),
                    Yaml::BadValue => Err(ConfigError::MissingParameter(format!(
                        "gravity: acceleration"
                    ))),
                    _ => Err(ConfigError::InvalidArrayFormat(acceleration.clone())),
                }?;
                Ok(Some(GravitySolver::ConstantAcceleration(acceleration)))
            }
            "self-gravity" => {
                let softening_length = cfg["softening-length"].as_f64().unwrap_or(0.);
                Ok(Some(GravitySolver::SelfGravity { softening_length }))
            }
            _ => Err(ConfigError::UnknownGravity(kind.to_string())),
        }
    }

    pub fn accelerations(&self, particles: &[Part]) -> Vec<DVec3> {
        match self {
            GravitySolver::ConstantAcceleration(a) => vec![*a; particles.len()],
            Self::SelfGravity { softening_length } => {
                compute_self_gravity(particles, *softening_length)
            }
        }
    }
}
