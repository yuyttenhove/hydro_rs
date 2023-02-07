use yaml_rust::Yaml;

use crate::errors::ConfigError;

#[derive(Clone, Copy)]
pub enum EquationOfState {
    Ideal { gamma: f64 },
    Isothermal,
}

impl EquationOfState {
    pub fn new(cfg: &Yaml) -> Result<Self, ConfigError> {
        let gamma = cfg["gamma"].as_f64().unwrap_or(5. / 3.);
        Ok(EquationOfState::Ideal { gamma })
    }
    pub fn gas_internal_energy_from_pressure(&self, pressure: f64, density: f64) -> f64 {
        match self {
            EquationOfState::Ideal { gamma } => pressure / density / (gamma - 1.),
            _ => todo!(),
        }
    }

    pub fn gas_pressure_from_internal_energy(&self, internal_energy: f64, density: f64) -> f64 {
        match self {
            EquationOfState::Ideal { gamma } => (gamma - 1.) * internal_energy * density,
            _ => todo!(),
        }
    }

    pub fn sound_speed(&self, pressure: f64, density_inv: f64) -> f64 {
        match self {
            EquationOfState::Ideal { gamma } => (gamma * pressure * density_inv).sqrt(),
            _ => unimplemented!(),
        }
    }

    pub fn gas_entropy_from_internal_energy(&self, internal_energy: f64, density: f64) -> f64 {
        match self {
            EquationOfState::Ideal { gamma } => {
                (gamma - 1.) * internal_energy * density.powf(1. - gamma)
            }
            _ => unimplemented!(),
        }
    }

    pub fn gamma(&self) -> f64 {
        match self {
            EquationOfState::Ideal { gamma } => *gamma,
            EquationOfState::Isothermal => panic!("Trying to query gamma of isothermal gas!"),
        }
    }
}
