use yaml_rust::Yaml;

use crate::errors::ConfigError;

#[derive(Clone, Copy)]
pub enum EquationOfState {
    Ideal {
        gamma: f64,
        // (gamma + 1) / (gamma)
        gp1dg: f64,
        // (gamma - 1) / (2 gamma)
        gm1d2g: f64,
        // (gamma - 1) / (gamma + 1)
        gm1dgp1: f64,
        // 1 / (gamma - 1)
        odgm1: f64,
        // 2 / (gamma - 1)
        tdgm1: f64,
        // 2 / (gamma + 1)
        tdgp1: f64,
    },
    Isothermal,
}

impl EquationOfState {
    pub fn new(cfg: &Yaml) -> Result<Self, ConfigError> {
        let gamma = cfg["gamma"].as_f64().unwrap_or(5. / 3.);
        Ok(EquationOfState::Ideal {
            gamma,
            gp1dg: (gamma + 1.) / gamma,
            gm1d2g: 0.5 * (gamma - 1.) / gamma,
            gm1dgp1: (gamma - 1.) / (gamma + 1.),
            odgm1: 1. / (gamma - 1.),
            tdgm1: 2. / (gamma - 1.),
            tdgp1: 2. / (gamma + 1.),
        })
    }
    pub fn gas_internal_energy_from_pressure(&self, pressure: f64, density: f64) -> f64 {
        match self {
            EquationOfState::Ideal { gamma, .. } => pressure / density / (gamma - 1.),
            _ => todo!(),
        }
    }

    pub fn gas_pressure_from_internal_energy(&self, internal_energy: f64, density: f64) -> f64 {
        match self {
            EquationOfState::Ideal { gamma, .. } => (gamma - 1.) * internal_energy * density,
            _ => todo!(),
        }
    }

    pub fn sound_speed(&self, pressure: f64, density_inv: f64) -> f64 {
        match self {
            EquationOfState::Ideal { gamma, .. } => (gamma * pressure * density_inv).sqrt(),
            _ => unimplemented!(),
        }
    }

    pub fn gas_entropy_from_internal_energy(&self, internal_energy: f64, density: f64) -> f64 {
        match self {
            EquationOfState::Ideal { gamma, .. } => {
                (gamma - 1.) * internal_energy * density.powf(1. - gamma)
            }
            _ => unimplemented!(),
        }
    }

    pub fn gamma(&self) -> f64 {
        match self {
            EquationOfState::Ideal { gamma, .. } => *gamma,
            _ => panic!("Trying to query gamma of isothermal gas!"),
        }
    }

    pub fn gp1dg(&self) -> f64 {
        match self {
            EquationOfState::Ideal { gp1dg, .. } => *gp1dg,
            _ => panic!("Trying to query gamma of isothermal gas!"),
        }
    }

    pub fn gm1d2g(&self) -> f64 {
        match self {
            EquationOfState::Ideal { gm1d2g, .. } => *gm1d2g,
            _ => panic!("Trying to query gamma of isothermal gas!"),
        }
    }

    pub fn gm1dgp1(&self) -> f64 {
        match self {
            EquationOfState::Ideal { gm1dgp1, .. } => *gm1dgp1,
            _ => panic!("Trying to query gamma of isothermal gas!"),
        }
    }

    pub fn odgm1(&self) -> f64 {
        match self {
            EquationOfState::Ideal { odgm1, .. } => *odgm1,
            _ => panic!("Trying to query gamma of isothermal gas!"),
        }
    }

    pub fn tdgm1(&self) -> f64 {
        match self {
            EquationOfState::Ideal { tdgm1, .. } => *tdgm1,
            _ => panic!("Trying to query gamma of isothermal gas!"),
        }
    }

    pub fn tdgp1(&self) -> f64 {
        match self {
            EquationOfState::Ideal { tdgp1, .. } => *tdgp1,
            _ => panic!("Trying to query gamma of isothermal gas!"),
        }
    }
}
