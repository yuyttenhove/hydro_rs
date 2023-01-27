use std::{
    error::Error,
    fmt::{Debug, Display},
};

#[derive(Debug)]
pub enum ConfigError {
    MissingParameter(String),
    UnknownRunner(String),
    UnknownICs(String),
    UnknownRiemannSolver(String),
    UnknownParticleMotion(String),
    UnknownBoundaryConditions(String),
}

impl Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::MissingParameter(name) => {
                write!(f, "Missing required parameter in configuration: {}", name)
            }
            ConfigError::UnknownRunner(name) => {
                write!(f, "Unknown type of runner configured: {}", name)
            }
            ConfigError::UnknownRiemannSolver(name) => {
                write!(f, "Unknown type of runner configured: {name}")
            }
            ConfigError::UnknownICs(name) => {
                write!(f, "Unknown type of initial conditions configured: {}", name)
            }
            ConfigError::UnknownParticleMotion(name) => {
                write!(f, "Unknown type of particle motion configured: {}", name)
            }
            ConfigError::UnknownBoundaryConditions(name) => {
                write!(f, "Unknown type of boundary condition configured: {}", name)
            }
        }
    }
}

impl Error for ConfigError {}
