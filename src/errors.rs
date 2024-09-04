use std::{
    error::Error,
    fmt::{Debug, Display},
};

use yaml_rust::Yaml;

#[derive(Debug)]
pub enum ConfigError {
    MissingParameter(String),
    UnknownRunner(String),
    UnknownICs(String),
    UnknownRiemannSolver(String),
    UnknownGravity(String),
    UnknownParticleMotion(String),
    UnknownBoundaryConditions(String),
    UnknownEOS(String),
    IllegalBoxSize(String),
    InvalidArrayFormat(Yaml),
    InvalidArrayLength(usize, usize),
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
            ConfigError::UnknownGravity(name) => {
                write!(f, "Unknown type of gravity configured: {name}")
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
            ConfigError::UnknownEOS(name) => {
                write!(f, "Unknown type of equation of stated configured: {}", name)
            }
            ConfigError::IllegalBoxSize(name) => {
                write!(f, "Illegal box_size format: {}!", name)
            }
            ConfigError::InvalidArrayFormat(value) => {
                write!(f, "Expected array but found: {:?}", value)
            }
            ConfigError::InvalidArrayLength(a, b) => {
                write!(f, "Expected array of lenght {}, but found {}", a, b)
            }
        }
    }
}

impl Error for ConfigError {}
