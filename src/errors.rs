use std::{
    error::Error,
    fmt::{Debug, Display},
};

#[derive(Debug)]
pub enum ConfigError<'a> {
    MissingParameter(&'a str),
    UnknownRunner(String),
    UnknownICs(String),
    UnknownRiemannSolver(String),
}

impl<'a> Display for ConfigError<'a> {
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
        }
    }
}

impl<'a> Error for ConfigError<'a> {}
