use std::{error::Error, fmt::{Display, Debug}};

#[derive(Debug)]
pub enum ConfigError<'a> {
    MissingParameter(&'a str)
}

impl<'a> Display for ConfigError<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::MissingParameter(name) 
                => write!(f, "Missing required parameter in configuration: {}", name),
        }
    }
}

impl<'a> Error for ConfigError<'a>{}