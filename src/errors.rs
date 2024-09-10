use std::{
    error::Error,
    fmt::{Debug, Display},
};

#[derive(Debug)]
pub enum MVMMError {
    UnknownRiemannSolver(String),
    MissingAIRSThreshold,
}

impl Display for MVMMError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MVMMError::UnknownRiemannSolver(name) => {
                write!(f, "Unknown Riemann solver: {:}", name)
            }
            MVMMError::MissingAIRSThreshold => {
                write!(f, "Missing AIRS threshold!")
            }
        }
    }
}

impl Error for MVMMError {}
