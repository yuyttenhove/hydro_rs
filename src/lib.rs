//! Meshless-Voronoi Moving Mesh hydrodynamics code/library focussed on flexibility rather than raw performance.
//!
//! The library also provides access to a variety of Riemann Solvers for the Euler equations.

pub use engine::{Engine, ParticleMotion, TimestepInfo};
pub use initial_conditions::InitialConditions;
pub use runner::{hydrodynamics, Runner};
pub use space::{Boundary, Space};

/// The dimensionality of the hydrodynamics simulation.
pub type Dimensionality = meshless_voronoi::Dimensionality;

mod cell;
mod engine;
mod errors;
pub mod finite_volume_solver;
pub mod gas_law;
mod gradients;
pub mod gravity;
mod initial_conditions;
mod kernels;
mod macros;
mod part;
mod physical_constants;
pub mod physical_quantities;
pub mod riemann_solver;
mod runner;
mod space;
mod time_integration;
mod timeline;
mod utils;
