//! Meshless-Voronoi Moving Mesh hydrodynamics code/library focussed on flexibility rather than raw performance.
//!
//! The library also provides access to a variety of Riemann Solvers for the Euler equations.

pub use engine::{Engine, EngineTrait, TimestepInfo, ParticleMotion};
pub use gravity::{GravitySolver, KeplerianPotential, Potential};
pub use initial_conditions::InitialConditions;
pub use space::{Boundary, Space};
pub use time_integration::Runner;

/// The dimensionality of the hydrodynamics simulation.
pub type Dimensionality = meshless_voronoi::Dimensionality;

mod cell;
mod engine;
mod errors;
mod flux;
pub mod gas_law;
mod gradients;
mod gravity;
mod initial_conditions;
mod kernels;
mod macros;
mod part;
mod physical_constants;
pub mod physical_quantities;
pub mod riemann_solver;
mod space;
mod time_integration;
mod timeline;
mod utils;
