pub use crate::physical_quantities::{Primitives, Conserved};

#[derive(Default, Debug, Clone)]
pub struct Part {
    pub primitives: Primitives,
    pub gradients: Primitives,
    pub conserved: Conserved,
    pub fluxes: Conserved,
    pub gravity_mflux: f64,

    pub volume: f64,
    pub x: f64,
    pub dt: f64,

    pub a_grav: f64,
}