use glam::DVec3;

use crate::physical_quantities::{Conserved, Primitive, State};

use super::RiemannFluxSolver;

pub struct LinearAdvectionRiemannSover {
    velocity: DVec3,
}

impl LinearAdvectionRiemannSover {
    pub fn new(velocity: DVec3) -> Self {
        Self { velocity }
    }
}

impl RiemannFluxSolver for LinearAdvectionRiemannSover {
    fn solve_for_flux(
        &self,
        left: &State<Primitive>,
        right: &State<Primitive>,
        interface_velocity: DVec3,
        n_unit: DVec3,
        _eos: &crate::gas_law::GasLaw,
    ) -> State<Conserved> {
        // Sample left or right state based on whether the advection velocity is to the left or right in the frame of the face
        let v = (self.velocity - interface_velocity).dot(n_unit);
        if v > 0. {
            v * State::<Conserved>::new(left.density(), left.velocity(), left.pressure())
        } else {
            v * State::<Conserved>::new(right.density(), right.velocity(), right.pressure())
        }
    }
}
