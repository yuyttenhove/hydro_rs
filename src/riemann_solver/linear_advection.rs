use glam::DVec3;

use crate::{
    gas_law::GasLaw,
    physical_quantities::{Conserved, Primitive, State},
};

use super::{RiemannFluxSolver, RiemannWafFluxSolver};

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
        _eos: &GasLaw,
    ) -> State<Conserved> {
        // Boost to interface frame
        let left = left.boost(-interface_velocity);
        let right = right.boost(-interface_velocity);

        // Sample left or right state based on whether the advection velocity is to the left or right in the frame of the face
        let v = (self.velocity - interface_velocity).dot(n_unit);
        if v > 0. {
            v * State::<Conserved>::new(left.density(), DVec3::ZERO, 0.)
        } else {
            v * State::<Conserved>::new(right.density(), DVec3::ZERO, 0.)
        }
    }
}

impl RiemannWafFluxSolver for LinearAdvectionRiemannSover {
    fn solve_for_waf_flux(
        &self,
        left: &State<Primitive>,
        right: &State<Primitive>,
        dx_left: DVec3,
        dx_right: DVec3,
        drho_left: f64,
        drho_right: f64,
        interface_velocity: DVec3,
        dt: f64,
        n_unit: DVec3,
        _eos: &GasLaw,
    ) -> State<Conserved> {
        // Boost to interface frame
        let left = left.boost(-interface_velocity);
        let right = right.boost(-interface_velocity);
        let drho = right.density() - left.density();

        let dx_left = dx_left.dot(n_unit);
        let dx_right = dx_right.dot(n_unit);
        let dx = dx_left + dx_right;
        assert!(dx_left >= 0.);
        assert!(dx_right >= 0.);
        let v = (self.velocity - interface_velocity).dot(n_unit);
        let c = (0.5 * v * dt).abs();
        assert!(dx_left >= c);
        assert!(dx_right >= c);
        let (phi_left, phi_right) = if v > 0. {
            let r = drho_left / drho;
            let psi = (r + r.abs()) / (1. + r);
            (
                1. - (dx_left - c) / (2. * dx_left) * psi,
                1. - (dx_right - c) / (2. * dx_right) * psi,
            )
        } else {
            ((dx_left - c) / dx, (dx_right + c) / dx);
            let r = drho_right / drho;
            let psi = (r + r.abs()) / (1. + r);
            (
                1. - (dx_left - c) / dx * psi,
                1. - (dx_right - c) / dx * psi,
            )
        };
        // 0.5 * v * ((1. + phi_left) * State::<Conserved>::new(left.density(), DVec3::ZERO, 0.) + (1. - phi_right) * State::<Conserved>::new(right.density(), DVec3::ZERO, 0.))

        1. / (dx_left + dx_right)
            * (v * (dx_left + 0.5 * v * dt)
                * State::<Conserved>::new(left.density(), DVec3::ZERO, 0.)
                + v * (dx_right - 0.5 * v * dt)
                    * State::<Conserved>::new(right.density(), DVec3::ZERO, 0.))
    }
}
