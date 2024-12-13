use glam::DVec3;

use super::{RiemannStarSolver, RiemannStarValues, RiemannWafFluxSolver};
use crate::finite_volume_solver::FluxLimiter;
use crate::gas_law::AdiabaticIndex;
use crate::{
    gas_law::GasLaw,
    physical_quantities::{Conserved, Primitive, State},
};

pub struct LinearAdvectionRiemannSover {
    velocity: DVec3,
}

impl LinearAdvectionRiemannSover {
    pub fn new(velocity: DVec3) -> Self {
        Self { velocity }
    }
}

impl RiemannStarSolver for LinearAdvectionRiemannSover {
    fn solve_for_star_state(
        &self,
        left: &State<Primitive>,
        right: &State<Primitive>,
        v_l: f64,
        v_r: f64,
        a_l: f64,
        a_r: f64,
        gamma: &AdiabaticIndex,
    ) -> RiemannStarValues {
        RiemannStarValues::default()
    }
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
        left_flux_limiter: &FluxLimiter,
        right_flux_limiter: &FluxLimiter,
        r: f64,
        do_limit: bool,
        interface_velocity: DVec3,
        dt: f64,
        n_unit: DVec3,
        eos: &GasLaw,
    ) -> State<Conserved> {
        // Boost to interface frame
        let left = left.boost(-interface_velocity);
        let right = right.boost(-interface_velocity);
        let star_states = self.solve_for_star_state(&left, &right, 0., 0., 0., 0., eos.gamma());
        let jumps_local = DVec3::new(
            star_states.rho_l - left.density(),
            star_states.rho_r - star_states.rho_l,
            right.density() - star_states.rho_r,
        );
        let jumps_left = left_flux_limiter.apply(jumps_local, r);
        let jumps_right = right_flux_limiter.apply(jumps_local, r);
        let drho = right.density() - left.density();
        let drho_left = jumps_left.element_sum();
        let drho_right = jumps_right.element_sum();

        let dx_left = dx_left.dot(n_unit);
        let dx_right = dx_right.dot(n_unit);
        let dx = dx_left + dx_right;
        assert!(dx_left >= 0.);
        assert!(dx_right >= 0.);
        let v = (self.velocity - interface_velocity).dot(n_unit);
        let c = v * dt;
        assert!(dx_left >= 0.5 * c);
        assert!(dx_right >= 0.5 * c);
        let (phi_left, phi_right) = if do_limit {
            let c_left = 0.5 * c / dx_left;
            let c_right = 0.5 * c / dx_right;
            let r = if v > 0. {
                drho_right / drho
            } else {
                drho_left / drho
            };
            let psi_r = f64::max(0., f64::max(f64::min(1., 2. * r), f64::min(2., r)));
            (
                dx_left * (1. - (1. - c_left.abs()) * psi_r),
                dx_right * (1. - (1. - c_right.abs()) * psi_r),
            )
        } else {
            (0.5 * c, 0.5 * c)
        };

        let flux_new = 1. / dx
            * (v * (dx_left + phi_left) * State::<Conserved>::new(left.density(), DVec3::ZERO, 0.)
                + v * (dx_right - phi_right)
                    * State::<Conserved>::new(right.density(), DVec3::ZERO, 0.));

        let flux = 1. / (dx_left + dx_right)
            * (v * (dx_left + 0.5 * v * dt)
                * State::<Conserved>::new(left.density(), DVec3::ZERO, 0.)
                + v * (dx_right - 0.5 * v * dt)
                    * State::<Conserved>::new(right.density(), DVec3::ZERO, 0.));

        if dt > 0. && drho > 1e-5 {
            println!("{:?}", flux);
        }
        flux_new
    }
}
