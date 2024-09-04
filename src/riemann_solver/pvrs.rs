use crate::physical_quantities::{Primitive, State};

use super::{RiemannStarSolver, RiemannStarValues};

pub struct PVRiemannSolver;

impl PVRiemannSolver {
    pub(super) fn rho_bar(rho_l: f64, rho_r: f64) -> f64 {
        0.5 * (rho_l + rho_r)
    }

    pub(super) fn p_bar(p_l: f64, p_r: f64) -> f64 {
        0.5 * (p_l + p_r)
    }

    pub(super) fn a_bar(a_l: f64, a_r: f64) -> f64 {
        0.5 * (a_l + a_r)
    }

    pub(super) fn p_star(rho_bar: f64, p_bar: f64, a_bar: f64, v_l: f64, v_r: f64) -> f64 {
        p_bar + 0.5 * (v_l - v_r) * rho_bar * a_bar
    }
}

impl RiemannStarSolver for PVRiemannSolver {
    fn solve_for_star_state(
        &self,
        left: &State<Primitive>,
        right: &State<Primitive>,
        v_l: f64,
        v_r: f64,
        a_l: f64,
        a_r: f64,
        _eos: &crate::gas_law::AdiabaticIndex,
    ) -> RiemannStarValues {
        let rho_bar = Self::rho_bar(left.density(), right.density());
        let p_bar = Self::p_bar(left.pressure(), right.pressure());
        let a_bar = Self::a_bar(a_l, a_r);

        let p = Self::p_star(rho_bar, p_bar, a_bar, v_l, v_r);
        let u = 0.5 * ((v_l + v_r) + (left.pressure() - right.pressure()) / (rho_bar * a_bar));
        let rho_l = left.density() + (v_l - u) * rho_bar / a_bar;
        let rho_r = right.density() + (u - v_r) * rho_bar / a_bar;

        RiemannStarValues { rho_l, rho_r, u, p }
    }
}

#[cfg(test)]
mod test {
    use float_cmp::assert_approx_eq;
    use glam::DVec3;

    use crate::{
        gas_law::{EquationOfState, GasLaw},
        physical_quantities::{Conserved, Primitive},
        riemann_solver::RiemannFluxSolver,
    };

    use super::*;

    const GAMMA: f64 = 5. / 3.;

    #[test]
    fn test_invariance() {
        let eos = GasLaw::new(GAMMA, EquationOfState::Ideal);

        let left = State::<Primitive>::new(1.5, 0.2 * DVec3::X, 1.2);
        let right = State::<Primitive>::new(0.7, -0.4 * DVec3::X, 0.1);

        let interface_velocity = -0.17 * DVec3::X;

        let w_half_lab = PVRiemannSolver.sample(
            &left,
            &right,
            left.velocity().x,
            right.velocity().x,
            eos.sound_speed(left.pressure(), 1. / left.density()),
            eos.sound_speed(right.pressure(), 1. / right.density()),
            DVec3::X,
            eos.gamma(),
        );
        let roe = w_half_lab.pressure() * eos.gamma().odgm1()
            + 0.5 * w_half_lab.density() * w_half_lab.velocity().length_squared();
        let fluxes_lab = PVRiemannSolver.solve_for_flux(&left, &right, DVec3::ZERO, DVec3::X, &eos)
            - State::<Conserved>::new(
                w_half_lab.density() * interface_velocity.x,
                w_half_lab.density() * w_half_lab.velocity() * interface_velocity,
                roe * interface_velocity.x,
            );
        let fluxes_face = PVRiemannSolver.solve_for_flux(
            &left.boost(-interface_velocity),
            &right.boost(-interface_velocity),
            interface_velocity,
            DVec3::X,
            &eos,
        );
        assert_approx_eq!(f64, fluxes_lab.mass(), fluxes_face.mass());
        assert_approx_eq!(f64, fluxes_lab.momentum().x, fluxes_face.momentum().x);
        assert_approx_eq!(f64, fluxes_lab.energy(), fluxes_face.energy());
    }
}
