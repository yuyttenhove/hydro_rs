use super::{RiemannStarSolver, RiemannStarValues};

pub struct PVRiemannSolver;

impl RiemannStarSolver for PVRiemannSolver {
    fn solve_for_star_state(
        &self,
        left: &crate::physical_quantities::Primitives,
        right: &crate::physical_quantities::Primitives,
        v_l: f64,
        v_r: f64,
        a_l: f64,
        a_r: f64,
        _eos: &crate::equation_of_state::EquationOfState,
    ) -> RiemannStarValues {
        let rho_bar = 0.5 * (left.density() + right.density());
        let a_bar = 0.5 * (a_l + a_r);

        let p = 0.5 * ((left.pressure() + right.pressure()) + (v_l - v_r) * rho_bar * a_bar);
        let u = 0.5 * ((v_l + v_r) + (left.pressure() - right.pressure()) / (rho_bar * a_bar));
        let rho_l = left.density() + (v_l - u) * rho_bar / a_bar;
        let rho_r = right.density() + (u - v_r) * rho_bar / a_bar;

        RiemannStarValues { rho_l, rho_r, u, p }
    }
}
