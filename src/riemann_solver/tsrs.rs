use crate::{physical_quantities::{Primitive, State}, EquationOfState};

use super::{ExactRiemannSolver, PVRiemannSolver, RiemannStarSolver};


/// Two-shock Riemann solver.
/// 
/// This Riemann solver computes an estimate for the pressure in the star region 
/// by assuming both the left and right waves are shock waves.
/// 
/// The relations of the exact Riemann solver are then used to compute the other 
/// quantities in the star region.
pub struct TSRiemannSolver;

impl TSRiemannSolver {
    fn g(p: f64, rho_state: f64, p_state: f64,  eos: &EquationOfState) -> f64 {
        let a = eos.tdgp1() / rho_state;
        let b = eos.gm1dgp1() * p_state;
        f64::sqrt(a / (p + b))
    }
}

impl RiemannStarSolver for TSRiemannSolver {
    fn solve_for_star_state(
        &self,
        left: &State<Primitive>,
        right: &State<Primitive>,
        v_l: f64,
        v_r: f64,
        a_l: f64,
        a_r: f64,
        eos: &EquationOfState,
    ) -> super::RiemannStarValues {

        let p_guess = PVRiemannSolver::p_star(PVRiemannSolver::rho_bar(left.density(), right.density()), PVRiemannSolver::p_bar(left.pressure(), right.pressure()), PVRiemannSolver::a_bar(a_l, a_r), v_l, v_r);

        let g_l = Self::g(p_guess, left.density(), left.pressure(), eos);
        let g_r = Self::g(p_guess, right.density(), right.pressure(), eos);
        let pstar = (g_l * left.pressure() + g_r * right.pressure() - (v_r - v_l)) / (g_l + g_r);

        // Use relations of exact riemann solver once we have an estimate for pstar
        ExactRiemannSolver::star_state_from_pstar(pstar, left, right, v_l, v_r, a_l, a_r, eos)
    }
}
