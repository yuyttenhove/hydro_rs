use crate::gas_law::AdiabaticIndex;

use super::{ExactRiemannSolver, RiemannStarSolver};

/// Two-rarefaction Riemann solver.
///
/// This Riemann solver computes an estimate for the pressure in the star region
/// by assuming both the left and right waves are rarefaction waves.
///
/// The relations of the exact Riemann solver are then used to compute the other
/// quantities in the star region.
pub struct TRRiemannSolver;

impl RiemannStarSolver for TRRiemannSolver {
    fn solve_for_star_state(
        &self,
        left: &crate::physical_quantities::State<crate::physical_quantities::Primitive>,
        right: &crate::physical_quantities::State<crate::physical_quantities::Primitive>,
        v_l: f64,
        v_r: f64,
        a_l: f64,
        a_r: f64,
        eos: &AdiabaticIndex,
    ) -> super::RiemannStarValues {
        let beta = eos.gm1d2g();
        let gm1d2 = 0.5 * (eos.gamma() - 1.);
        let num = a_l + a_r - gm1d2 * (v_r - v_l);
        let denom = a_l * left.pressure().powf(-beta) + a_r * right.pressure().powf(-beta);
        let pstar = (num / denom).powf(1. / beta);
        ExactRiemannSolver::star_state_from_pstar(pstar, left, right, v_l, v_r, a_l, a_r, eos)
    }
}
