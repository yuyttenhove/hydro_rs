use crate::{
    gas_law::AdiabaticIndex,
    physical_quantities::{Primitive, State},
};

use super::{
    ExactRiemannSolver, PVRiemannSolver, RiemannStarSolver, TRRiemannSolver, TSRiemannSolver,
};

/// A non-iterative adaptive Riemann solver.
///
/// This solver is different from the non-iterative solver presented by Toro 2009.
/// It does not use the primitive value riemann solver, but instead switches between the TRRS and
/// TSRS depending on wether the TRRS is exact or not.
pub struct ANRiemannSolver {
    threshold: f64,
}

impl ANRiemannSolver {
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }
}

impl RiemannStarSolver for ANRiemannSolver {
    fn solve_for_star_state(
        &self,
        left: &State<Primitive>,
        right: &State<Primitive>,
        v_l: f64,
        v_r: f64,
        a_l: f64,
        a_r: f64,
        eos: &AdiabaticIndex,
    ) -> super::RiemannStarValues {
        /* The TRRS is exact when f(min(PL, PR)) >= 0 (Toro 2009, section 9.4.1) */
        let p_min = left.pressure().min(right.pressure());
        if ExactRiemannSolver::f(p_min, left, right, v_l, v_r, a_l, a_r, eos) >= 0. {
            return TRRiemannSolver.solve_for_star_state(left, right, v_l, v_r, a_l, a_r, eos);
        }
        /* If pmin and pmax are not too different, use the primitive value riemann solvers pressure
        estimate in the star region to determine whether the two rarefaction wave solution is
        accurate */
        let p_max = left.pressure().max(right.pressure());
        if p_max / p_min < 2.
            && PVRiemannSolver::p_star(
                PVRiemannSolver::rho_bar(left.density(), right.density()),
                PVRiemannSolver::p_bar(left.pressure(), right.pressure()),
                PVRiemannSolver::a_bar(a_l, a_r),
                v_l,
                v_r,
            ) < p_min
        {
            return TRRiemannSolver.solve_for_star_state(left, right, v_l, v_r, a_l, a_r, eos);
        }
        /* TRRS probably not a good approximation, use more robust TSRS */
        TSRiemannSolver.solve_for_star_state(left, right, v_l, v_r, a_l, a_r, eos)
    }
}
