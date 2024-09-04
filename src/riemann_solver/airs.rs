use crate::{
    gas_law::AdiabaticIndex,
    physical_quantities::{Primitive, State},
};

use super::{ExactRiemannSolver, PVRiemannSolver, RiemannStarSolver};

pub struct AIRiemannSolver {
    threshold: f64,
}

impl AIRiemannSolver {
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }
}

impl RiemannStarSolver for AIRiemannSolver {
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
        let star = PVRiemannSolver.solve_for_star_state(left, right, v_l, v_r, a_l, a_r, eos);
        let p_max = left.pressure().max(right.pressure());
        let p_min = left.pressure().min(right.pressure());
        if p_max / p_min < self.threshold {
            star
        } else {
            ExactRiemannSolver.solve_for_star_state(left, right, v_l, v_r, a_l, a_r, eos)
        }
    }
}
