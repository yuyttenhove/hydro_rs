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

#[cfg(test)]
mod test {
    use float_cmp::assert_approx_eq;
    use glam::DVec3;
    use yaml_rust::YamlLoader;

    use crate::{equation_of_state::EquationOfState, physical_quantities::{Primitives, Conserved}, riemann_solver::RiemannFluxSolver};

    use super::*;

    const GAMMA: f64 = 5. / 3.;

    #[test]
    fn test_invariance() {
        let eos = EquationOfState::new(
            &YamlLoader::load_from_str(&format!("gamma: {:}", GAMMA)).unwrap()[0],
        )
        .unwrap();

        let left = Primitives::new(1.5, 0.2 * DVec3::X, 1.2);
        let right = Primitives::new(0.7, -0.4 * DVec3::X, 0.1);

        let interface_velocity = -0.17 * DVec3::X;

        let w_half_lab = PVRiemannSolver.sample(
            &left,
            &right,
            left.velocity().x,
            right.velocity().x,
            eos.sound_speed(left.pressure(), 1. / left.density()),
            eos.sound_speed(right.pressure(), 1. / right.density()),
            DVec3::X,
            &eos,
        );
        let roe = w_half_lab.pressure() * eos.odgm1()
            + 0.5 * w_half_lab.density() * w_half_lab.velocity().length_squared();
        let fluxes_lab =
            PVRiemannSolver.solve_for_flux(&left, &right, DVec3::ZERO, DVec3::X, &eos)
                - Conserved::new(
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
