use glam::DVec3;

use crate::{equation_of_state::EquationOfState, physical_quantities::Primitives};

use super::{RiemannStarSolver, RiemannStarValues};

/// Exact Vacuum Riemann solver.
pub struct VacuumRiemannSolver;

impl VacuumRiemannSolver {
    pub fn is_vacuum(
        left: &Primitives,
        right: &Primitives,
        a_l: f64,
        a_r: f64,
        v_r_m_v_l: f64,
        eos: &EquationOfState,
    ) -> bool {
        /* vacuum */
        if left.density() == 0. || right.density() == 0. {
            true
        }
        /* vacuum generation */
        else if eos.tdgm1() * (a_l + a_r) <= v_r_m_v_l {
            true
        }
        /* no vacuum */
        else {
            false
        }
    }

    fn sample_vacuum() -> Primitives {
        Primitives::vacuum()
    }

    fn sample_right_vacuum(
        left: &Primitives,
        v_l: f64,
        a_l: f64,
        s_l: f64,
        n_unit: DVec3,
        eos: &EquationOfState,
    ) -> Primitives {
        if 0. <= v_l - a_l {
            *left
        } else if 0. < s_l {
            // v_l - a_l < 0. < s_l
            Self::sample_rarefaction_fan(left, a_l, v_l, n_unit, eos)
        } else {
            // s_l <= 0.
            Self::sample_vacuum()
        }
    }

    fn sample_left_vacuum(
        right: &Primitives,
        v_r: f64,
        a_r: f64,
        s_r: f64,
        n_unit: DVec3,
        eos: &EquationOfState,
    ) -> Primitives {
        if 0. <= s_r {
            Self::sample_vacuum()
        } else if 0. < v_r + a_r {
            // s_r < 0. < v_r + a_r
            Self::sample_rarefaction_fan(right, -a_r, v_r, n_unit, eos)
        } else {
            // v_r + a_r <= 0.
            *right
        }
    }

    fn sample_vacuum_creation(
        left: &Primitives,
        right: &Primitives,
        v_l: f64,
        v_r: f64,
        a_l: f64,
        a_r: f64,
        n_unit: DVec3,
        eos: &EquationOfState,
    ) -> Primitives {
        let s_l = v_l + eos.tdgm1() * a_l;
        let s_r = v_r - eos.tdgm1() * a_r;
        if 0. <= s_l {
            Self::sample_right_vacuum(left, v_l, a_l, s_l, n_unit, eos)
        } else if 0. < s_r {
            // s_l < 0. < s_r
            Self::sample_vacuum()
        } else {
            // s_r <= 0.
            Self::sample_left_vacuum(right, v_r, a_r, s_r, n_unit, eos)
        }
    }
}

impl RiemannStarSolver for VacuumRiemannSolver {
    fn solve_for_star_state(
        &self,
        _left: &Primitives,
        _right: &Primitives,
        _v_l: f64,
        _v_r: f64,
        _a_l: f64,
        _a_r: f64,
        _eos: &EquationOfState,
    ) -> RiemannStarValues {
        panic!("The vacuum riemann solver should sample the half state immediately!");
    }

    fn sample(
        &self,
        left: &Primitives,
        right: &Primitives,
        v_l: f64,
        v_r: f64,
        a_l: f64,
        a_r: f64,
        n_unit: DVec3,
        eos: &EquationOfState,
    ) -> Primitives {
        debug_assert!(Self::is_vacuum(left, right, a_l, a_r, v_r - v_l, eos));

        if right.density() == 0. {
            let s_l = v_l + eos.tdgm1() * a_l;
            Self::sample_right_vacuum(left, v_l, a_l, s_l, n_unit, eos)
        } else if left.density() == 0. {
            let s_r = v_r - eos.tdgm1() * a_r;
            Self::sample_left_vacuum(right, v_r, a_r, s_r, n_unit, eos)
        } else {
            Self::sample_vacuum_creation(left, right, v_l, v_r, a_l, a_r, n_unit, eos)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::physical_quantities::Conserved;

    use super::super::RiemannFluxSolver;
    use super::*;
    use float_cmp::assert_approx_eq;
    use yaml_rust::YamlLoader;

    const GAMMA: f64 = 5. / 3.;

    #[test]
    fn test_vacuum_solver_symmetry() {
        let interface_velocity = 0.15 * DVec3::X;
        let eos = EquationOfState::new(
            &YamlLoader::load_from_str(&format!("gamma: {:}", GAMMA)).unwrap()[0],
        )
        .unwrap();
        let vel_l = DVec3 {
            x: 0.3,
            y: 0.,
            z: 0.,
        };
        let left = Primitives::new(1., vel_l, 0.5);
        let left_reversed = Primitives::new(1., -vel_l, 0.5);
        let right = Primitives::new(0., DVec3::ZERO, 0.);
        let fluxes = VacuumRiemannSolver.solve_for_flux(
            &left.boost(-interface_velocity),
            &right,
            interface_velocity,
            DVec3::X,
            &eos,
        );
        let fluxes_reversed = VacuumRiemannSolver.solve_for_flux(
            &right,
            &left_reversed.boost(interface_velocity),
            -interface_velocity,
            DVec3::X,
            &eos,
        );

        assert_approx_eq!(f64, fluxes.mass(), -fluxes_reversed.mass());
        assert_approx_eq!(f64, fluxes.momentum().x, fluxes_reversed.momentum().x);
        assert_approx_eq!(f64, fluxes.momentum().y, fluxes_reversed.momentum().y);
        assert_approx_eq!(f64, fluxes.momentum().z, fluxes_reversed.momentum().z);
        assert_approx_eq!(f64, fluxes.energy(), -fluxes_reversed.energy());
    }

    #[test]
    fn test_vacuum_solver_half_state() {
        let interface_velocity = 0.15 * DVec3::X;
        let eos = EquationOfState::new(
            &YamlLoader::load_from_str(&format!("gamma: {:}", GAMMA)).unwrap()[0],
        )
        .unwrap();
        let rho_l = 1.;
        let v_l = 0.3;
        let p_l = 0.5;
        let a_l = eos.sound_speed(p_l, 1. / rho_l);
        let rho_r = 0.;
        let v_r = 0.;
        let p_r = 0.;
        let a_r = 0.;
        let left = Primitives::new(rho_l, v_l * DVec3::X, p_l);
        let right = Primitives::new(rho_r, v_r * DVec3::X, p_r);
        let w_half = VacuumRiemannSolver.sample(
            &left.boost(-interface_velocity),
            &right,
            v_l - interface_velocity.x,
            v_r,
            a_l,
            a_r,
            DVec3::X,
            &eos,
        );

        // Reference solution
        let half_s = Primitives::new(
            0.4950623323204316,
            0.7221531968814576 * DVec3::X,
            0.15490656018956153,
        );

        assert_approx_eq!(f64, w_half.density(), half_s.density());
        assert_approx_eq!(f64, w_half.velocity().x, half_s.velocity().x);
        assert_approx_eq!(f64, w_half.pressure(), half_s.pressure());
    }

    #[test]
    fn test_vacuum_solver_invariance() {
        let eos = EquationOfState::new(
            &YamlLoader::load_from_str(&format!("gamma: {:}", GAMMA)).unwrap()[0],
        )
        .unwrap();

        let left = Primitives::new(1.5, -0.2 * DVec3::X, 1.2);
        let right = Primitives::vacuum();

        let interface_velocity = 0.3 * DVec3::X;

        let w_half_lab = VacuumRiemannSolver.sample(
            &left,
            &right,
            left.velocity().x,
            0.,
            eos.sound_speed(left.pressure(), 1. / left.density()),
            0.,
            DVec3::X,
            &eos,
        );
        let roe = w_half_lab.pressure() * eos.odgm1()
            + 0.5 * w_half_lab.density() * w_half_lab.velocity().length_squared();
        let fluxes_lab =
            VacuumRiemannSolver.solve_for_flux(&left, &right, DVec3::ZERO, DVec3::X, &eos)
                - Conserved::new(
                    w_half_lab.density() * interface_velocity.x,
                    w_half_lab.density() * w_half_lab.velocity() * interface_velocity,
                    roe * interface_velocity.x,
                );
        let fluxes_face = VacuumRiemannSolver.solve_for_flux(
            &left.boost(-interface_velocity),
            &right,
            interface_velocity,
            DVec3::X,
            &eos,
        );
        assert_approx_eq!(f64, fluxes_lab.mass(), fluxes_face.mass());
        assert_approx_eq!(f64, fluxes_lab.momentum().x, fluxes_face.momentum().x);
        assert_approx_eq!(f64, fluxes_lab.energy(), fluxes_face.energy());
    }
}
