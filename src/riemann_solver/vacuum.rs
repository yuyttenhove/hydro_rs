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

    fn sample_half_vacuum(
        non_vacuum: &Primitives,
        v: f64,
        a: f64,
        n_unit: DVec3,
        eos: &EquationOfState,
    ) -> Primitives {
        let base = eos.tdgp1() + eos.gm1dgp1() * v / a;
        let v_half = eos.tdgp1() * (a + v / eos.tdgm1()) - v;
        Primitives::new(
            non_vacuum.density() * base.powf(eos.tdgm1()),
            non_vacuum.velocity() + n_unit * v_half,
            non_vacuum.pressure() * base.powf(eos.gamma() * eos.tdgm1()),
        )
    }

    fn sample_right_vacuum(
        left: &Primitives,
        v_l: f64,
        a_l: f64,
        n_unit: DVec3,
        eos: &EquationOfState,
    ) -> Primitives {
        if v_l < a_l {
            let s_l = v_l + eos.tdgm1() * a_l;
            if s_l > 0. {
                Self::sample_half_vacuum(left, v_l, a_l, n_unit, eos)
            } else {
                Primitives::vacuum()
            }
        } else {
            *left
        }
    }

    fn sample_left_vacuum(
        right: &Primitives,
        v_r: f64,
        a_r: f64,
        n_unit: DVec3,
        eos: &EquationOfState,
    ) -> Primitives {
        if v_r > -a_r {
            let s_r = v_r - eos.tdgm1() * a_r;
            if s_r < 0. {
                Self::sample_half_vacuum(right, v_r, -a_r, n_unit, eos)
            } else {
                Primitives::vacuum()
            }
        } else {
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
        let s_l = v_l - eos.tdgm1() * a_l;
        let s_r = v_r - eos.tdgm1() * a_r;

        if s_l >= 0. {
            if a_l > v_l {
                Self::sample_half_vacuum(left, v_l, a_l, n_unit, eos)
            } else {
                *left
            }
        } else if s_r <= 0. {
            if -a_r < v_r {
                Self::sample_half_vacuum(right, v_r, -a_r, n_unit, eos)
            } else {
                *right
            }
        } else {
            debug_assert!(s_r > 0. && s_l < 0.);
            Primitives::vacuum()
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
            Self::sample_right_vacuum(left, v_l, a_l, n_unit, eos)
        } else if left.density() == 0. {
            Self::sample_left_vacuum(right, v_r, a_r, n_unit, eos)
        } else {
            Self::sample_vacuum_creation(left, right, v_l, v_r, a_l, a_r, n_unit, eos)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::riemann_solver::flux_from_half_state;

    use super::super::RiemannFluxSolver;
    use super::*;
    use float_cmp::assert_approx_eq;
    use yaml_rust::YamlLoader;

    #[test]
    fn test_vacuum_solver_symmetry() {
        let interface_velocity = 0.15 * DVec3::X;
        let eos = EquationOfState::new(&YamlLoader::load_from_str("gamma: 1.666667").unwrap()[0])
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
    fn test_vacuum_solver() {
        let interface_velocity = 0.15 * DVec3::X;
        let eos = EquationOfState::new(&YamlLoader::load_from_str("gamma: 1.666667").unwrap()[0])
            .unwrap();
        let left = Primitives::new(1., 0.3 * DVec3::X, 0.5);
        let right = Primitives::new(0., DVec3::ZERO, 0.);
        let fluxes = VacuumRiemannSolver.solve_for_flux(
            &left.boost(-interface_velocity),
            &right,
            interface_velocity,
            DVec3::X,
            &eos,
        );

        // Reference solution calculated in lab frame
        let half_s = Primitives::new(
            0.57625934,
            0.7596532 * DVec3::X - interface_velocity,
            0.19952622,
        );
        let flux_s = flux_from_half_state(&half_s, interface_velocity, DVec3::X, &eos);

        assert_approx_eq!(f64, fluxes.mass(), flux_s.mass());
        assert_approx_eq!(f64, fluxes.momentum().x, flux_s.momentum().x);
        assert_approx_eq!(f64, fluxes.momentum().y, flux_s.momentum().y);
        assert_approx_eq!(f64, fluxes.momentum().z, flux_s.momentum().z);
        assert_approx_eq!(f64, fluxes.energy(), flux_s.energy());
    }
}
