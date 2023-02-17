use glam::DVec3;

use crate::{
    equation_of_state::EquationOfState,
    physical_quantities::{Conserved, Primitives},
};

use super::*;

/// HLLC Riemann solver
pub struct HLLCRiemannSolver;

impl RiemannFluxSolver for HLLCRiemannSolver {
    /// See Section 10.4, 10.5 and 10.6 in Toro (2009)
    fn solve_for_flux(
        &self,
        left: &Primitives,
        right: &Primitives,
        interface_velocity: DVec3,
        n_unit: DVec3,
        eos: &EquationOfState,
    ) -> Conserved {
        // Inverse densities
        let rho_l_inv = 1. / left.density();
        let rho_r_inv = 1. / right.density();
        let v_l = left.velocity().dot(n_unit);
        let v_r = right.velocity().dot(n_unit);
        let a_l = eos.sound_speed(left.pressure(), rho_l_inv);
        let a_r = eos.sound_speed(right.pressure(), rho_r_inv);

        // velocity difference
        let v_r_m_v_l = v_r - v_l;

        // handle vacuum
        if VacuumRiemannSolver::is_vacuum(left, right, a_l, a_r, v_r_m_v_l, eos) {
            let w_half = VacuumRiemannSolver.sample(left, right, v_l, v_r, a_l, a_r, n_unit, eos);
            return flux_from_half_state(&w_half, interface_velocity, n_unit, eos);
        }

        // STEP 1: Pressure estimate
        let ppvrs = 0.5 * (left.pressure() + right.pressure())
            - 0.125 * v_r_m_v_l * (left.density() + right.density()) * (a_l + a_r);
        let p_star = ppvrs.max(0.);

        // STEP 2: wave speed estimates
        let mut q_l = 1.;
        if p_star > left.pressure() && left.pressure() > 0. {
            q_l = (1. + 0.5 * eos.gp1dg() * (p_star / left.pressure() - 1.)).sqrt();
        }
        let mut q_r = 1.;
        if p_star > right.pressure() && right.pressure() > 0. {
            q_r = (1. + 0.5 * eos.gp1dg() * (p_star / right.pressure() - 1.)).sqrt();
        }

        let s_l_m_v_l = -a_l * q_l;
        let s_r_m_v_r = a_r * q_r;
        let s_star = (right.pressure() - left.pressure() + left.density() * v_l * s_l_m_v_l
            - right.density() * v_r * s_r_m_v_r)
            / (left.density() * s_l_m_v_l - right.density() * s_r_m_v_r);

        // STEP 3: HLLC flux in a frame moving with the interface velocity
        let mut flux;
        if s_star >= 0. {
            // flux FL
            let v_l2 = left.velocity().length_squared();
            let rho_l_v_l = left.density() * v_l;
            let e_l = eos.odgm1() * left.pressure() * rho_l_inv + 0.5 * v_l2;
            let rho_l_e_l = left.density() * e_l;
            let s_l = s_l_m_v_l + v_l;
            flux = Conserved::new(
                rho_l_v_l,
                rho_l_v_l * left.velocity() + left.pressure() * n_unit,
                (rho_l_e_l + left.pressure()) * v_l,
            );
            if s_l < 0. {
                // flux FL*
                let starfac = left.density() * s_l_m_v_l / (s_l - s_star);
                let e_star = e_l
                    + (s_star - v_l) * (s_star + left.pressure() / (left.density() * s_l_m_v_l));
                let u_star =
                    starfac * Conserved::new(1., (s_star - v_l) * n_unit + left.velocity(), e_star);
                let u_l = Conserved::new(left.density(), left.velocity(), rho_l_e_l);
                flux += s_l * (u_star - u_l);
            }
        } else {
            // flux FR
            let v_r2 = right.velocity().length_squared();
            let rho_r_v_r = right.density() * v_r;
            let e_r = eos.odgm1() * right.pressure() * rho_r_inv + 0.5 * v_r2;
            let rho_r_e_r = right.density() * e_r;
            let s_r = s_r_m_v_r + v_r;
            flux = Conserved::new(
                rho_r_v_r,
                rho_r_v_r * right.velocity() + right.pressure() * n_unit,
                (rho_r_e_r + right.pressure()) * v_r,
            );
            if s_r > 0. {
                // flux FR*
                let starfac = right.density() * s_r_m_v_r / (s_r - s_star);
                let e_star = e_r
                    + (s_star - v_r) * (s_star + right.pressure() / (right.density() * s_r_m_v_r));
                let u_star = starfac
                    * Conserved::new(1., (s_star - v_r) * n_unit + right.velocity(), e_star);
                let u_r = Conserved::new(right.density(), right.velocity(), rho_r_e_r);
                flux += s_r * (u_star - u_r);
            }
        }
        debug_assert!(!(flux.mass().is_nan() || flux.mass().is_infinite()));

        // Deboost to lab frame
        flux += Conserved::new(
            0.,
            interface_velocity * flux.mass(),
            interface_velocity.dot(flux.momentum())
                + 0.5 * interface_velocity.length_squared() * flux.mass(),
        );
        flux
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;
    use yaml_rust::YamlLoader;

    const GAMMA: f64 = 5. / 3.;

    fn get_eos() -> EquationOfState {
        EquationOfState::new(&YamlLoader::load_from_str(&format!("gamma: {:}", GAMMA)).unwrap()[0])
            .unwrap()
    }

    #[test]
    fn test_hllc_solver_symmetry() {
        let interface_velocity = -3e-1 * DVec3::X;
        let eos = get_eos();
        let left = Primitives::new(1., DVec3::ZERO, 1.);
        let left_reversed = Primitives::new(1., DVec3::ZERO, 1.);
        let right = Primitives::new(1., -6e-1 * DVec3::X, 1.);
        let right_reversed = Primitives::new(1., 6e-1 * DVec3::X, 1.);

        let fluxes = HLLCRiemannSolver.solve_for_flux(
            &left.boost(-interface_velocity),
            &right.boost(-interface_velocity),
            interface_velocity,
            DVec3::X,
            &eos,
        );
        let fluxes_reversed = HLLCRiemannSolver.solve_for_flux(
            &right_reversed.boost(interface_velocity),
            &left_reversed.boost(interface_velocity),
            -interface_velocity,
            DVec3::X,
            &eos,
        );

        assert_approx_eq!(f64, fluxes.mass(), -fluxes_reversed.mass());
        assert_approx_eq!(f64, fluxes.momentum().x, fluxes_reversed.momentum().x);
        assert_approx_eq!(f64, fluxes.energy(), -fluxes_reversed.energy());
    }

    #[test]
    fn test_hllc_solver() {
        let interface_velocity = -0.2 * DVec3::X;
        let eos = get_eos();
        let left = Primitives::new(1., 0.2 * DVec3::X, 0.5);
        let right = Primitives::new(0.5, 0.1 * DVec3::X, 0.1);
        let fluxes = HLLCRiemannSolver.solve_for_flux(
            &left.boost(-interface_velocity),
            &right.boost(-interface_velocity),
            interface_velocity,
            DVec3::X,
            &eos,
        );

        // reference solution
        let flux_s = ExactRiemannSolver.solve_for_flux(
            &left.boost(-interface_velocity),
            &right.boost(-interface_velocity),
            interface_velocity,
            DVec3::X,
            &eos,
        );

        let tol = 5e-2;
        assert_approx_eq!(f64, fluxes.mass(), flux_s.mass(), epsilon = tol);
        assert_approx_eq!(f64, fluxes.momentum().x, flux_s.momentum().x, epsilon = tol);
        assert_approx_eq!(f64, fluxes.energy(), flux_s.energy(), epsilon = tol);
    }
}
