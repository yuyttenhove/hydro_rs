use glam::DVec3;

use crate::physical_quantities::{Conserved, Primitive};

use super::*;

/// HLLC Riemann solver
pub struct HLLCRiemannSolver;

impl RiemannFluxSolver for HLLCRiemannSolver {
    /// See Section 10.4, 10.5 and 10.6 in Toro (2009)
    fn solve_for_flux(
        &self,
        left: &State<Primitive>,
        right: &State<Primitive>,
        interface_velocity: DVec3,
        n_unit: DVec3,
        eos: &GasLaw,
    ) -> State<Conserved> {
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
        if VacuumRiemannSolver::is_vacuum(left, right, a_l, a_r, v_r_m_v_l, eos.gamma()) {
            let w_half =
                VacuumRiemannSolver.sample(left, right, v_l, v_r, a_l, a_r, n_unit, eos.gamma());
            return flux_from_half_state(&w_half, interface_velocity, n_unit, eos.gamma());
        }

        // STEP 1: Pressure estimate
        let ppvrs = 0.5 * (left.pressure() + right.pressure())
            - 0.125 * v_r_m_v_l * (left.density() + right.density()) * (a_l + a_r);
        let p_star = ppvrs.max(0.);

        // STEP 2: wave speed estimates
        let mut q_l = 1.;
        if p_star > left.pressure() && left.pressure() > 0. {
            q_l = (1. + 0.5 * eos.gamma().gp1dg() * (p_star / left.pressure() - 1.)).sqrt();
        }
        let mut q_r = 1.;
        if p_star > right.pressure() && right.pressure() > 0. {
            q_r = (1. + 0.5 * eos.gamma().gp1dg() * (p_star / right.pressure() - 1.)).sqrt();
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
            let e_l =
                eos.gas_internal_energy_from_pressure(left.pressure(), rho_l_inv) + 0.5 * v_l2;
            flux = Self::flux(left, v_l, e_l, n_unit);
            let s_l = s_l_m_v_l + v_l;
            if s_l < 0. {
                flux += Self::flux_star(left, v_l, e_l, s_l, s_star, n_unit);
            }
        } else {
            // flux FR
            let v_r2 = right.velocity().length_squared();
            let e_r =
                eos.gas_internal_energy_from_pressure(right.pressure(), rho_r_inv) + 0.5 * v_r2;
            flux = Self::flux(right, v_r, e_r, n_unit);
            let s_r = s_r_m_v_r + v_r;
            if s_r > 0. {
                flux += Self::flux_star(right, v_r, e_r, s_r, s_star, n_unit);
            }
        }
        debug_assert!(!(flux.mass().is_nan() || flux.mass().is_infinite()));

        // Deboost to lab frame
        flux += State::<Conserved>::new(
            0.,
            interface_velocity * flux.mass(),
            interface_velocity.dot(flux.momentum())
                + 0.5 * interface_velocity.length_squared() * flux.mass(),
        );
        flux
    }
}

impl HLLCRiemannSolver {
    /// See (10.5) in Toro.
    fn flux(state: &State<Primitive>, v: f64, e: f64, n_unit: DVec3) -> State<Conserved> {
        let rho_v = state.density() * v;
        State::<Conserved>::new(
            rho_v,
            rho_v * state.velocity() + state.pressure() * n_unit,
            ((state.density() * e) + state.pressure()) * v,
        )
    }

    /// See (10.38), (10.39) in Toro 2009.
    fn flux_star(
        state: &State<Primitive>,
        v: f64,
        e: f64,
        s: f64,
        s_star: f64,
        n_unit: DVec3,
    ) -> State<Conserved> {
        let s_m_v = s - v;
        let rho = state.density();
        let starfac = rho * s_m_v / (s - s_star);
        let v_star = (s_star - v) * n_unit + state.velocity();
        let e_star = e + (s_star - v) * (s_star + state.pressure() / (rho * s_m_v));
        let u_star = starfac * State::<Conserved>::new(1., v_star, e_star);
        let u_l = State::<Conserved>::new(rho, rho * state.velocity(), rho * e);
        s * (u_star - u_l)
    }
}

#[cfg(test)]
mod tests {
    use crate::gas_law::EquationOfState;

    use super::*;
    use float_cmp::assert_approx_eq;

    const GAMMA: f64 = 5. / 3.;

    fn get_eos() -> GasLaw {
        GasLaw::new(GAMMA, EquationOfState::Ideal)
    }

    #[test]
    fn test_hllc_solver_symmetry() {
        let interface_velocity = -3e-1 * DVec3::X;
        let eos = get_eos();
        let left = State::<Primitive>::new(1., DVec3::ZERO, 1.);
        let left_reversed = State::<Primitive>::new(1., DVec3::ZERO, 1.);
        let right = State::<Primitive>::new(1., -6e-1 * DVec3::X, 1.);
        let right_reversed = State::<Primitive>::new(1., 6e-1 * DVec3::X, 1.);

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
        let interface_velocity = 0.15 * DVec3::X;
        let eos = get_eos();
        let left = State::<Primitive>::new(1., 0.2 * DVec3::X, 0.5);
        let right = State::<Primitive>::new(0.5, 0.1 * DVec3::X, 0.1);
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
