use crate::{
    equation_of_state::EquationOfState,
    errors::ConfigError,
    part::{Conserved, Primitives},
};

pub trait RiemannSolver {
    fn solve_for_flux(
        &self,
        left: &Primitives,
        right: &Primitives,
        interface_velocity: f64,
        eos: &EquationOfState,
    ) -> Conserved;
}

pub fn get_solver<'a>(kind: String, gamma: f64) -> Result<Box<dyn RiemannSolver>, ConfigError<'a>> {
    match kind.as_str() {
        "HLLC" => Ok(Box::new(HLLCRiemannSolver::new(gamma))),
        "exact" => todo!(),
        _ => Err(ConfigError::UnknownRiemannSolver(kind)),
    }
}

/// Exact Vacuum Riemann solver.
struct VacuumRiemannSolver {
    gamma: f64,
    // (gamma - 1) / (gamma + 1)
    gm1dgp1: f64,
    /* \f$\frac{1}{\gamma{}-1}\f$. */
    odgm1: f64,
    // 2 / (gamma - 1)
    tdgm1: f64,
    // 2 / (gamma + 1)
    tdgp1: f64,
}

impl VacuumRiemannSolver {
    fn new(gamma: f64) -> Self {
        Self {
            gamma,
            gm1dgp1: (gamma - 1.) / (gamma + 1.),
            odgm1: 1. / (gamma - 1.),
            tdgm1: 2. / (gamma - 1.),
            tdgp1: 2. / (gamma + 1.),
        }
    }

    fn is_vacuum(
        &self,
        left: &Primitives,
        right: &Primitives,
        a_l: f64,
        a_r: f64,
        v_r_m_v_l: f64,
    ) -> bool {
        /* vacuum */
        if left.density() == 0. || right.density() == 0. {
            true
        }
        /* vacuum generation */
        else if self.tdgm1 * (a_l + a_r) <= v_r_m_v_l {
            true
        }
        /* no vacuum */
        else {
            false
        }
    }

    fn sample_half_vacuum(&self, non_vacuum: &Primitives, a: f64) -> Primitives {
        let base = self.tdgp1 + self.gm1dgp1 * non_vacuum.velocity() / a;
        Primitives::new(
            non_vacuum.density() * base.powf(self.tdgm1),
            self.tdgp1 * (a + non_vacuum.velocity() / self.tdgm1),
            non_vacuum.pressure() * base.powf(self.gamma * self.tdgm1),
        )
    }

    fn sample_right_vacuum(&self, left: &Primitives, a_l: f64) -> Primitives {
        if left.velocity() < a_l {
            let s_l = left.velocity() + self.tdgm1 * a_l;
            if s_l > 0. {
                self.sample_half_vacuum(left, a_l)
            } else {
                Primitives::vacuum()
            }
        } else {
            *left
        }
    }

    fn sample_left_vacuum(&self, right: &Primitives, a_r: f64) -> Primitives {
        if -a_r < right.velocity() {
            let s_r = right.velocity() - self.tdgm1 * a_r;
            if s_r >= 0. {
                Primitives::vacuum()
            } else {
                self.sample_half_vacuum(right, -a_r)
            }
        } else {
            *right
        }
    }

    fn sample_vacuum_creation(
        &self,
        left: &Primitives,
        right: &Primitives,
        a_l: f64,
        a_r: f64,
    ) -> Primitives {
        let s_l = left.velocity() - self.tdgm1 * a_l;
        let s_r = right.velocity() - self.tdgm1 * a_r;

        if s_l >= 0. {
            if a_l > left.velocity() {
                self.sample_half_vacuum(left, a_l)
            } else {
                *left
            }
        } else if s_r <= 0. {
            if -a_r < right.velocity() {
                self.sample_half_vacuum(right, -a_r)
            } else {
                *right
            }
        } else {
            debug_assert!(s_r > 0. && s_l < 0.);
            Primitives::vacuum()
        }
    }

    fn solve(&self, left: &Primitives, right: &Primitives, a_l: f64, a_r: f64) -> Primitives {
        if right.density() == 0. {
            self.sample_right_vacuum(left, a_l)
        } else if left.density() == 0. {
            self.sample_left_vacuum(right, a_r)
        } else {
            self.sample_vacuum_creation(left, right, a_l, a_r)
        }
    }

    fn solve_for_flux(
        &self,
        left: &Primitives,
        right: &Primitives,
        a_l: f64,
        a_r: f64,
        interface_velocity: f64,
    ) -> Conserved {
        debug_assert!(self.is_vacuum(left, right, a_l, a_r, left.velocity() - right.velocity()));

        // Solve for total vacuum
        if left.density() == 0. && right.density() == 0. {
            return Conserved::vacuum();
        }

        // Get primitives at interface
        let half = self.solve(left, right, a_l, a_r);

        // Note: All velocities are in the interface frame.
        let vtot = half.velocity() + interface_velocity;
        let mass_flux = half.density() * half.velocity();
        let momentum_flux = half.density() * vtot * half.velocity() + half.pressure();
        // F_P = \rho e ( \vec{v} - \vec{v_{ij}} ) + P \vec{v}
        // \rho e = P / (\gamma-1) + 1/2 \rho^2 \vec{v}^2
        let v_2 = vtot * vtot;
        let energy_flux = (half.pressure() * self.odgm1 + 0.5 * half.density() * v_2)
            * half.velocity()
            + half.pressure() * vtot;

        Conserved::new(mass_flux, momentum_flux, energy_flux)
    }
}

/// HLLC Riemann solver
pub struct HLLCRiemannSolver {
    /* Adiabatic index \f$\gamma{}\f$. */
    gamma: f64,
    /* \f$\frac{2(\gamma{}+1)}{\gamma{}}\f$. */
    hgp1dg: f64,
    /* \f$\frac{1}{\gamma{}-1}\f$. */
    odgm1: f64,
    vacuum_solver: VacuumRiemannSolver,
}

impl HLLCRiemannSolver {
    pub fn new(gamma: f64) -> Self {
        Self {
            gamma,
            hgp1dg: (gamma + 1.) / gamma,
            odgm1: 1. / (gamma - 1.),
            vacuum_solver: VacuumRiemannSolver::new(gamma),
        }
    }
}

impl RiemannSolver for HLLCRiemannSolver {
    fn solve_for_flux(
        &self,
        left: &Primitives,
        right: &Primitives,
        interface_velocity: f64,
        eos: &EquationOfState,
    ) -> Conserved {
        // Consistency check
        if let EquationOfState::Ideal { gamma } = eos {
            assert_eq!(*gamma, self.gamma);
        }

        // Inverse densities
        let rho_l_inv = 1. / left.density();
        let rho_r_inv = 1. / right.density();
        let a_l = eos.sound_speed(left.pressure(), rho_l_inv);
        let a_r = eos.sound_speed(right.pressure(), rho_r_inv);

        // velocity difference
        let v_r_m_v_l = right.velocity() - left.velocity();

        // handle vacuum
        if self
            .vacuum_solver
            .is_vacuum(left, right, a_l, a_r, v_r_m_v_l)
        {
            return self
                .vacuum_solver
                .solve_for_flux(left, right, a_l, a_r, interface_velocity);
        }

        // STEP 1: Pressure estimate
        let ppvrs = 0.5 * (left.pressure() + right.pressure())
            - 0.25 * v_r_m_v_l * (left.density() + right.density()) * (a_l + a_r); // -0.125
        let p_star = ppvrs.max(0.);

        // STEP 2: wave speed estimates
        let mut q_l = 1.;
        if p_star > left.pressure() && left.pressure() > 0. {
            q_l = (1. + 0.5 * self.hgp1dg * (p_star / left.pressure() - 1.)).sqrt();
        }
        let mut q_r = 1.;
        if p_star > right.pressure() && right.pressure() > 0. {
            q_r = (1. + 0.5 * self.hgp1dg * (p_star / right.pressure() - 1.)).sqrt();
        }

        let s_l_m_v_l = -a_l * q_l;
        let s_r_m_v_r = a_r * q_r;
        let s_star = (right.pressure() - left.pressure()
            + left.density() * left.velocity() * s_l_m_v_l
            - right.density() * right.velocity() * s_r_m_v_r)
            / (left.density() * s_l_m_v_l - right.density() * s_r_m_v_r);

        // STEP 3: HLLC flux in a frame moving with the interface velocity
        let mut flux;
        if s_star >= 0. {
            // flux FL
            let v_l2 = left.velocity() * left.velocity();
            let rho_l_v_l = left.density() * left.velocity();
            let e_l = self.odgm1 * left.pressure() * rho_l_inv + 0.5 * v_l2;
            let rho_l_e_l = left.density() * e_l;
            let s_l = s_l_m_v_l + left.velocity();
            flux = Conserved::new(
                rho_l_v_l,
                left.density() * v_l2 + left.pressure(),
                (rho_l_e_l + left.pressure()) * left.velocity(),
            );
            if s_l < 0. {
                // flux FL*
                let starfac = s_l_m_v_l / (s_l - s_star) - 1.0;
                let rho_l_s_l = left.density() * s_l;
                let s_star_m_v_l = s_star - left.velocity();
                let rho_l_s_l_starfac = rho_l_s_l * starfac;
                let rho_l_s_l_s_star_m_v_l = rho_l_s_l * s_star_m_v_l;

                flux += Conserved::new(
                    rho_l_s_l_starfac,
                    rho_l_s_l_starfac * left.velocity() + rho_l_s_l_s_star_m_v_l,
                    rho_l_s_l_starfac * e_l
                        + rho_l_s_l_s_star_m_v_l
                            * (s_star + left.pressure() / (left.density() * s_l_m_v_l)),
                );
            }
        } else {
            // flux FR
            let v_r2 = right.velocity() * right.velocity();
            let rho_r_v_r = right.density() * right.velocity();
            let e_r = self.odgm1 * right.pressure() * rho_r_inv + 0.5 * v_r2;
            let rho_r_e_r = right.density() * e_r;
            let s_r = s_r_m_v_r + right.velocity();
            flux = Conserved::new(
                rho_r_v_r,
                right.density() * v_r2 + right.pressure(),
                (rho_r_e_r + right.pressure()) * right.velocity(),
            );
            if s_r > 0. {
                // flux FR*
                let starfac = s_r_m_v_r / (s_r - s_star) - 1.0;
                let rho_r_s_r = right.density() * s_r;
                let s_star_m_v_r = s_star - right.velocity();
                let rho_r_s_r_starfac = rho_r_s_r * starfac;
                let rho_r_s_r_s_star_m_v_r = rho_r_s_r * s_star_m_v_r;

                flux += Conserved::new(
                    rho_r_s_r_starfac,
                    rho_r_s_r_starfac * right.velocity() + rho_r_s_r_s_star_m_v_r,
                    rho_r_s_r_starfac * e_r
                        + rho_r_s_r_s_star_m_v_r
                            * (s_star + right.pressure() / (right.density() * s_r_m_v_r)),
                );
            }
        }
        debug_assert!(!(flux.mass().is_nan() || flux.mass().is_infinite()));

        // Deboost to lab frame
        flux += Conserved::new(
            0.,
            interface_velocity * flux.mass(),
            interface_velocity * flux.momentum()
                + 0.5 * interface_velocity * interface_velocity * flux.mass(),
        );
        flux
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    fn flux_from_riemann_solution(
        half: Primitives,
        interface_velocity: f64,
        gamma: f64,
    ) -> Conserved {
        Conserved::new(
            half.density() * (half.velocity() - interface_velocity),
            half.density() * half.velocity() * (half.velocity() - interface_velocity)
                + half.pressure(),
            (0.5 * half.density() * half.velocity() * half.velocity()
                + half.pressure() / (gamma - 1.))
                * (half.velocity() - interface_velocity)
                + half.pressure() * half.velocity(),
        )
    }

    #[test]
    fn test_vacuum_solver() {
        let gamma = 5. / 3.;
        let interface_velocity = 0.8;
        let solver = VacuumRiemannSolver::new(gamma);
        let eos = EquationOfState::Ideal { gamma };
        let left = Primitives::new(1., 0.3, 0.5);
        let right = Primitives::new(0., 0., 0.);
        let a_l = eos.sound_speed(left.pressure(), 1. / left.density());
        let a_r = eos.sound_speed(right.pressure(), 1. / right.density());
        let half = solver.solve(&left.boost(-interface_velocity), &right, a_l, a_r);
        let fluxes = solver.solve_for_flux(
            &left.boost(-interface_velocity),
            &right,
            a_l,
            a_r,
            interface_velocity,
        );

        // Reference solution
        let half_s = Primitives::new(0.57625934, 0.7596532, 0.19952622);
        let flux_s = flux_from_riemann_solution(half_s, interface_velocity, gamma);

        println!("fluxes: {fluxes:?} (reference: {flux_s:?})");
        println!("half: {half:?} (reference {half_s:?})");

        assert_approx_eq!(half.density(), half_s.density(), half_s.density() * 1e-6);
        assert_approx_eq!(half.velocity(), half_s.velocity(), half_s.velocity() * 1e-6);
        assert_approx_eq!(half.pressure(), half_s.pressure(), half_s.pressure() * 1e-6);

        assert_approx_eq!(fluxes.mass(), flux_s.mass(), fluxes.mass() * 1e-5);
        assert_approx_eq!(
            fluxes.momentum(),
            flux_s.momentum(),
            fluxes.momentum() * 1e-5
        );
        assert_approx_eq!(fluxes.energy(), flux_s.energy(), fluxes.energy() * 1e-5);
    }

    #[test]
    fn test_hllc_solver() {
        let gamma = 5. / 3.;
        let interface_velocity = -0.2;
        let solver = HLLCRiemannSolver::new(gamma);
        let eos = EquationOfState::Ideal { gamma };
        let left = Primitives::new(1., 0.2, 0.5);
        let right = Primitives::new(0.5, 0.1, 0.1);
        let fluxes = solver.solve_for_flux(
            &left.boost(-interface_velocity),
            &right.boost(-interface_velocity),
            interface_velocity,
            &eos,
        );

        // reference solution
        let half_s = Primitives::new(0.7064862, 0.49950013, 0.28020517);
        let flux_s = flux_from_riemann_solution(half_s, interface_velocity, gamma);

        assert_approx_eq!(fluxes.mass(), flux_s.mass(), fluxes.mass() * 2e-1);
        assert_approx_eq!(
            fluxes.momentum(),
            flux_s.momentum(),
            fluxes.momentum() * 2e-1
        );
        assert_approx_eq!(fluxes.energy(), flux_s.energy(), fluxes.energy() * 2e-1);
    }
}
