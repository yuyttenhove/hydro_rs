use crate::{part::{Primitives, Conserved}, equation_of_state::EquationOfState};

/// HLLC Riemann solver
pub struct RiemannSolver {
    /* Adiabatic index \f$\gamma{}\f$. */
    gamma: f64,
    /* \f$\frac{2(\gamma{}+1)}{\gamma{}}\f$. */
    hgp1dg: f64,
    /* \f$\frac{1}{\gamma{}-1}\f$. */
    odgm1: f64,
}

impl RiemannSolver {
    pub fn new(gamma: f64) -> Self {
        RiemannSolver { gamma, hgp1dg: 0.5 * (gamma + 1.) / gamma, odgm1: 1. / (gamma - 1.) }
    }

    pub fn solve_for_flux(&self, left: &Primitives, right: &Primitives, interface_velocity: f64, eos: &EquationOfState) -> Conserved {
        // Consistency check
        if let EquationOfState::Ideal { gamma } = eos { assert_eq!(*gamma, self.gamma); }

        // Handle full vacuum 
        if left.density() == 0. && right.density() == 0. { Conserved::vacuum(); } 

        // Inverse densities
        let rho_l_inv = 1. / left.density();
        let rho_r_inv = 1. / right.density();
        let a_l = eos.sound_speed(left.pressure(), rho_l_inv);
        let a_r = eos.sound_speed(right.pressure(), rho_r_inv).sqrt();

        // velocity difference
        let v_r_m_v_l = right.velocity() - left.velocity();

        // handle vacuum
        if left.density() == 0. || right.density() == 0. {
            todo!();
        }
        if 2. * self.odgm1 * (a_l + a_r) <= v_r_m_v_l {
            todo!();
        }

        // STEP 1: Pressure estimate 
        let ppvrs = 0.5 * (left.pressure() + right.pressure()) 
                         - 0.25 * v_r_m_v_l * (left.density() + right.density()) * (a_l + a_r); // -0.125
        let p_star = ppvrs.max(0.);

        // STEP 2: wave speed estimates
        let mut q_l = 1.;
        if p_star > left.pressure() {
            q_l = (1. + self.hgp1dg * (p_star / left.pressure() - 1.)).sqrt();
        }
        let mut q_r = 1.;
        if p_star > right.pressure() {
            q_r = (1. + self.hgp1dg * (p_star / right.pressure() - 1.)).sqrt();
        }

        let s_l_m_v_l = - a_l * q_l;
        let s_r_m_v_r = a_r * q_r;
        let s_star = (right.pressure() - left.pressure() + left.density() * left.velocity() * s_l_m_v_l 
                            - right.density() * right.velocity() * s_r_m_v_r) 
                         / (left.density() * s_l_m_v_l - right.density() * s_r_m_v_r);
        
        // STEP 3: HLLC flux in a frame moving with the interface velocity            
        let mut flux;
        if s_star  >= 0. {
            // flux FL
            let v_l2 = left.velocity() * left.velocity();
            let rho_l_v_l = left.density() * left.velocity();
            let e_l = self.odgm1 * left.pressure() * rho_l_inv + 0.5 * v_l2;
            let rho_l_e_l = left.density() * e_l;
            let s_l = s_l_m_v_l + left.velocity();
            flux = Conserved::new(rho_l_v_l, 
                                  left.density() * v_l2 + left.pressure(), 
                                  (rho_l_e_l + left.pressure()) * left.velocity());
            if s_l < 0. {
                // flux FL*
                let rho_star_l = left.density() * s_l_m_v_l / (s_l - s_star);
                let u_star_l = rho_star_l * s_star;
                let p_star_l = rho_star_l * (e_l + (s_star - left.velocity()) * (s_star + left.pressure() / (left.density() * s_l_m_v_l)));
                flux += s_l * Conserved::new(rho_star_l - left.density(), 
                                             u_star_l - rho_l_v_l, 
                                             p_star_l - rho_l_e_l);
            }
        } else {
            // flux FR
            let v_r2 = right.velocity() * right.velocity();
            let rho_r_v_r = right.density() * right.velocity();
            let e_r = self.odgm1 * right.pressure() * rho_r_inv + 0.5 * v_r2;
            let rho_r_e_r = right.density() * e_r;
            let s_r = s_r_m_v_r + right.velocity();
            flux = Conserved::new(rho_r_v_r, 
                                  right.density() * v_r2 + right.pressure(), 
                                  (rho_r_e_r + right.pressure()) * right.velocity());
            if s_r > 0. {
                // flux FR*
                let rho_star_r = right.density() * s_r_m_v_r / (s_r - s_star);
                let u_star_r = rho_star_r * s_star;
                let p_star_r = rho_star_r * (e_r + (s_star - right.velocity()) * (s_star + right.pressure() / (right.density() * s_r_m_v_r)));
                flux += s_r * Conserved::new(rho_star_r - right.density(), 
                                             u_star_r - rho_r_v_r, 
                                             p_star_r - rho_r_e_r);
            }
        }
        debug_assert!(!(flux.mass().is_nan() || flux.mass().is_infinite()));

        // Deboost to lab frame
        flux += Conserved::new(
            0., 
            interface_velocity * flux.mass(), 
            interface_velocity * flux.momentum() + 0.5 * interface_velocity * interface_velocity * flux.mass()
        );
        flux
    }
}