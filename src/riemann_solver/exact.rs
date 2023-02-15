use crate::{equation_of_state::EquationOfState, physical_quantities::Primitives};

use super::{RiemannStarSolver, RiemannStarValues};

pub struct ExactRiemannSolver {
    // Adiabatic index gamma.
    gamma: f64,
    // (gamma - 1) / (2 gamma)
    gm1d2g: f64,
    // (gamma - 1) / (gamma + 1)
    gm1dgp1: f64,
    // 2 / (gamma - 1)
    tdgm1: f64,
    // 2 / (gamma + 1)
    tdgp1: f64,
}

impl ExactRiemannSolver {
    pub fn new(gamma: f64) -> Self {
        Self {
            gamma,
            gm1dgp1: (gamma - 1.) / (gamma + 1.),
            gm1d2g: (gamma - 1.) / (2. * gamma),
            tdgm1: 2. / (gamma - 1.),
            tdgp1: 2. / (gamma + 1.),
        }
    }

    /// Functions (4.6) and (4.7) in Toro.
    fn fb(&self, p: f64, state: &Primitives, a: f64) -> f64 {
        if p > state.pressure() {
            let cap_a = self.tdgp1 / state.density();
            let cap_b = self.gm1dgp1 * state.pressure();
            (p - state.pressure()) * (cap_a / (p + cap_b)).sqrt()
        } else {
            self.tdgm1 * a * ((p / state.pressure()).powf(self.gm1d2g) - 1.)
        }
    }

    /// Function (4.5) in Toro
    fn f(
        &self,
        p: f64,
        left: &Primitives,
        right: &Primitives,
        v_l: f64,
        v_r: f64,
        a_l: f64,
        a_r: f64,
    ) -> f64 {
        self.fb(p, left, a_l) + self.fb(p, right, a_r) + (v_r - v_l)
    }

    /// Function (4.37) in Toro
    fn fprimeb(&self, p: f64, state: &Primitives, a: f64) -> f64 {
        if p > state.pressure() {
            let cap_a = self.tdgp1 / state.density();
            let cap_b = self.gm1dgp1 * state.pressure();
            (1. - 0.5 * (p - state.pressure()) / (cap_b + p)) * (cap_a / (p + cap_b)).sqrt()
        } else {
            1. / state.density() / a * (p / state.pressure()).powf(-self.gm1d2g)
        }
    }

    /// The derivative of riemann_f w.r.t. p
    fn fprime(&self, p: f64, left: &Primitives, right: &Primitives, a_l: f64, a_r: f64) -> f64 {
        self.fprimeb(p, left, a_l) + self.fprimeb(p, right, a_r)
    }

    /// Bottom function of (4.48) in Toro
    fn gb(&self, p: f64, state: &Primitives) -> f64 {
        let cap_a = self.tdgp1 / state.density();
        let cap_b = self.gm1dgp1 * state.pressure();
        (cap_a / (p + cap_b)).sqrt()
    }

    /// Get a good first guess for the pressure in the iterative scheme
    ///
    /// This function is based on (4.47) and (4.48) in Toro and on the
    /// FORTRAN code provided in Toro p.156-157
    fn guess_p(
        &self,
        left: &Primitives,
        right: &Primitives,
        v_l: f64,
        v_r: f64,
        a_l: f64,
        a_r: f64,
    ) -> f64 {
        let p_min = left.pressure().min(right.pressure());
        let p_max = left.pressure().max(right.pressure());
        let q_max = p_max / p_min;
        let ppv = 0.5 * (left.pressure() + right.pressure())
            - 0.125 * (v_r - v_l) * (left.density() + right.density()) * (a_l + a_r);
        let ppv = ppv.max(1e-8);
        let p_guess = if q_max <= 2. && p_min <= ppv && ppv <= p_max {
            ppv
        } else if ppv < p_min {
            // two rarefactions
            let base = (a_l + a_r - 0.5 * (self.gamma - 1.) * (v_r - v_l))
                / (a_l / left.pressure().powf(self.gm1d2g)
                    + a_r / right.pressure().powf(self.gm1d2g));
            base.powf(self.gamma * self.tdgm1)
        } else {
            // two shocks
            (self.gb(ppv, left) * left.pressure() + self.gb(ppv, right) * right.pressure() - v_r
                + v_l)
                / (self.gb(ppv, left) + self.gb(ppv, right))
        };

        p_guess.max(1e-8)
    }

    /// Find the zeropoint of riemann_f(p) using Brent's method
    fn solve_brent(
        &self,
        lower_lim: f64,
        upper_lim: f64,
        low_f: f64,
        up_f: f64,
        error_tol: f64,
        left: &Primitives,
        right: &Primitives,
        v_l: f64,
        v_r: f64,
        a_l: f64,
        a_r: f64,
    ) -> f64 {
        let mut a = lower_lim;
        let mut b = upper_lim;
        let mut c;
        let mut d = f64::INFINITY;

        let mut fa = low_f;
        let mut fb = up_f;
        let mut fc;

        let mut s;
        let mut fs;

        if fa * fb > 0. {
            panic!("Brent's method was called with equal sign function values!");
        }

        // if |f(a)| < |f(b)| then swap (a,b)
        if fa.abs() < fb.abs() {
            (a, b) = (b, a);
            (fa, fb) = (fb, fa);
        }

        c = a;
        fc = fa;
        let mut mflag = true;

        while fb != 0. && (a - b).abs() > error_tol * 0.5 * (a + b) {
            s = if fa != fc && fb != fc {
                // Inverse quadratic interpolation
                a * fb * fc / (fa - fb) / (fa - fc)
                    + b * fa * fc / (fb - fa) / (fb - fc)
                    + c * fa * fb / (fc - fa) / (fc - fb)
            } else {
                // Secant rule
                b - fb * (b - a) / (fb - fa)
            };

            let tmp = 0.25 * (3. * a + b);

            if !((s > tmp && s < b) || (s < tmp && s > b))
                || (mflag && (s - b).abs() >= (0.5 * (b - c).abs()))
                || (!mflag && (s - b).abs() >= (0.5 * (c - d).abs()))
                || (mflag && (b - c).abs() < 0.5 * error_tol * (b + c))
                || (!mflag && (c - d).abs() < 0.5 * error_tol * (c + d))
            {
                s = 0.5 * (a + b);
                mflag = true;
            } else {
                mflag = false;
            }

            fs = self.f(s, left, right, v_l, v_r, a_l, a_r);
            d = c;
            c = b;
            fc = fb;
            if fa * fs < 0. {
                b = s;
                fb = fs;
            } else {
                a = s;
                fa = fs;
            }

            // if |f(a)| < |f(b)| then swap (a,b)
            if fa.abs() < fb.abs() {
                (a, b) = (b, a);
                (fa, fb) = (fb, fa);
            }
        }

        b
    }

    fn shock_middle_density(&self, pdps: f64, state: &Primitives, eos: &EquationOfState) -> f64 {
        state.density() * (pdps + eos.gm1dgp1()) / (eos.gm1dgp1() * pdps + 1.)
    }

    fn rarefaction_middle_density(
        &self,
        pdps: f64,
        state: &Primitives,
        eos: &EquationOfState,
    ) -> f64 {
        state.density() * (pdps).powf(1. / eos.gamma())
    }

    fn middle_density(&self, p: f64, state: &Primitives, eos: &EquationOfState) -> f64 {
        let pdps = p / state.pressure();
        if pdps > 1. {
            self.shock_middle_density(pdps, state, eos)
        } else {
            self.rarefaction_middle_density(pdps, state, eos)
        }
    }
}

impl RiemannStarSolver for ExactRiemannSolver {
    fn solve_for_star_state(
        &self,
        left: &Primitives,
        right: &Primitives,
        v_l: f64,
        v_r: f64,
        a_l: f64,
        a_r: f64,
        eos: &EquationOfState,
    ) -> RiemannStarValues {
        /* We normally use a Newton-Raphson iteration to find the zeropoint
        of riemann_f(p), but if pstar is close to 0, we risk negative p values.
        Since riemann_f(p) is undefined for negative pressures, we don't
        want this to happen.
        We therefore use Brent's method if riemann_f(0) is larger than some
        value. -5 makes the iteration fail safe while almost never invoking
        the expensive Brent solver. */
        let mut p = 0.;
        let mut p_guess = self.guess_p(left, right, v_l, v_r, a_l, a_r);
        let mut fp = self.f(p, left, right, v_l, v_r, a_l, a_r);
        let mut fp_guess = self.f(p_guess, left, right, v_l, v_r, a_l, a_r);
        if fp * fp_guess >= 0. {
            // Newton-Raphson until convergence or until suitable interval is found
            // to use Brent's method
            let mut counter = 0;
            while (p - p_guess).abs() > 1e-6 * 0.5 * (p + p_guess) && fp_guess < 0.0 {
                p = p_guess;
                p_guess = p_guess - fp_guess / self.fprime(p_guess, left, right, a_l, a_r);
                fp_guess = self.f(p_guess, left, right, v_l, v_r, a_l, a_r);
                counter += 1;
                if counter > 1000 {
                    panic!("Stuck in Newton-Raphson iteration!");
                }
            }
        }

        // As soon as there is a suitable interval: use Brent's method
        if (p - p_guess).abs() > 1e-6 * 0.5 * (p + p_guess) && fp_guess > 0. {
            p = 0.;
            fp = self.f(p, left, right, v_l, v_r, a_l, a_r);
            p = self.solve_brent(
                p, p_guess, fp, fp_guess, 1e-6, left, right, v_l, v_r, a_l, a_r,
            );
        } else {
            p = p_guess;
        }

        // calculate the velocity in the intermediate state
        let u = 0.5 * (v_l + v_r) + 0.5 * (self.fb(p, right, a_r) - self.fb(p, left, a_l));

        // calculate the left and right intermediate densities
        let rho_l = self.middle_density(p, left, eos);
        let rho_r = self.middle_density(p, right, eos);

        RiemannStarValues { rho_l, rho_r, u, p }
    }
}

#[cfg(test)]
mod test {
    use crate::riemann_solver::RiemannFluxSolver;

    use super::*;

    use float_cmp::assert_approx_eq;
    use glam::DVec3;
    use yaml_rust::YamlLoader;

    const GAMMA: f64 = 5. / 3.;

    fn get_eos(gamma: f64) -> EquationOfState {
        EquationOfState::new(&YamlLoader::load_from_str(&format!("gamma: {:}", gamma)).unwrap()[0])
            .unwrap()
    }

    #[test]
    fn test_symmetry() {
        let solver = ExactRiemannSolver::new(GAMMA);
        let interface_velocity = -3e-1 * DVec3::X;
        let left = Primitives::new(1., DVec3::ZERO, 1.);
        let left_reversed = Primitives::new(1., DVec3::ZERO, 1.);
        let right = Primitives::new(1., -6e-1 * DVec3::X, 1.);
        let right_reversed = Primitives::new(1., 6e-1 * DVec3::X, 1.);
        let eos = get_eos(GAMMA);

        let fluxes = solver.solve_for_flux(
            &left.boost(-interface_velocity),
            &right.boost(-interface_velocity),
            interface_velocity,
            DVec3::X,
            &eos,
        );
        let fluxes_reversed = solver.solve_for_flux(
            &right_reversed.boost(interface_velocity),
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
    fn test_half_state() {
        let solver = ExactRiemannSolver::new(GAMMA);

        fn get_half(
            rho_l: f64,
            v_l: f64,
            p_l: f64,
            rho_r: f64,
            v_r: f64,
            p_r: f64,
            solver: &ExactRiemannSolver,
        ) -> Primitives {
            let left = Primitives::new(rho_l, v_l * DVec3::X, p_l);
            let right = Primitives::new(rho_r, v_r * DVec3::X, p_r);
            let eos = get_eos(GAMMA);
            let a_l = eos.sound_speed(left.pressure(), 1. / left.density());
            let a_r = eos.sound_speed(right.pressure(), 1. / right.density());

            let star = solver.solve_for_star_state(&left, &right, v_l, v_r, a_l, a_r, &eos);
            solver.sample(&star, &left, &right, v_l, v_r, a_l, a_r, DVec3::X, &eos)
        }

        // Left shock
        let half = get_half(0.75, 0.6, 0.8, 0.55, -0.25, 0.5, &solver);
        assert_approx_eq!(f64, half.density(), 0.8927393244785966);
        assert_approx_eq!(f64, half.velocity().x, 0.35966250129769556);
        assert_approx_eq!(f64, half.pressure(), 1.0709476357371466);

        // right shock
        let half = get_half(0.75, 0.3, 0.8, 0.55, -0.75, 0.5, &solver);
        assert_approx_eq!(f64, half.density(), 0.9114320476995261);
        assert_approx_eq!(f64, half.velocity().x, -0.03895103710637815);
        assert_approx_eq!(f64, half.pressure(), 1.201228148376388);

        // Left rarefaction (middle state)
        let half = get_half(0.75, 0.1, 0.8, 0.55, 0.5, 0.5, &solver);
        assert_approx_eq!(f64, half.density(), 0.5562398903431287);
        assert_approx_eq!(f64, half.velocity().x, 0.4792910646431622);
        assert_approx_eq!(f64, half.pressure(), 0.4861363537732538);

        // Left rarefaction (fan)
        let half = get_half(1.75, -0.1, 0.75, 0.55, 1.5, 0.5, &solver);
        assert_approx_eq!(f64, half.density(), 0.65432665196683);
        assert_approx_eq!(f64, half.velocity().x, 0.6088656910463873);
        assert_approx_eq!(f64, half.pressure(), 0.14554217677392042);

        // Right rarefaction (middle state)
        let half = get_half(0.75, -0.5, 0.8, 0.55, -0.1, 0.5, &solver);
        assert_approx_eq!(f64, half.density(), 0.5407985848180775);
        assert_approx_eq!(f64, half.velocity().x, -0.1207089353568378);
        assert_approx_eq!(f64, half.pressure(), 0.4861363537732538);

        // Right rarefaction (fan)
        let half = get_half(1.75, -1.5, 0.75, 1.55, 0.1, 0.5, &solver);
        assert_approx_eq!(f64, half.density(), 0.5687181105004402);
        assert_approx_eq!(f64, half.velocity().x, -0.5249266813300747);
        assert_approx_eq!(f64, half.pressure(), 0.09402548983542296);
    }
}
