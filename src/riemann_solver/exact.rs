use crate::{
    gas_law::AdiabaticIndex,
    physical_quantities::{Primitive, State},
};

use super::{RiemannStarSolver, RiemannStarValues};

pub struct ExactRiemannSolver;

impl ExactRiemannSolver {
    /// Functions (4.6) and (4.7) in Toro.
    fn fb(p: f64, state: &State<Primitive>, a: f64, gamma: &AdiabaticIndex) -> f64 {
        if p > state.pressure() {
            let cap_a = gamma.tdgp1() / state.density();
            let cap_b = gamma.gm1dgp1() * state.pressure();
            (p - state.pressure()) * (cap_a / (p + cap_b)).sqrt()
        } else {
            gamma.tdgm1() * a * ((p / state.pressure()).powf(gamma.gm1d2g()) - 1.)
        }
    }

    /// Function (4.5) in Toro
    fn f(
        p: f64,
        left: &State<Primitive>,
        right: &State<Primitive>,
        v_l: f64,
        v_r: f64,
        a_l: f64,
        a_r: f64,
        gamma: &AdiabaticIndex,
    ) -> f64 {
        Self::fb(p, left, a_l, gamma) + Self::fb(p, right, a_r, gamma) + (v_r - v_l)
    }

    /// Function (4.37) in Toro
    fn fprimeb(p: f64, state: &State<Primitive>, a: f64, gamma: &AdiabaticIndex) -> f64 {
        if p > state.pressure() {
            let cap_a = gamma.tdgp1() / state.density();
            let cap_b = gamma.gm1dgp1() * state.pressure();
            (1. - 0.5 * (p - state.pressure()) / (cap_b + p)) * (cap_a / (p + cap_b)).sqrt()
        } else {
            1. / state.density() / a * (p / state.pressure()).powf(-gamma.gm1d2g())
        }
    }

    /// The derivative of riemann_f w.r.t. p
    fn fprime(
        p: f64,
        left: &State<Primitive>,
        right: &State<Primitive>,
        a_l: f64,
        a_r: f64,
        gamma: &AdiabaticIndex,
    ) -> f64 {
        Self::fprimeb(p, left, a_l, gamma) + Self::fprimeb(p, right, a_r, gamma)
    }

    /// Bottom function of (4.48) in Toro
    fn gb(p: f64, state: &State<Primitive>, gamma: &AdiabaticIndex) -> f64 {
        let cap_a = gamma.tdgp1() / state.density();
        let cap_b = gamma.gm1dgp1() * state.pressure();
        (cap_a / (p + cap_b)).sqrt()
    }

    /// Get a good first guess for the pressure in the iterative scheme
    ///
    /// This function is based on (4.47) and (4.48) in Toro and on the
    /// FORTRAN code provided in Toro p.156-157
    fn guess_p(
        left: &State<Primitive>,
        right: &State<Primitive>,
        v_l: f64,
        v_r: f64,
        a_l: f64,
        a_r: f64,
        gamma: &AdiabaticIndex,
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
            let base = (a_l + a_r - 0.5 * (gamma.gamma() - 1.) * (v_r - v_l))
                / (a_l / left.pressure().powf(gamma.gm1d2g())
                    + a_r / right.pressure().powf(gamma.gm1d2g()));
            base.powf(gamma.gamma() * gamma.tdgm1())
        } else {
            // two shocks
            (Self::gb(ppv, left, gamma) * left.pressure()
                + Self::gb(ppv, right, gamma) * right.pressure()
                - v_r
                + v_l)
                / (Self::gb(ppv, left, gamma) + Self::gb(ppv, right, gamma))
        };

        p_guess.max(1e-8)
    }

    /// Find the zeropoint of riemann_f(p) using Brent's method
    fn solve_brent(
        lower_lim: f64,
        upper_lim: f64,
        low_f: f64,
        up_f: f64,
        error_tol: f64,
        left: &State<Primitive>,
        right: &State<Primitive>,
        v_l: f64,
        v_r: f64,
        a_l: f64,
        a_r: f64,
        gamma: &AdiabaticIndex,
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

            fs = Self::f(s, left, right, v_l, v_r, a_l, a_r, gamma);
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

    fn shock_middle_density(pdps: f64, state: &State<Primitive>, gamma: &AdiabaticIndex) -> f64 {
        state.density() * (pdps + gamma.gm1dgp1()) / (gamma.gm1dgp1() * pdps + 1.)
    }

    fn rarefaction_middle_density(
        pdps: f64,
        state: &State<Primitive>,
        gamma: &AdiabaticIndex,
    ) -> f64 {
        state.density() * (pdps).powf(1. / gamma.gamma())
    }

    fn middle_density(p: f64, state: &State<Primitive>, gamma: &AdiabaticIndex) -> f64 {
        let pdps = p / state.pressure();
        if pdps > 1. {
            Self::shock_middle_density(pdps, state, gamma)
        } else {
            Self::rarefaction_middle_density(pdps, state, gamma)
        }
    }

    pub(super) fn star_state_from_pstar(
        pstar: f64,
        left: &State<Primitive>,
        right: &State<Primitive>,
        v_l: f64,
        v_r: f64,
        a_l: f64,
        a_r: f64,
        gamma: &AdiabaticIndex,
    ) -> RiemannStarValues {
        // calculate the velocity in the intermediate state
        let u = 0.5 * (v_l + v_r)
            + 0.5 * (Self::fb(pstar, right, a_r, gamma) - Self::fb(pstar, left, a_l, gamma));

        // calculate the left and right intermediate densities
        let rho_l = Self::middle_density(pstar, left, gamma);
        let rho_r = Self::middle_density(pstar, right, gamma);

        RiemannStarValues {
            rho_l,
            rho_r,
            u,
            p: pstar,
        }
    }
}

impl RiemannStarSolver for ExactRiemannSolver {
    fn solve_for_star_state(
        &self,
        left: &State<Primitive>,
        right: &State<Primitive>,
        v_l: f64,
        v_r: f64,
        a_l: f64,
        a_r: f64,
        gamma: &AdiabaticIndex,
    ) -> RiemannStarValues {
        /* We normally use a Newton-Raphson iteration to find the zeropoint
        of riemann_f(p), but if pstar is close to 0, we risk negative p values.
        Since riemann_f(p) is undefined for negative pressures, we don't
        want this to happen.
        We therefore use Brent's method if riemann_f(0) is larger than some
        value. -5 makes the iteration fail safe while almost never invoking
        the expensive Brent solver. */
        let mut p = 0.;
        let mut p_guess = Self::guess_p(left, right, v_l, v_r, a_l, a_r, gamma);
        let mut fp = Self::f(p, left, right, v_l, v_r, a_l, a_r, gamma);
        let mut fp_guess = Self::f(p_guess, left, right, v_l, v_r, a_l, a_r, gamma);
        if fp * fp_guess >= 0. {
            // Newton-Raphson until convergence or until suitable interval is found
            // to use Brent's method
            let mut counter = 0;
            while (p - p_guess).abs() > 1e-6 * 0.5 * (p + p_guess) && fp_guess < 0.0 {
                p = p_guess;
                p_guess = p_guess - fp_guess / Self::fprime(p_guess, left, right, a_l, a_r, gamma);
                fp_guess = Self::f(p_guess, left, right, v_l, v_r, a_l, a_r, gamma);
                counter += 1;
                if counter > 1000 {
                    panic!("Stuck in Newton-Raphson iteration!");
                }
            }
        }

        // As soon as there is a suitable interval: use Brent's method
        if (p - p_guess).abs() > 1e-6 * 0.5 * (p + p_guess) && fp_guess > 0. {
            p = 0.;
            fp = Self::f(p, left, right, v_l, v_r, a_l, a_r, gamma);
            p = Self::solve_brent(
                p, p_guess, fp, fp_guess, 1e-6, left, right, v_l, v_r, a_l, a_r, gamma,
            );
        } else {
            p = p_guess;
        }

        Self::star_state_from_pstar(p, left, right, v_l, v_r, a_l, a_r, gamma)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        gas_law::{EquationOfState, GasLaw},
        physical_quantities::{Conserved, State},
        riemann_solver::RiemannFluxSolver,
    };

    use super::*;

    use float_cmp::assert_approx_eq;
    use glam::DVec3;

    const GAMMA: f64 = 5. / 3.;

    fn get_eos(gamma: f64) -> GasLaw {
        GasLaw::new(gamma, EquationOfState::Ideal)
    }

    #[test]
    fn test_symmetry() {
        let interface_velocity = -3e-1 * DVec3::X;
        let left = State::<Primitive>::new(1., DVec3::ZERO, 1.);
        let left_reversed = State::<Primitive>::new(1., DVec3::ZERO, 1.);
        let right = State::<Primitive>::new(1., -6e-1 * DVec3::X, 1.);
        let right_reversed = State::<Primitive>::new(1., 6e-1 * DVec3::X, 1.);
        let eos = get_eos(GAMMA);
        let solver = ExactRiemannSolver;

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
        fn get_half(
            rho_l: f64,
            v_l: f64,
            p_l: f64,
            rho_r: f64,
            v_r: f64,
            p_r: f64,
        ) -> State<Primitive> {
            let left = State::<Primitive>::new(rho_l, v_l * DVec3::X, p_l);
            let right = State::<Primitive>::new(rho_r, v_r * DVec3::X, p_r);
            let eos = get_eos(GAMMA);
            let a_l = eos.sound_speed(left.pressure(), 1. / left.density());
            let a_r = eos.sound_speed(right.pressure(), 1. / right.density());

            ExactRiemannSolver.sample(&left, &right, v_l, v_r, a_l, a_r, DVec3::X, eos.gamma())
        }

        // Left shock
        let half = get_half(0.75, 0.6, 0.8, 0.55, -0.25, 0.5);
        assert_approx_eq!(f64, half.density(), 0.8927393244785966);
        assert_approx_eq!(f64, half.velocity().x, 0.35966250129769556);
        assert_approx_eq!(f64, half.pressure(), 1.0709476357371466);

        // right shock
        let half = get_half(0.75, 0.3, 0.8, 0.55, -0.75, 0.5);
        assert_approx_eq!(f64, half.density(), 0.9114320476995261);
        assert_approx_eq!(f64, half.velocity().x, -0.03895103710637815);
        assert_approx_eq!(f64, half.pressure(), 1.201228148376388);

        // Left rarefaction (middle state)
        let half = get_half(0.75, 0.1, 0.8, 0.55, 0.5, 0.5);
        assert_approx_eq!(f64, half.density(), 0.5562398903431287);
        assert_approx_eq!(f64, half.velocity().x, 0.4792910646431622);
        assert_approx_eq!(f64, half.pressure(), 0.4861363537732538);

        // Left rarefaction (fan)
        let half = get_half(1.75, -0.1, 0.75, 0.55, 1.5, 0.5);
        assert_approx_eq!(f64, half.density(), 0.65432665196683);
        assert_approx_eq!(f64, half.velocity().x, 0.6088656910463873);
        assert_approx_eq!(f64, half.pressure(), 0.14554217677392042);

        // Right rarefaction (middle state)
        let half = get_half(0.75, -0.5, 0.8, 0.55, -0.1, 0.5);
        assert_approx_eq!(f64, half.density(), 0.5407985848180775);
        assert_approx_eq!(f64, half.velocity().x, -0.1207089353568378);
        assert_approx_eq!(f64, half.pressure(), 0.4861363537732538);

        // Right rarefaction (fan)
        let half = get_half(1.75, -1.5, 0.75, 1.55, 0.1, 0.5);
        assert_approx_eq!(f64, half.density(), 0.5687181105004402);
        assert_approx_eq!(f64, half.velocity().x, -0.5249266813300747);
        assert_approx_eq!(f64, half.pressure(), 0.09402548983542296);
    }

    #[test]
    fn test_invariance() {
        let eos: GasLaw = get_eos(GAMMA);

        let left = State::<Primitive>::new(1.5, 0.2 * DVec3::X, 1.2);
        let right = State::<Primitive>::new(0.7, -0.4 * DVec3::X, 0.1);

        let interface_velocity = -0.1 * DVec3::X;

        let w_half_lab = ExactRiemannSolver.sample(
            &left,
            &right,
            left.velocity().x,
            right.velocity().x,
            eos.sound_speed(left.pressure(), 1. / left.density()),
            eos.sound_speed(right.pressure(), 1. / right.density()),
            DVec3::X,
            eos.gamma(),
        );
        let roe = w_half_lab.pressure() * eos.gamma().odgm1()
            + 0.5 * w_half_lab.density() * w_half_lab.velocity().length_squared();
        let fluxes_lab =
            ExactRiemannSolver.solve_for_flux(&left, &right, DVec3::ZERO, DVec3::X, &eos)
                - State::<Conserved>::new(
                    w_half_lab.density() * interface_velocity.x,
                    w_half_lab.density() * w_half_lab.velocity() * interface_velocity,
                    roe * interface_velocity.x,
                );
        let fluxes_face = ExactRiemannSolver.solve_for_flux(
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

    #[test]
    fn test_problem_swift() {
        let left = State::<Primitive>::new(
            2.1669723793e-9,
            DVec3::new(-2.9266309738e+00, 7.4537420273e+00, 1.4240785599e+01),
            1.4698121886e-10,
        );
        let right = State::<Primitive>::new(
            1.7172516742e-09,
            DVec3::new(1.8704223633e+00, -7.5396900177e+00, -1.4482626915e+01),
            1.1778487907e-10,
        );
        let v_l = -1.5249514580e+00;
        let v_r = 4.9803251028e-01;
        let a_l = 3.3622390032e-01;
        let a_r = 3.3810544014e-01;

        let eos = get_eos(GAMMA);

        ExactRiemannSolver.solve_for_star_state(&left, &right, v_l, v_r, a_l, a_r, eos.gamma());
    }
}
