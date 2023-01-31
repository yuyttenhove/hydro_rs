use glam::DVec3;

use crate::{
    equation_of_state::EquationOfState,
    physical_quantities::{Conserved, Primitives},
};

use super::{flux_from_half_state, RiemannSolver, VacuumRiemannSolver};

pub struct ExactRiemannSolver {
    // Adiabatic index gamma.
    gamma: f64,
    // (gamma + 1) / (gamma)
    gp1dg: f64,
    // (gamma - 1) / (2 gamma)
    gm1d2g: f64,
    // (gamma - 1) / (gamma + 1)
    gm1dgp1: f64,
    // 1 / (gamma - 1)
    odgm1: f64,
    // 2 / (gamma - 1)
    tdgm1: f64,
    // 2 / (gamma + 1)
    tdgp1: f64,

    vacuum_solver: VacuumRiemannSolver,
}

impl ExactRiemannSolver {
    pub fn new(gamma: f64) -> Self {
        Self {
            gamma,
            gm1dgp1: (gamma - 1.) / (gamma + 1.),
            gm1d2g: (gamma - 1.) / (2. * gamma),
            odgm1: 1. / (gamma - 1.),
            tdgm1: 2. / (gamma - 1.),
            tdgp1: 2. / (gamma + 1.),
            gp1dg: (gamma + 1.) / gamma,
            vacuum_solver: VacuumRiemannSolver::new(gamma),
        }
    }

    /// Functions (4.6) and (4.7) in Toro.
    fn fb(&self, p: f64, state: &Primitives, a: f64) -> f64 {
        if p > state.pressure() {
            let cap_a = self.tdgp1 / state.density();
            let cap_b = self.gm1dgp1 * state.pressure();
            (p - state.pressure()) * (cap_a / (p + cap_b)).sqrt()
        } else {
            self.tdgm1 * a * (p / state.pressure() - 1.).powf(self.gm1d2g)
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
    fn fprime(
        &self,
        p: f64,
        left: &Primitives,
        right: &Primitives,
        v_l: f64,
        v_r: f64,
        a_l: f64,
        a_r: f64,
    ) -> f64 {
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
        let p_guess = if ppv < p_min {
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
        let mut c = 0.;
        let mut d = f64::INFINITY;

        let mut fa = low_f;
        let mut fb = up_f;
        let mut fc = 0.;
        let mut s = 0.;
        let mut fs = 0.;

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

            if !((s > tmp && s < b)
                || (s < tmp && s > b)
                || (mflag && (s - b).abs() >= (0.5 * (b - c).abs()))
                || (!mflag && (s - b).abs() >= (0.5 * (c - d).abs()))
                || (mflag && (b - c).abs() < error_tol)
                || (!mflag && (c - d).abs() < error_tol))
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

    fn sample_shock_wave(
        &self,
        p: f64,
        u: f64,
        state: &Primitives,
        v: f64,
        a: f64,
        n_unit: DVec3,
    ) -> Primitives {
        let pdp = p / state.pressure();
        let s = v - a * (0.5 * self.gp1dg * pdp + self.gm1d2g).sqrt();
        if s < 0. {
            Primitives::new(
                state.density() * (pdp + self.gm1dgp1) / (self.gm1dgp1 * pdp + 1.),
                state.velocity() + (u - v) * n_unit,
                state.pressure(),
            )
        } else {
            *state
        }
    }

    fn sample_rarefaction_wave(
        &self,
        p: f64,
        u: f64,
        state: &Primitives,
        v: f64,
        a: f64,
        n_unit: DVec3,
    ) -> Primitives {
        let sh = v - a;
        if sh < 0. {
            let st = u - a * (p / state.pressure()).powf(self.gm1d2g);
            if st > 0. {
                let v_half = self.tdgp1 * (a + 0.5 * (self.gamma - 1.) * v) - v;
                let base = self.tdgp1 + self.gm1dgp1 / a * v;
                Primitives::new(
                    state.density() * base.powf(self.tdgm1),
                    state.velocity() + v_half * n_unit,
                    state.pressure() * base.powf(self.tdgp1),
                )
            } else {
                Primitives::new(
                    state.density() * (p / state.pressure()).powf(1. / self.gamma),
                    state.velocity() * (u - v) * n_unit,
                    state.pressure(),
                )
            }
        } else {
            *state
        }
    }

    /// Solve the Riemann problem between the given left and right state and
    /// along the given interface normal for the half state.
    ///
    /// Based on chapter 4 in Toro
    fn solve(
        &self,
        left: &Primitives,
        right: &Primitives,
        v_l: f64,
        v_r: f64,
        a_l: f64,
        a_r: f64,
        n_unit: DVec3,
    ) -> Primitives {
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
                p_guess =
                    p_guess - fp_guess / self.fprime(p_guess, left, right, v_l, v_r, a_l, a_r);
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

        // Sample the solution.
        // This corresponds to the flow chart in Fig. 4.14 in Toro
        if u < 0. {
            if p > right.pressure() {
                self.sample_shock_wave(p, u, right, v_r, -a_r, n_unit)
            } else {
                self.sample_rarefaction_wave(p, u, right, v_r, -a_r, n_unit)
            }
        } else {
            if p > left.pressure() {
                self.sample_shock_wave(p, u, left, v_l, a_l, n_unit)
            } else {
                self.sample_rarefaction_wave(p, u, left, v_l, a_l, n_unit)
            }
        }
    }
}

impl RiemannSolver for ExactRiemannSolver {
    fn solve_for_flux(
        &self,
        left: &Primitives,
        right: &Primitives,
        interface_velocity: glam::DVec3,
        n_unit: glam::DVec3,
        eos: &EquationOfState,
    ) -> Conserved {
        let v_l = left.velocity().dot(n_unit);
        let v_r = right.velocity().dot(n_unit);
        let a_l = eos.sound_speed(left.pressure(), 1. / left.density());
        let a_r = eos.sound_speed(right.pressure(), 1. / right.density());

        // velocity difference
        let v_r_m_v_l = v_r - v_l;

        // handle vacuum
        if self
            .vacuum_solver
            .is_vacuum(left, right, a_l, a_r, v_r_m_v_l)
        {
            return self.vacuum_solver.solve_for_flux(
                left,
                right,
                a_l,
                a_r,
                n_unit,
                interface_velocity,
            );
        }

        let half = self.solve(left, right, v_l, v_r, a_l, a_r, n_unit);
        flux_from_half_state(&half, interface_velocity, self.odgm1, n_unit)
    }
}