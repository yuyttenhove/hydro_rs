mod airs;
mod exact;
mod hllc;
mod pvrs;
mod trrs;
mod tsrs;
mod vacuum;
mod linear_advection;
mod anrs;

use glam::DVec3;

use crate::{
    errors::MVMMError,
    gas_law::{AdiabaticIndex, GasLaw},
    physical_quantities::{Conserved, Primitive, State},
};

pub use airs::AIRiemannSolver;
pub use exact::ExactRiemannSolver;
pub use hllc::HLLCRiemannSolver;
pub use pvrs::PVRiemannSolver;
pub use trrs::TRRiemannSolver;
pub use tsrs::TSRiemannSolver;
pub use linear_advection::LinearAdvectionRiemannSover;

use self::vacuum::VacuumRiemannSolver;

pub fn riemann_solver(
    kind: &str,
    airs_threshold: Option<f64>,
) -> Result<Box<dyn RiemannFluxSolver>, MVMMError> {
    match kind {
        "HLLC" => Ok(Box::new(HLLCRiemannSolver)),
        "Exact" => Ok(Box::new(ExactRiemannSolver)),
        "PVRS" => Ok(Box::new(PVRiemannSolver)),
        "AIRS" => {
            let threshold = airs_threshold.ok_or(MVMMError::MissingAIRSThreshold)?;
            Ok(Box::new(AIRiemannSolver::new(threshold)))
        }
        "TSRS" => Ok(Box::new(TSRiemannSolver)),
        "TRRS" => Ok(Box::new(TRRiemannSolver)),
        _ => Err(MVMMError::UnknownRiemannSolver(kind.to_string())),
    }
}

fn flux_from_half_state(
    half: &State<Primitive>,
    interface_velocity: DVec3,
    n_unit: DVec3,
    gamma: &AdiabaticIndex,
) -> State<Conserved> {
    let v_tot = interface_velocity + half.velocity();
    let mut flux = [DVec3::ZERO; 5];

    flux[0] = half.density() * half.velocity();

    flux[1] = half.density() * v_tot.x * half.velocity();
    flux[1].x += half.pressure();
    flux[2] = half.density() * v_tot.y * half.velocity();
    flux[2].y += half.pressure();
    flux[3] = half.density() * v_tot.z * half.velocity();
    flux[3].z += half.pressure();

    let roe = half.pressure() * gamma.odgm1() + 0.5 * half.density() * v_tot.length_squared();
    flux[4] = roe * half.velocity() + half.pressure() * v_tot;

    State::<Conserved>::new(
        flux[0].dot(n_unit),
        DVec3 {
            x: flux[1].dot(n_unit),
            y: flux[2].dot(n_unit),
            z: flux[3].dot(n_unit),
        },
        flux[4].dot(n_unit),
    )
}

pub trait RiemannFluxSolver: Sync {
    fn solve_for_flux(
        &self,
        left: &State<Primitive>,
        right: &State<Primitive>,
        interface_velocity: DVec3,
        n_unit: DVec3,
        eos: &GasLaw,
    ) -> State<Conserved>;
}

pub trait RiemannWafFluxSolver: Sync {
    fn solve_for_waf_flux(
        &self,
        left: &State<Primitive>,
        right: &State<Primitive>,
        dx_left: DVec3,
        dx_right: DVec3,
        drho_left: f64,
        drho_right: f64,
        interface_velocity: DVec3,
        dt: f64,
        n_unit: DVec3,
        eos: &GasLaw,
    ) -> State<Conserved>;
}

#[derive(Default)]
struct RiemannStarValues {
    rho_l: f64,
    rho_r: f64,
    u: f64,
    p: f64,
}
trait RiemannStarSolver: RiemannFluxSolver {
    /// Solve for the flux in a reference frame where the interface is not moving
    fn solve_for_star_state(
        &self,
        left: &State<Primitive>,
        right: &State<Primitive>,
        v_l: f64,
        v_r: f64,
        a_l: f64,
        a_r: f64,
        gamma: &AdiabaticIndex,
    ) -> RiemannStarValues;

    fn shock_speed(v: f64, a: f64, pdps: f64, gamma: &AdiabaticIndex) -> f64 {
        v - a * (0.5 * gamma.gp1dg() * pdps + gamma.gm1d2g()).sqrt()
    }

    fn rarefaction_head_speed(v: f64, a: f64) -> f64 {
        v - a
    }

    fn rarefaction_tail_speed(u: f64, a: f64, pdps: f64, gamma: &AdiabaticIndex) -> f64 {
        u - a * pdps.powf(gamma.gm1d2g())
    }

    fn sample_rarefaction_fan(
        state: &State<Primitive>,
        a: f64,
        v: f64,
        n_unit: DVec3,
        gamma: &AdiabaticIndex,
    ) -> State<Primitive> {
        let v_half = gamma.tdgp1() * (a + 0.5 * (gamma.gamma() - 1.) * v);
        let base = gamma.tdgp1() + gamma.gm1dgp1() / a * v;
        State::<Primitive>::new(
            state.density() * base.powf(gamma.tdgm1()),
            state.velocity() + (v_half - v) * n_unit,
            state.pressure() * base.powf(gamma.gamma() * gamma.tdgm1()),
        )
    }

    fn sample_middle_state(
        rho: f64,
        u: f64,
        p: f64,
        state: &State<Primitive>,
        v: f64,
        n_unit: DVec3,
    ) -> State<Primitive> {
        State::<Primitive>::new(rho, state.velocity() + (u - v) * n_unit, p)
    }

    fn sample_left_rarefaction_wave(
        rho: f64,
        u: f64,
        p: f64,
        left: &State<Primitive>,
        v: f64,
        a: f64,
        n_unit: DVec3,
        gamma: &AdiabaticIndex,
    ) -> State<Primitive> {
        if Self::rarefaction_head_speed(v, a) < 0. {
            if Self::rarefaction_tail_speed(u, a, p / left.pressure(), gamma) > 0. {
                Self::sample_rarefaction_fan(left, a, v, n_unit, gamma)
            } else {
                Self::sample_middle_state(rho, u, p, left, v, n_unit)
            }
        } else {
            *left
        }
    }

    fn sample_right_rarefaction_wave(
        rho: f64,
        u: f64,
        p: f64,
        right: &State<Primitive>,
        v: f64,
        a: f64,
        n_unit: DVec3,
        gamma: &AdiabaticIndex,
    ) -> State<Primitive> {
        if Self::rarefaction_head_speed(v, -a) > 0. {
            if Self::rarefaction_tail_speed(u, -a, p / right.pressure(), gamma) < 0. {
                Self::sample_rarefaction_fan(right, -a, v, n_unit, gamma)
            } else {
                Self::sample_middle_state(rho, u, p, right, v, n_unit)
            }
        } else {
            *right
        }
    }

    fn sample_left_shock_wave(
        rho: f64,
        u: f64,
        p: f64,
        left: &State<Primitive>,
        v: f64,
        a: f64,
        n_unit: DVec3,
        gamma: &AdiabaticIndex,
    ) -> State<Primitive> {
        if Self::shock_speed(v, a, p / left.pressure(), gamma) < 0. {
            Self::sample_middle_state(rho, u, p, left, v, n_unit)
        } else {
            *left
        }
    }

    fn sample_right_shock_wave(
        rho: f64,
        u: f64,
        p: f64,
        right: &State<Primitive>,
        v: f64,
        a: f64,
        n_unit: DVec3,
        gamma: &AdiabaticIndex,
    ) -> State<Primitive> {
        if Self::shock_speed(v, -a, p / right.pressure(), gamma) > 0. {
            Self::sample_middle_state(rho, u, p, right, v, n_unit)
        } else {
            *right
        }
    }

    /// Sample the solution at x/t = 0
    fn sample(
        &self,
        left: &State<Primitive>,
        right: &State<Primitive>,
        v_l: f64,
        v_r: f64,
        a_l: f64,
        a_r: f64,
        n_unit: DVec3,
        gamma: &AdiabaticIndex,
    ) -> State<Primitive> {
        let star = self.solve_for_star_state(left, right, v_l, v_r, a_l, a_r, gamma);
        if star.u < 0. {
            if star.p > right.pressure() {
                Self::sample_right_shock_wave(
                    star.rho_r, star.u, star.p, right, v_r, a_r, n_unit, gamma,
                )
            } else {
                Self::sample_right_rarefaction_wave(
                    star.rho_r, star.u, star.p, right, v_r, a_r, n_unit, gamma,
                )
            }
        } else if star.p > left.pressure() {
            Self::sample_left_shock_wave(
                star.rho_l, star.u, star.p, left, v_l, a_l, n_unit, gamma,
            )
        } else {
            Self::sample_left_rarefaction_wave(
                star.rho_l, star.u, star.p, left, v_l, a_l, n_unit, gamma,
            )
        }
    }
}

impl<T: RiemannStarSolver> RiemannFluxSolver for T {
    fn solve_for_flux(
        &self,
        left: &State<Primitive>,
        right: &State<Primitive>,
        interface_velocity: DVec3,
        n_unit: DVec3,
        eos: &GasLaw,
    ) -> State<Conserved> {

        // Boost to interface frame
        let left = left.boost(-interface_velocity);
        let right = right.boost(-interface_velocity);

        let v_l = left.velocity().dot(n_unit);
        let v_r = right.velocity().dot(n_unit);
        let a_l = eos.sound_speed(left.pressure(), 1. / left.density());
        let a_r = eos.sound_speed(right.pressure(), 1. / right.density());

        // velocity difference
        let v_r_m_v_l = v_r - v_l;

        // handle vacuum
        let w_half =
            if VacuumRiemannSolver::is_vacuum(&left, &right, a_l, a_r, v_r_m_v_l, eos.gamma()) {
                VacuumRiemannSolver.sample(&left, &right, v_l, v_r, a_l, a_r, n_unit, eos.gamma())
            } else {
                // Sample the solution.
                // This corresponds to the flow chart in Fig. 4.14 in Toro
                self.sample(&left, &right, v_l, v_r, a_l, a_r, n_unit, eos.gamma())
            };

        flux_from_half_state(&w_half, interface_velocity, n_unit, eos.gamma())
    }
}
