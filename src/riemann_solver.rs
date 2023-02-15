mod airs;
mod exact;
mod hllc;
mod pvrs;
mod vacuum;

use glam::DVec3;
use yaml_rust::Yaml;

use crate::{
    equation_of_state::EquationOfState,
    errors::ConfigError,
    physical_quantities::{Conserved, Primitives},
};

pub use airs::AIRiemannSolver;
pub use exact::ExactRiemannSolver;
pub use hllc::HLLCRiemannSolver;
pub use pvrs::PVRiemannSolver;

use self::vacuum::VacuumRiemannSolver;

pub fn get_solver(cfg: &Yaml) -> Result<Box<dyn RiemannFluxSolver>, ConfigError> {
    let kind = cfg["solver"]
        .as_str()
        .ok_or(ConfigError::MissingParameter("solver:kind".to_string()))?;
    match kind {
        "HLLC" => Ok(Box::new(HLLCRiemannSolver)),
        "Exact" => Ok(Box::new(ExactRiemannSolver)),
        "PVRS" => Ok(Box::new(PVRiemannSolver)),
        "AIRS" => {
            let threshold = cfg["threshold"]
                .as_f64()
                .ok_or(ConfigError::MissingParameter(
                    "solver:threshold".to_string(),
                ))?;
            Ok(Box::new(AIRiemannSolver::new(threshold)))
        }
        _ => Err(ConfigError::UnknownRiemannSolver(kind.to_string())),
    }
}

fn flux_from_half_state(
    half: &Primitives,
    interface_velocity: DVec3,
    n_unit: DVec3,
    eos: &EquationOfState,
) -> Conserved {
    let v_tot = interface_velocity + half.velocity();
    let mut flux = [DVec3::ZERO; 5];

    flux[0] = half.density() * half.velocity();

    flux[1] = half.density() * v_tot.x * half.velocity();
    flux[1].x += half.pressure();
    flux[2] = half.density() * v_tot.y * half.velocity();
    flux[2].y += half.pressure();
    flux[3] = half.density() * v_tot.z * half.velocity();
    flux[3].z += half.pressure();

    let roe = half.pressure() * eos.odgm1() + 0.5 * half.density() * v_tot.length_squared();
    flux[4] = roe * half.velocity() + half.pressure() * v_tot;

    Conserved::new(
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
        left: &Primitives,
        right: &Primitives,
        interface_velocity: DVec3,
        n_unit: DVec3,
        eos: &EquationOfState,
    ) -> Conserved;
}

#[derive(Default)]
struct RiemannStarValues {
    rho_l: f64,
    rho_r: f64,
    u: f64,
    p: f64,
}
trait RiemannStarSolver: Sync {
    fn solve_for_star_state(
        &self,
        left: &Primitives,
        right: &Primitives,
        v_l: f64,
        v_r: f64,
        a_l: f64,
        a_r: f64,
        eos: &EquationOfState,
    ) -> RiemannStarValues;

    fn shock_speed(v: f64, a: f64, pdps: f64, eos: &EquationOfState) -> f64 {
        v - a * (0.5 * eos.gp1dg() * pdps + eos.gm1d2g()).sqrt()
    }

    fn rarefaction_head_speed(v: f64, a: f64) -> f64 {
        v - a
    }

    fn rarefaction_tail_speed(u: f64, a: f64, pdps: f64, eos: &EquationOfState) -> f64 {
        u - a * pdps.powf(eos.gm1d2g())
    }

    fn sample_rarefaction_fan(
        state: &Primitives,
        a: f64,
        v: f64,
        n_unit: DVec3,
        eos: &EquationOfState,
    ) -> Primitives {
        let v_half = eos.tdgp1() * (a + 0.5 * (eos.gamma() - 1.) * v);
        let base = eos.tdgp1() + eos.gm1dgp1() / a * v;
        Primitives::new(
            state.density() * base.powf(eos.tdgm1()),
            state.velocity() + (v_half - v) * n_unit,
            state.pressure() * base.powf(eos.gamma() * eos.tdgm1()),
        )
    }

    fn sample_middle_state(
        rho: f64,
        u: f64,
        p: f64,
        state: &Primitives,
        v: f64,
        n_unit: DVec3,
    ) -> Primitives {
        Primitives::new(rho, state.velocity() + (u - v) * n_unit, p)
    }

    fn sample_left_rarefaction_wave(
        rho: f64,
        u: f64,
        p: f64,
        left: &Primitives,
        v: f64,
        a: f64,
        n_unit: DVec3,
        eos: &EquationOfState,
    ) -> Primitives {
        if Self::rarefaction_head_speed(v, a) < 0. {
            if Self::rarefaction_tail_speed(u, a, p / left.pressure(), eos) > 0. {
                Self::sample_rarefaction_fan(left, a, v, n_unit, eos)
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
        right: &Primitives,
        v: f64,
        a: f64,
        n_unit: DVec3,
        eos: &EquationOfState,
    ) -> Primitives {
        if Self::rarefaction_head_speed(v, -a) > 0. {
            if Self::rarefaction_tail_speed(u, -a, p / right.pressure(), eos) < 0. {
                Self::sample_rarefaction_fan(right, -a, v, n_unit, eos)
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
        left: &Primitives,
        v: f64,
        a: f64,
        n_unit: DVec3,
        eos: &EquationOfState,
    ) -> Primitives {
        if Self::shock_speed(v, a, p / left.pressure(), eos) < 0. {
            Self::sample_middle_state(rho, u, p, left, v, n_unit)
        } else {
            *left
        }
    }

    fn sample_right_shock_wave(
        rho: f64,
        u: f64,
        p: f64,
        right: &Primitives,
        v: f64,
        a: f64,
        n_unit: DVec3,
        eos: &EquationOfState,
    ) -> Primitives {
        if Self::shock_speed(v, -a, p / right.pressure(), eos) > 0. {
            Self::sample_middle_state(rho, u, p, right, v, n_unit)
        } else {
            *right
        }
    }

    /// Sample the solution at x/t = 0
    fn sample(
        star: &RiemannStarValues,
        left: &Primitives,
        right: &Primitives,
        v_l: f64,
        v_r: f64,
        a_l: f64,
        a_r: f64,
        n_unit: DVec3,
        eos: &EquationOfState,
    ) -> Primitives {
        if star.u < 0. {
            if star.p > right.pressure() {
                Self::sample_right_shock_wave(
                    star.rho_r, star.u, star.p, right, v_r, a_r, n_unit, eos,
                )
            } else {
                Self::sample_right_rarefaction_wave(
                    star.rho_r, star.u, star.p, right, v_r, a_r, n_unit, eos,
                )
            }
        } else {
            if star.p > left.pressure() {
                Self::sample_left_shock_wave(
                    star.rho_l, star.u, star.p, left, v_l, a_l, n_unit, eos,
                )
            } else {
                Self::sample_left_rarefaction_wave(
                    star.rho_l, star.u, star.p, left, v_l, a_l, n_unit, eos,
                )
            }
        }
    }
}

impl<T: RiemannStarSolver> RiemannFluxSolver for T {
    fn solve_for_flux(
        &self,
        left: &Primitives,
        right: &Primitives,
        interface_velocity: DVec3,
        n_unit: DVec3,
        eos: &EquationOfState,
    ) -> Conserved {
        let v_l = left.velocity().dot(n_unit);
        let v_r = right.velocity().dot(n_unit);
        let a_l = eos.sound_speed(left.pressure(), 1. / left.density());
        let a_r = eos.sound_speed(right.pressure(), 1. / right.density());

        // velocity difference
        let v_r_m_v_l = v_r - v_l;

        // handle vacuum
        let w_half = if VacuumRiemannSolver::is_vacuum(left, right, a_l, a_r, v_r_m_v_l, eos) {
            VacuumRiemannSolver::sample(
                &RiemannStarValues::default(),
                left,
                right,
                v_l,
                v_r,
                a_l,
                a_r,
                n_unit,
                eos,
            )
        } else {
            let star = self.solve_for_star_state(left, right, v_l, v_r, a_l, a_r, eos);
            // Sample the solution.
            // This corresponds to the flow chart in Fig. 4.14 in Toro
            T::sample(&star, left, right, v_l, v_r, a_l, a_r, n_unit, eos)
        };

        flux_from_half_state(&w_half, interface_velocity, n_unit, eos)
    }
}
