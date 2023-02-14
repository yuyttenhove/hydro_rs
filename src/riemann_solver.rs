mod airs;
mod exact;
mod hllc;
mod pvrs;

use glam::DVec3;
use yaml_rust::{Yaml, YamlLoader};

use crate::{
    equation_of_state::EquationOfState,
    errors::ConfigError,
    physical_quantities::{Conserved, Primitives},
};

pub use airs::AIRiemannSolver;
pub use exact::ExactRiemannSolver;
pub use hllc::HLLCRiemannSolver;
pub use pvrs::PVRiemannSolver;

pub fn get_solver(
    cfg: &Yaml,
    eos: &EquationOfState,
) -> Result<Box<dyn RiemannFluxSolver>, ConfigError> {
    let &EquationOfState::Ideal { gamma, .. } = eos else {
        panic!("Only Ideal gasses are supported right now!");
    };
    let kind = cfg["kind"]
        .as_str()
        .ok_or(ConfigError::MissingParameter("solver:kind".to_string()))?;
    match kind {
        "HLLC" => Ok(Box::new(HLLCRiemannSolver::new(gamma))),
        "Exact" => Ok(Box::new(ExactRiemannSolver::new(gamma))),
        "PVRS" => Ok(Box::new(PVRiemannSolver)),
        "AIRS" => {
            let threshold = cfg["threshold"]
                .as_f64()
                .ok_or(ConfigError::MissingParameter(
                    "solver:threshold".to_string(),
                ))?;
            Ok(Box::new(AIRiemannSolver::new(gamma, threshold)))
        }
        _ => Err(ConfigError::UnknownRiemannSolver(kind.to_string())),
    }
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

    fn shock_speed(&self, v: f64, a: f64, pdps: f64, eos: &EquationOfState) -> f64 {
        v - a * (0.5 * eos.gp1dg() * pdps + eos.gm1d2g()).sqrt()
    }

    fn rarefaction_head_speed(&self, v: f64, a: f64) -> f64 {
        v - a
    }

    fn rarefaction_tail_speed(&self, u: f64, a: f64, pdps: f64, eos: &EquationOfState) -> f64 {
        u - a * pdps.powf(eos.gm1d2g())
    }

    fn sample_rarefaction_fan(
        &self,
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
        &self,
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
        &self,
        rho: f64,
        u: f64,
        p: f64,
        left: &Primitives,
        v: f64,
        a: f64,
        n_unit: DVec3,
        eos: &EquationOfState,
    ) -> Primitives {
        if self.rarefaction_head_speed(v, a) < 0. {
            if self.rarefaction_tail_speed(u, a, p / left.pressure(), eos) > 0. {
                self.sample_rarefaction_fan(left, a, v, n_unit, eos)
            } else {
                self.sample_middle_state(rho, u, p, left, v, n_unit)
            }
        } else {
            *left
        }
    }

    fn sample_right_rarefaction_wave(
        &self,
        rho: f64,
        u: f64,
        p: f64,
        right: &Primitives,
        v: f64,
        a: f64,
        n_unit: DVec3,
        eos: &EquationOfState,
    ) -> Primitives {
        if self.rarefaction_head_speed(v, -a) > 0. {
            if self.rarefaction_tail_speed(u, -a, p / right.pressure(), eos) < 0. {
                self.sample_rarefaction_fan(right, -a, v, n_unit, eos)
            } else {
                self.sample_middle_state(rho, u, p, right, v, n_unit)
            }
        } else {
            *right
        }
    }

    fn sample_left_shock_wave(
        &self,
        rho: f64,
        u: f64,
        p: f64,
        left: &Primitives,
        v: f64,
        a: f64,
        n_unit: DVec3,
        eos: &EquationOfState,
    ) -> Primitives {
        if self.shock_speed(v, a, p / left.pressure(), eos) < 0. {
            self.sample_middle_state(rho, u, p, left, v, n_unit)
        } else {
            *left
        }
    }

    fn sample_right_shock_wave(
        &self,
        rho: f64,
        u: f64,
        p: f64,
        right: &Primitives,
        v: f64,
        a: f64,
        n_unit: DVec3,
        eos: &EquationOfState,
    ) -> Primitives {
        if self.shock_speed(v, -a, p / right.pressure(), eos) > 0. {
            self.sample_middle_state(rho, u, p, right, v, n_unit)
        } else {
            *right
        }
    }

    /// Sample the solution at x/t = 0
    fn sample(
        &self,
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
                self.sample_right_shock_wave(
                    star.rho_r, star.u, star.p, right, v_r, a_r, n_unit, eos,
                )
            } else {
                self.sample_right_rarefaction_wave(
                    star.rho_r, star.u, star.p, right, v_r, a_r, n_unit, eos,
                )
            }
        } else {
            if star.p > left.pressure() {
                self.sample_left_shock_wave(star.rho_l, star.u, star.p, left, v_l, a_l, n_unit, eos)
            } else {
                self.sample_left_rarefaction_wave(
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
        if VacuumRiemannSolver::is_vacuum(left, right, a_l, a_r, v_r_m_v_l, eos) {
            return VacuumRiemannSolver::solve_for_flux(
                left,
                right,
                a_l,
                a_r,
                n_unit,
                interface_velocity,
                eos,
            );
        }

        let star = self.solve_for_star_state(left, right, v_l, v_r, a_l, a_r, eos);

        // Sample the solution.
        // This corresponds to the flow chart in Fig. 4.14 in Toro
        let w_half = self.sample(&star, left, right, v_l, v_r, a_l, a_r, n_unit, eos);
        flux_from_half_state(&w_half, interface_velocity, n_unit, eos)
    }
}

/// Exact Vacuum Riemann solver.
struct VacuumRiemannSolver;

impl VacuumRiemannSolver {
    fn is_vacuum(
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

    fn solve(
        left: &Primitives,
        right: &Primitives,
        v_l: f64,
        v_r: f64,
        a_l: f64,
        a_r: f64,
        n_unit: DVec3,
        eos: &EquationOfState,
    ) -> Primitives {
        if right.density() == 0. {
            Self::sample_right_vacuum(left, v_l, a_l, n_unit, eos)
        } else if left.density() == 0. {
            Self::sample_left_vacuum(right, v_r, a_r, n_unit, eos)
        } else {
            Self::sample_vacuum_creation(left, right, v_l, v_r, a_l, a_r, n_unit, eos)
        }
    }

    fn solve_for_flux(
        left: &Primitives,
        right: &Primitives,
        a_l: f64,
        a_r: f64,
        n_unit: DVec3,
        interface_velocity: DVec3,
        eos: &EquationOfState,
    ) -> Conserved {
        // Calculate velocities of the right and left states in a frame aligned with the face normal
        let v_l = left.velocity().dot(n_unit);
        let v_r = right.velocity().dot(n_unit);

        debug_assert!(Self::is_vacuum(left, right, a_l, a_r, v_r - v_l, eos));

        // Solve for total vacuum
        if left.density() == 0. && right.density() == 0. {
            return Conserved::vacuum();
        }

        // Get primitives at interface
        let half = Self::solve(left, right, v_l, v_r, a_l, a_r, n_unit, eos);

        let eos = EquationOfState::new(&YamlLoader::load_from_str("gamma: 1.666667").unwrap()[0])
            .unwrap();
        flux_from_half_state(&half, interface_velocity, n_unit, &eos)
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

#[cfg(test)]
mod tests {
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
        let a_l = eos.sound_speed(left.pressure(), 1. / left.density());
        let a_r = eos.sound_speed(right.pressure(), 1. / right.density());
        let fluxes = VacuumRiemannSolver::solve_for_flux(
            &left.boost(-interface_velocity),
            &right,
            a_l,
            a_r,
            DVec3::X,
            interface_velocity,
            &eos,
        );
        let fluxes_reversed = VacuumRiemannSolver::solve_for_flux(
            &right,
            &left_reversed.boost(interface_velocity),
            a_r,
            a_l,
            DVec3::X,
            -interface_velocity,
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
        let a_l = eos.sound_speed(left.pressure(), 1. / left.density());
        let a_r = eos.sound_speed(right.pressure(), 1. / right.density());
        let fluxes = VacuumRiemannSolver::solve_for_flux(
            &left.boost(-interface_velocity),
            &right,
            a_l,
            a_r,
            DVec3::X,
            interface_velocity,
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
