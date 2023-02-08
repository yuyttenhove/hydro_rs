mod exact;
mod hllc;

use glam::DVec3;

use crate::{
    equation_of_state::EquationOfState,
    errors::ConfigError,
    physical_quantities::{Conserved, Primitives},
};

pub use exact::ExactRiemannSolver;
pub use hllc::HLLCRiemannSolver;

pub trait RiemannSolver: Sync {
    fn solve_for_flux(
        &self,
        left: &Primitives,
        right: &Primitives,
        interface_velocity: DVec3,
        n_unit: DVec3,
        eos: &EquationOfState,
    ) -> Conserved;
}

pub fn get_solver(
    kind: &str,
    eos: &EquationOfState,
) -> Result<Box<dyn RiemannSolver>, ConfigError> {
    let &EquationOfState::Ideal { gamma } = eos else {
        panic!("Only Ideal gasses are supported right now!");
    };
    match kind {
        "HLLC" => Ok(Box::new(HLLCRiemannSolver::new(gamma))),
        "Exact" => Ok(Box::new(ExactRiemannSolver::new(gamma))),
        _ => Err(ConfigError::UnknownRiemannSolver(kind.to_string())),
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

    fn sample_half_vacuum(
        &self,
        non_vacuum: &Primitives,
        v: f64,
        a: f64,
        n_unit: DVec3,
    ) -> Primitives {
        let base = self.tdgp1 + self.gm1dgp1 * v / a;
        let v_half = self.tdgp1 * (a + v / self.tdgm1) - v;
        Primitives::new(
            non_vacuum.density() * base.powf(self.tdgm1),
            non_vacuum.velocity() + n_unit * v_half,
            non_vacuum.pressure() * base.powf(self.gamma * self.tdgm1),
        )
    }

    fn sample_right_vacuum(
        &self,
        left: &Primitives,
        v_l: f64,
        a_l: f64,
        n_unit: DVec3,
    ) -> Primitives {
        if v_l < a_l {
            let s_l = v_l + self.tdgm1 * a_l;
            if s_l > 0. {
                self.sample_half_vacuum(left, v_l, a_l, n_unit)
            } else {
                Primitives::vacuum()
            }
        } else {
            *left
        }
    }

    fn sample_left_vacuum(
        &self,
        right: &Primitives,
        v_r: f64,
        a_r: f64,
        n_unit: DVec3,
    ) -> Primitives {
        if v_r > -a_r {
            let s_r = v_r - self.tdgm1 * a_r;
            if s_r < 0. {
                self.sample_half_vacuum(right, v_r, -a_r, n_unit)
            } else {
                Primitives::vacuum()
            }
        } else {
            *right
        }
    }

    fn sample_vacuum_creation(
        &self,
        left: &Primitives,
        right: &Primitives,
        v_l: f64,
        v_r: f64,
        a_l: f64,
        a_r: f64,
        n_unit: DVec3,
    ) -> Primitives {
        let s_l = v_l - self.tdgm1 * a_l;
        let s_r = v_r - self.tdgm1 * a_r;

        if s_l >= 0. {
            if a_l > v_l {
                self.sample_half_vacuum(left, v_l, a_l, n_unit)
            } else {
                *left
            }
        } else if s_r <= 0. {
            if -a_r < v_r {
                self.sample_half_vacuum(right, v_r, -a_r, n_unit)
            } else {
                *right
            }
        } else {
            debug_assert!(s_r > 0. && s_l < 0.);
            Primitives::vacuum()
        }
    }

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
        if right.density() == 0. {
            self.sample_right_vacuum(left, v_l, a_l, n_unit)
        } else if left.density() == 0. {
            self.sample_left_vacuum(right, v_r, a_r, n_unit)
        } else {
            self.sample_vacuum_creation(left, right, v_l, v_r, a_l, a_r, n_unit)
        }
    }

    fn solve_for_flux(
        &self,
        left: &Primitives,
        right: &Primitives,
        a_l: f64,
        a_r: f64,
        n_unit: DVec3,
        interface_velocity: DVec3,
    ) -> Conserved {
        // Calculate velocities of the right and left states in a frame aligned with the face normal
        let v_l = left.velocity().dot(n_unit);
        let v_r = right.velocity().dot(n_unit);

        debug_assert!(self.is_vacuum(left, right, a_l, a_r, v_r - v_l));

        // Solve for total vacuum
        if left.density() == 0. && right.density() == 0. {
            return Conserved::vacuum();
        }

        // Get primitives at interface
        let half = self.solve(left, right, v_l, v_r, a_l, a_r, n_unit);

        flux_from_half_state(&half, interface_velocity, self.odgm1, n_unit)
    }
}

fn flux_from_half_state(
    half: &Primitives,
    interface_velocity: DVec3,
    one_over_gamma_minus_one: f64,
    n_unit: DVec3,
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

    let roe =
        half.pressure() * one_over_gamma_minus_one + 0.5 * half.density() * v_tot.length_squared();
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

    #[test]
    fn test_vacuum_solver_symmetry() {
        let gamma = 5. / 3.;
        let interface_velocity = 0.15 * DVec3::X;
        let solver = VacuumRiemannSolver::new(gamma);
        let eos = EquationOfState::Ideal { gamma };
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
        let fluxes = solver.solve_for_flux(
            &left.boost(-interface_velocity),
            &right,
            a_l,
            a_r,
            DVec3::X,
            interface_velocity,
        );
        let fluxes_reversed = solver.solve_for_flux(
            &right,
            &left_reversed.boost(interface_velocity),
            a_r,
            a_l,
            DVec3::X,
            -interface_velocity,
        );

        assert_approx_eq!(f64, fluxes.mass(), -fluxes_reversed.mass());
        assert_approx_eq!(f64, fluxes.momentum().x, fluxes_reversed.momentum().x);
        assert_approx_eq!(f64, fluxes.momentum().y, fluxes_reversed.momentum().y);
        assert_approx_eq!(f64, fluxes.momentum().z, fluxes_reversed.momentum().z);
        assert_approx_eq!(f64, fluxes.energy(), -fluxes_reversed.energy());
    }

    #[test]
    fn test_vacuum_solver() {
        let gamma = 5. / 3.;
        let interface_velocity = 0.15 * DVec3::X;
        let solver = VacuumRiemannSolver::new(gamma);
        let eos = EquationOfState::Ideal { gamma };
        let left = Primitives::new(1., 0.3 * DVec3::X, 0.5);
        let right = Primitives::new(0., DVec3::ZERO, 0.);
        let a_l = eos.sound_speed(left.pressure(), 1. / left.density());
        let a_r = eos.sound_speed(right.pressure(), 1. / right.density());
        let fluxes = solver.solve_for_flux(
            &left.boost(-interface_velocity),
            &right,
            a_l,
            a_r,
            DVec3::X,
            interface_velocity,
        );

        // Reference solution calculated in lab frame
        let half_s = Primitives::new(
            0.57625934,
            0.7596532 * DVec3::X - interface_velocity,
            0.19952622,
        );
        let flux_s = flux_from_half_state(&half_s, interface_velocity, solver.odgm1, DVec3::X);

        assert_approx_eq!(f64, fluxes.mass(), flux_s.mass());
        assert_approx_eq!(f64, fluxes.momentum().x, flux_s.momentum().x);
        assert_approx_eq!(f64, fluxes.momentum().y, flux_s.momentum().y);
        assert_approx_eq!(f64, fluxes.momentum().z, flux_s.momentum().z);
        assert_approx_eq!(f64, fluxes.energy(), flux_s.energy());
    }
}
