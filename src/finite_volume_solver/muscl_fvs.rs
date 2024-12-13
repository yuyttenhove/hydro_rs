use crate::{
    gas_law::GasLaw, gradients::pairwise_limiter, part::Particle, physical_quantities::State,
    riemann_solver::RiemannFluxSolver, Boundary,
};

use super::{FiniteVolumeSolver, FluxInfo};

use crate::physical_quantities::Primitive;
use glam::DVec3;
use meshless_voronoi::VoronoiFace;
use rayon::prelude::*;

pub struct MusclFvs<R: RiemannFluxSolver> {
    riemann_solver: R,
    cfl: f64,
    gas_law: GasLaw,
    tvd: bool,
}

impl<R: RiemannFluxSolver> MusclFvs<R> {
    pub fn new(riemann_solver: R, cfl: f64, gas_law: GasLaw, tvd: bool) -> Self {
        Self {
            riemann_solver,
            cfl,
            gas_law,
            tvd,
        }
    }
}

impl<R: RiemannFluxSolver> FiniteVolumeSolver for MusclFvs<R> {
    fn compute_fluxes(
        &self,
        faces: &[meshless_voronoi::VoronoiFace],
        particles: &[Particle],
        part_is_active: &[bool],
        boundary: Boundary,
    ) -> Vec<FluxInfo> {
        faces
            .par_iter()
            .map(|face| {
                let left = &particles[face.left()];
                let left_active = part_is_active[face.left()];
                match face.right() {
                    Some(right_idx) => {
                        let right = &particles[right_idx];
                        let right_active = part_is_active[right_idx];
                        // Do the flux exchange only when at least one particle is active *and* the particle with the strictly smallest timestep is active
                        if (!left_active && !right_active)
                            || (right.dt < left.dt && !right_active)
                            || (left.dt < right.dt && !left_active)
                        {
                            return FluxInfo::zero();
                        }
                        let dt = left.dt.min(right.dt);
                        flux_exchange(
                            left,
                            right,
                            dt,
                            face,
                            0.5,
                            self.do_gradients_limit(),
                            &self.gas_law,
                            &self.riemann_solver,
                        )
                    }
                    None => {
                        if left_active {
                            flux_exchange_boundary(
                                left,
                                face,
                                boundary,
                                0.5,
                                self.do_gradients_limit(),
                                &self.gas_law,
                                &self.riemann_solver,
                            )
                        } else {
                            FluxInfo::zero()
                        }
                    }
                }
            })
            .collect()
    }

    fn eos(&self) -> &GasLaw {
        &self.gas_law
    }

    fn cfl(&self) -> f64 {
        self.cfl
    }

    fn do_gradients(&self) -> bool {
        true
    }

    fn do_gradients_limit(&self) -> bool {
        self.tvd
    }
}

fn flux_exchange<RiemannSolver: RiemannFluxSolver>(
    left: &Particle,
    right: &Particle,
    dt: f64,
    face: &VoronoiFace,
    time_extrapolate_fac: f64,
    do_gradients_limit: bool,
    eos: &GasLaw,
    riemann_solver: &RiemannSolver,
) -> FluxInfo {
    // We extrapolate from the centroid of the particles.
    let dx_left = face.centroid() - left.gradients_centroid;
    let shift = face.shift().unwrap_or(DVec3::ZERO);
    let dx_right = face.centroid() - right.gradients_centroid - shift;
    let dx_gradients = (dx_left - dx_right).length();
    let dx_centroid = right.centroid + shift - left.centroid;
    let dx = right.loc + shift - left.loc;

    // Calculate the maximal signal velocity (used for timestep computation)
    let mut v_max = 0.;
    if left.primitives.density() > 0. {
        v_max += eos.sound_speed(left.primitives.pressure(), 1. / left.primitives.density());
    }
    if right.primitives.density() > 0. {
        v_max += eos.sound_speed(right.primitives.pressure(), 1. / right.primitives.density());
    }
    v_max -= (right.primitives.velocity() - left.primitives.velocity())
        .dot(dx_centroid)
        .min(0.);

    // Gradient + time extrapolation + pair wise limiting
    let dt_extrapolate = -time_extrapolate_fac * dt;
    let left_dash = left.primitives
        + left.gradients.dot(dx_left)
        + left.extrapolations
        + left.time_extrapolations(dt_extrapolate, eos);
    let right_dash = right.primitives
        + right.gradients.dot(dx_right)
        + right.extrapolations
        + right.time_extrapolations(dt_extrapolate, eos);
    let (primitives_left, primitives_right) = if do_gradients_limit {
        (
            pairwise_limiter(
                &left.primitives,
                &right.primitives,
                &left_dash,
                dx_left.length() / dx_gradients,
            ),
            pairwise_limiter(
                &right.primitives,
                &left.primitives,
                &right_dash,
                dx_right.length() / dx_gradients,
            ),
        )
    } else {
        (
            left_dash.pairwise_max(&State::<Primitive>::new(
                0.,
                DVec3::splat(f64::NEG_INFINITY),
                0.,
            )),
            right_dash.pairwise_max(&State::<Primitive>::new(
                0.,
                DVec3::splat(f64::NEG_INFINITY),
                0.,
            )),
        )
    };

    // Compute interface velocity (Springel (2010), eq. 33):
    let midpoint = 0.5 * (left.loc + right.loc + shift);
    let fac = (right.v - left.v).dot(face.centroid() - midpoint) / dx.length_squared();
    let v_face = 0.5 * (left.v + right.v) - fac * dx;

    // Calculate fluxes
    let fluxes = face.area()
        * riemann_solver.solve_for_flux(
            &primitives_left,
            &primitives_right,
            v_face,
            face.normal(),
            eos,
        );

    debug_assert!(fluxes.mass().is_finite());
    debug_assert!(fluxes.momentum().is_finite());
    debug_assert!(fluxes.energy().is_finite());

    FluxInfo {
        fluxes: dt * fluxes,
        mflux: dx * fluxes.mass(),
        v_max,
        a_over_r: face.area() / dx.length(),
    }
}

fn flux_exchange_boundary<RiemannSolver: RiemannFluxSolver>(
    part: &Particle,
    face: &VoronoiFace,
    boundary: Boundary,
    time_extrapolate_fac: f64,
    do_gradients_limit: bool,
    eos: &GasLaw,
    riemann_solver: &RiemannSolver,
) -> FluxInfo {
    // get reflected particle
    let mut reflected = part.reflect(face.centroid(), face.normal());
    let dx_face = face.centroid() - part.gradients_centroid;
    let dx_gradients = (reflected.gradients_centroid - part.gradients_centroid).length();
    let dx_centroid = reflected.centroid - part.centroid;
    let dx = reflected.loc - part.loc;

    // Calculate the maximal signal velocity (used for timestep computation)
    let mut v_max = 0.;
    if part.primitives.density() > 0. {
        v_max += eos.sound_speed(part.primitives.pressure(), 1. / part.primitives.density());
    }

    // Gradient extrapolation
    let dt_extraplotate = -time_extrapolate_fac * part.dt;
    let mut primitives_dash = part.primitives
        + part.gradients.dot(dx_face)
        + part.extrapolations
        + part.time_extrapolations(dt_extraplotate, eos);

    let primitives_out = match boundary {
        Boundary::Reflective => {
            // Also reflect velocity
            reflected = reflected.reflect_quantities(face.normal());
            v_max -= (reflected.primitives.velocity() - part.primitives.velocity())
                .dot(dx_centroid)
                .min(0.);
            reflected.primitives
        }
        Boundary::Open => part.primitives,
        Boundary::Vacuum => {
            v_max += part.primitives.velocity().dot(dx_centroid).min(0.);
            State::vacuum()
        }
        Boundary::Periodic => {
            unreachable!("This function should not be called with periodic boundary conditions!");
        }
    };

    // Gradient limiting
    if do_gradients_limit {
        primitives_dash = pairwise_limiter(
            &part.primitives,
            &primitives_out,
            &primitives_dash,
            dx_face.length() / dx_gradients,
        );
    }

    let primitives_boundary = match boundary {
        Boundary::Reflective => primitives_dash.reflect(face.normal()),
        Boundary::Open => primitives_dash,
        Boundary::Vacuum => State::vacuum(),
        Boundary::Periodic => {
            unreachable!("This function should not be called with periodic boundary conditions!");
        }
    };

    // Solve for flux
    let fluxes = face.area()
        * riemann_solver.solve_for_flux(
            &primitives_dash,
            &primitives_boundary,
            DVec3::ZERO,
            face.normal(),
            eos,
        );

    debug_assert!(fluxes.mass().is_finite());
    debug_assert!(fluxes.momentum().is_finite());
    debug_assert!(fluxes.energy().is_finite());

    FluxInfo {
        fluxes: part.dt * fluxes,
        mflux: dx * fluxes.mass(),
        v_max,
        a_over_r: face.area() / dx.length(),
    }
}
