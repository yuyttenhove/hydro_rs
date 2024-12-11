use crate::{
    gas_law::GasLaw, part::Particle, physical_quantities::State,
    riemann_solver::RiemannWafFluxSolver, Boundary,
};

use super::{FiniteVolumeSolver, FluxInfo};

use glam::DVec3;
use meshless_voronoi::VoronoiFace;
use rayon::prelude::*;

pub struct WafFvs<R: RiemannWafFluxSolver> {
    riemann_solver: R,
    cfl: f64,
    gas_law: GasLaw,
}

impl<R: RiemannWafFluxSolver> WafFvs<R> {
    pub fn new(riemann_solver: R, cfl: f64, eos: GasLaw) -> Self {
        Self {
            gas_law: eos,
            cfl,
            riemann_solver,
        }
    }
}

impl<R: RiemannWafFluxSolver> FiniteVolumeSolver for WafFvs<R> {
    fn compute_fluxes(
        &self,
        faces: &[meshless_voronoi::VoronoiFace],
        particles: &[crate::part::Particle],
        part_is_active: &[bool],
        boundary: crate::Boundary,
    ) -> Vec<super::FluxInfo> {
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
                        flux_exchange(left, right, dt, face, &self.gas_law, &self.riemann_solver)
                    }
                    None => {
                        if left_active {
                            flux_exchange_boundary(
                                left,
                                face,
                                boundary,
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
}

fn flux_exchange<RiemannSolver: RiemannWafFluxSolver>(
    left: &Particle,
    right: &Particle,
    dt: f64,
    face: &VoronoiFace,
    eos: &GasLaw,
    riemann_solver: &RiemannSolver,
) -> FluxInfo {
    let shift = face.shift().unwrap_or(DVec3::ZERO);
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

    // Compute interface velocity (Springel (2010), eq. 33):
    let midpoint = 0.5 * (left.loc + right.loc + shift);
    let fac = (right.v - left.v).dot(face.centroid() - midpoint) / dx.length_squared();
    let v_face = 0.5 * (left.v + right.v) - fac * dx;

    // Extrapolate back to midpoint of the timestep over which the fluxes are exchanged
    let dt_extrapolate = -0.5 * dt;
    let left_primitives = left.primitives - left.time_extrapolations(dt_extrapolate, eos);
    let right_primitives = right.primitives - right.time_extrapolations(dt_extrapolate, eos);

    // Terms for flux limiters
    let n_unit = face.normal();
    let dx_left = face.centroid() - left.centroid;
    let dx_right = right.centroid - face.centroid();
    let drho_left = left
        .gradients
        .dot(2. * dx_left.dot(n_unit) * n_unit)
        .density();
    let drho_right = right
        .gradients
        .dot(2. * dx_right.dot(n_unit) * n_unit)
        .density();

    // Calculate fluxes
    let fluxes = face.area()
        * riemann_solver.solve_for_waf_flux(
            &left_primitives,
            &right_primitives,
            dx_left,
            dx_right,
            drho_left,
            drho_right,
            v_face,
            dt,
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

fn flux_exchange_boundary<RiemannSolver: RiemannWafFluxSolver>(
    part: &Particle,
    face: &VoronoiFace,
    boundary: Boundary,
    eos: &GasLaw,
    riemann_solver: &RiemannSolver,
) -> FluxInfo {
    // get reflected particle
    let mut reflected = part.reflect(face.centroid(), face.normal());
    let dx_centroid = reflected.centroid - part.centroid;
    let dx = reflected.loc - part.loc;

    // Calculate the maximal signal velocity (used for timestep computation)
    let mut v_max = 0.;
    if part.primitives.density() > 0. {
        v_max += eos.sound_speed(part.primitives.pressure(), 1. / part.primitives.density());
    }

    let primitives = part.primitives + part.time_extrapolations(-0.5 * part.dt, eos);
    let primitives_boundary = match boundary {
        Boundary::Reflective => {
            // Also reflect velocity
            reflected = reflected.reflect_quantities(face.normal());

            v_max -= (reflected.primitives.velocity() - part.primitives.velocity())
                .dot(dx_centroid)
                .min(0.);

            primitives.reflect(face.normal())
        }
        Boundary::Open => primitives,
        Boundary::Vacuum => {
            let vacuum = State::vacuum();
            v_max += primitives.velocity().dot(dx_centroid).min(0.);
            vacuum
        }
        Boundary::Periodic => {
            unreachable!("This function should not be called with periodic boundary conditions!");
        }
    };

    // Terms for flux limiters
    let n_unit = face.normal();
    let dx_left = face.centroid() - part.centroid;
    let dx_right = dx_left;
    let drho_left = part
        .gradients
        .dot(2. * dx_left.dot(n_unit) * n_unit)
        .density();
    let drho_right = reflected
        .gradients
        .dot(2. * dx_right.dot(n_unit) * n_unit)
        .density();

    // Solve for flux
    let fluxes = face.area()
        * riemann_solver.solve_for_waf_flux(
            &primitives,
            &primitives_boundary,
            dx_left,
            dx_right,
            drho_left,
            drho_right,
            DVec3::ZERO,
            part.dt,
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
