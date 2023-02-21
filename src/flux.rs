use crate::engine::Engine;
use crate::equation_of_state::EquationOfState;
use crate::gradients::pairwise_limiter;
use crate::part::Particle;
use crate::physical_quantities::{Conserved, Primitives};
use crate::space::Boundary;
use glam::DVec3;
use meshless_voronoi::VoronoiFace;

pub struct FluxInfo {
    pub fluxes: Conserved,
    pub mflux: DVec3,
    pub v_max: f64,
}

pub fn flux_exchange(
    left: &Particle,
    right: &Particle,
    dt: f64,
    face: &VoronoiFace,
    eos: &EquationOfState,
    engine: &Engine,
) -> FluxInfo {
    // We extrapolate from the centroid of the particles.
    let dx_left = face.centroid() - left.centroid;
    let shift = face.shift().unwrap_or(DVec3::ZERO);
    let dx_right = face.centroid() - right.centroid - shift;
    let dx_centroid = right.centroid + shift - left.centroid;
    let dx = right.x + shift - left.x;

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

    // Gradient extrapolation
    let primitives_left = pairwise_limiter(
        &left.primitives,
        &right.primitives,
        &(left.primitives + left.gradients.dot(dx_left).into() + left.extrapolations),
        dx_left.length() / dx_centroid.length(),
    );
    let primitives_right = pairwise_limiter(
        &right.primitives,
        &left.primitives,
        &(right.primitives + right.gradients.dot(dx_right).into() + right.extrapolations),
        dx_right.length() / dx_centroid.length(),
    );

    // Compute interface velocity (Springel (2010), eq. 33):
    let midpoint = 0.5 * (left.x + right.x + shift);
    let fac = (right.v - left.v).dot(face.centroid() - midpoint) / dx.length_squared();
    let v_face = 0.5 * (left.v + right.v) - fac * dx;

    // Calculate fluxes
    let fluxes = face.area()
        * engine.hydro_solver.solve_for_flux(
            &primitives_left.boost(-v_face),
            &primitives_right.boost(-v_face),
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
    }
}

pub fn flux_exchange_boundary(
    part: &Particle,
    face: &VoronoiFace,
    boundary: Boundary,
    eos: &EquationOfState,
    engine: &Engine,
) -> FluxInfo {
    // get reflected particle
    let mut reflected = part.reflect(face.centroid(), face.normal());
    let dx_face = face.centroid() - part.centroid;
    let dx_centroid = reflected.centroid - part.centroid;
    let dx = reflected.x - part.x;

    // Calculate the maximal signal velocity (used for timestep computation)
    let mut v_max = 0.;
    if part.primitives.density() > 0. {
        v_max += eos.sound_speed(part.primitives.pressure(), 1. / part.primitives.density());
    }

    // Gradient extrapolation
    let mut primitives_dash =
        part.primitives + part.gradients.dot(dx_face).into() + part.extrapolations;

    let primitives_boundary = match boundary {
        Boundary::Reflective => {
            // Also reflect velocity
            reflected = reflected.reflect_quantities(face.normal());

            // Apply pairwise limiter
            primitives_dash = pairwise_limiter(
                &part.primitives,
                &reflected.primitives,
                &primitives_dash,
                dx_face.length() / dx_centroid.length(),
            );
            v_max -= (reflected.primitives.velocity() - part.primitives.velocity())
                .dot(dx_centroid)
                .min(0.);

            primitives_dash.reflect(face.normal())
        }
        Boundary::Open => {
            primitives_dash = pairwise_limiter(
                &part.primitives,
                &part.primitives,
                &primitives_dash,
                dx_face.length() / dx_centroid.length(),
            );

            primitives_dash
        }
        Boundary::Vacuum => {
            let vacuum = Primitives::vacuum();
            primitives_dash = pairwise_limiter(
                &part.primitives,
                &vacuum,
                &primitives_dash,
                dx_face.length() / dx_centroid.length(),
            );
            v_max += part.primitives.velocity().dot(dx_centroid).min(0.);

            vacuum
        }
        Boundary::Periodic => {
            unreachable!("This function should not be called with periodic boundary conditions!");
        }
    };

    // Solve for flux
    let fluxes = face.area()
        * engine.hydro_solver.solve_for_flux(
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
    }
}
