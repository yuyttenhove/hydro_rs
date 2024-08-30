use crate::engine::Engine;
use crate::equation_of_state::EquationOfState;
use crate::gradients::pairwise_limiter;
use crate::part::Particle;
use crate::physical_quantities::{Conserved, State};
use crate::space::Boundary;
use glam::DVec3;
use meshless_voronoi::VoronoiFace;

pub struct FluxInfo {
    pub fluxes: State<Conserved>,
    pub mflux: DVec3,
    pub v_max: f64,
    pub a_over_r: f64,
}

pub fn flux_exchange(
    left: &Particle,
    right: &Particle,
    dt: f64,
    face: &VoronoiFace,
    time_extrapolate_fac: f64,
    eos: &EquationOfState,
    engine: &Engine,
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
        + left.gradients.dot(dx_left).into()
        + left.extrapolations
        + left.time_extrapolations(dt_extrapolate, eos);
    let right_dash = right.primitives
        + right.gradients.dot(dx_right).into()
        + right.extrapolations
        + right.time_extrapolations(dt_extrapolate, eos);
    let primitives_left = pairwise_limiter(
        &left.primitives,
        &right.primitives,
        &left_dash,
        dx_left.length() / dx_gradients,
    );
    let primitives_right = pairwise_limiter(
        &right.primitives,
        &left.primitives,
        &right_dash,
        dx_right.length() / dx_gradients,
    );

    // Compute interface velocity (Springel (2010), eq. 33):
    let midpoint = 0.5 * (left.loc + right.loc + shift);
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
        a_over_r: face.area() / dx.length(),
    }
}

pub fn flux_exchange_boundary(
    part: &Particle,
    face: &VoronoiFace,
    boundary: Boundary,
    time_extrapolate_fac: f64,
    eos: &EquationOfState,
    engine: &Engine,
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
        + part.gradients.dot(dx_face).into()
        + part.extrapolations
        + part.time_extrapolations(dt_extraplotate, eos);

    let primitives_boundary = match boundary {
        Boundary::Reflective => {
            // Also reflect velocity
            reflected = reflected.reflect_quantities(face.normal());

            // Apply pairwise limiter
            primitives_dash = pairwise_limiter(
                &part.primitives,
                &reflected.primitives,
                &primitives_dash,
                dx_face.length() / dx_gradients,
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
                dx_face.length() / dx_gradients,
            );

            primitives_dash
        }
        Boundary::Vacuum => {
            let vacuum = State::vacuum();
            primitives_dash = pairwise_limiter(
                &part.primitives,
                &vacuum,
                &primitives_dash,
                dx_face.length() / dx_gradients,
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
        a_over_r: face.area() / dx.length(),
    }
}
