use crate::engine::Engine;
use crate::equation_of_state::EquationOfState;
use crate::part::Part;
use crate::physical_quantities::{Conserved, Primitives};
use crate::slope_limiters::pairwise_limiter;
use crate::space::Boundary;
use glam::DVec3;
use meshless_voronoi::VoronoiFace;

pub struct FluxInfo {
    pub fluxes: Conserved,
    pub mflux: DVec3,
    pub v_max: f64,
}

pub fn flux_exchange(
    left: &Part,
    right: &Part,
    dt: f64,
    face: &VoronoiFace,
    eos: &EquationOfState,
    engine: &Engine,
) -> FluxInfo {
    // We extrapolate from the centroid of the particles.
    let dx_left = face.centroid() - left.centroid;
    let dx_right = face.centroid() - right.centroid;
    let dx_centroid = right.centroid - left.centroid;
    let dx = right.x - left.x;

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
    );
    let primitives_right = pairwise_limiter(
        &right.primitives,
        &left.primitives,
        &(right.primitives + right.gradients.dot(dx_right).into() + right.extrapolations),
    );

    // Compute interface velocity (Springel (2010), eq. 33):
    let midpoint = 0.5 * (left.x + right.x);
    let fac = (right.v - left.v).dot(face.centroid() - midpoint) / dx.length_squared();
    let v_face = 0.5 * (left.v + right.v) - fac * dx;

    // Calculate fluxes
    let fluxes = face.area()
        * engine.solver.solve_for_flux(
            &primitives_left.boost(-v_face),
            &primitives_right.boost(-v_face),
            v_face,
            face.normal(),
            eos,
        );

    FluxInfo {
        fluxes: dt * fluxes,
        mflux: dx_centroid * fluxes.mass(),
        v_max,
    }
}

pub fn flux_exchange_boundary(
    part: &Part,
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
    let primitives_dash =
        part.primitives + part.gradients.dot(dx_face).into() + part.extrapolations;

    let fluxes = face.area()
        * match boundary {
            Boundary::Reflective => {
                // Also reflect velocity
                reflected = reflected.reflect_quantities(face.normal());

                v_max -= (reflected.primitives.velocity() - part.primitives.velocity())
                    .dot(dx_centroid)
                    .min(0.);

                let primitives =
                    pairwise_limiter(&part.primitives, &reflected.primitives, &primitives_dash);

                // get reflected primitives at face
                let primitives_reflected = primitives.reflect(face.normal());

                engine.solver.solve_for_flux(
                    &primitives,
                    &primitives_reflected,
                    DVec3::ZERO,
                    face.normal(),
                    eos,
                )
            }
            Boundary::Open => {
                let primitives =
                    pairwise_limiter(&part.primitives, &part.primitives, &primitives_dash);

                engine.solver.solve_for_flux(
                    &primitives,
                    &primitives,
                    DVec3::ZERO,
                    face.normal(),
                    eos,
                )
            }
            Boundary::Vacuum => {
                let vacuum = Primitives::vacuum();

                let primitives = pairwise_limiter(&part.primitives, &vacuum, &primitives_dash);

                engine
                    .solver
                    .solve_for_flux(&primitives, &vacuum, DVec3::ZERO, face.normal(), eos)
            }
            Boundary::Periodic => {
                unreachable!("This method should not be called with periodic boundary conditions!");
            }
        };

    FluxInfo {
        fluxes: part.dt * fluxes,
        mflux: dx * fluxes.mass(),
        v_max,
    }
}
