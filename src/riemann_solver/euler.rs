use crate::finite_volume_solver::{FluxLimiterData, FluxLimiterFunction};
use crate::gas_law::GasLaw;
use crate::physical_quantities::{Conserved, Primitive, State};
use crate::riemann_solver::{
    flux_from_half_state, ExactRiemannSolver, RiemannStarSolver, RiemannStarValues,
};
use glam::DVec3;

pub trait EulerSolver {}

pub(super) fn solve_for_waf_flux(
    left: &State<Primitive>,
    right: &State<Primitive>,
    star_values: &RiemannStarValues,
    dx_left: DVec3,
    dx_right: DVec3,
    left_flux_limiter: &FluxLimiterData,
    right_flux_limiter: &FluxLimiterData,
    r: f64,
    do_limit: bool,
    flux_limiter_function: &FluxLimiterFunction,
    interface_velocity: DVec3,
    dt: f64,
    n_unit: DVec3,
    eos: &GasLaw,
) -> State<Conserved> {
    // let godunov_flux = self.solve_for_flux(left, right, interface_velocity, n_unit, eos);

    // Boost to interface frame
    let left = left.boost(-interface_velocity);
    let right = right.boost(-interface_velocity);

    let v_l = left.velocity().dot(n_unit);
    let v_r = right.velocity().dot(n_unit);
    let a_l = eos.sound_speed(left.pressure(), 1. / left.density());
    let a_r = eos.sound_speed(right.pressure(), 1. / right.density());

    let a_star_l = eos.sound_speed(star_values.p, 1. / star_values.rho_l);
    let a_star_r = eos.sound_speed(star_values.p, 1. / star_values.rho_r);

    // Get 4 states and wave speeds
    let mut states = [
        left,
        State::<Primitive>::new(
            star_values.rho_l,
            left.velocity() + (star_values.u - v_l) * n_unit,
            star_values.p,
        ),
        State::<Primitive>::new(
            star_values.rho_r,
            right.velocity() + (star_values.u - v_r) * n_unit,
            star_values.p,
        ),
        right,
    ];
    let wave_speeds = [
        if left.pressure() < star_values.p {
            // shock wave
            <ExactRiemannSolver as RiemannStarSolver>::shock_speed(
                star_values.u,
                a_star_l,
                star_values.p / left.pressure(),
                eos.gamma(),
            )
        } else {
            // rarefaction wave, sonic?
            let s_hl = v_l - a_l;
            let s_tl = star_values.u - a_star_l;
            if s_hl * s_tl < 0. {
                // Sonic, update left middle state
                states[1] = <ExactRiemannSolver as RiemannStarSolver>::sample_rarefaction_fan(
                    &left,
                    a_l,
                    v_l,
                    n_unit,
                    eos.gamma(),
                );
            }
            s_hl
        },
        star_values.u,
        if right.pressure() < star_values.p {
            // shock wave
            <ExactRiemannSolver as RiemannStarSolver>::shock_speed(
                star_values.u,
                -a_star_r,
                star_values.p / right.pressure(),
                eos.gamma(),
            )
        } else {
            // rarefaction wave, sonic?
            let s_hl = v_r + a_r;
            let s_tl = star_values.u + a_star_r;
            if s_hl * s_tl < 0. {
                // Sonic, update left middle state
                states[2] = <ExactRiemannSolver as RiemannStarSolver>::sample_rarefaction_fan(
                    &right,
                    -a_r,
                    v_r,
                    n_unit,
                    eos.gamma(),
                );
            }
            s_hl
        },
    ];

    // Compute fluxes of 4 states
    let fluxes = [
        flux_from_half_state(&states[0], interface_velocity, n_unit, eos.gamma()),
        flux_from_half_state(&states[1], interface_velocity, n_unit, eos.gamma()),
        flux_from_half_state(&states[2], interface_velocity, n_unit, eos.gamma()),
        flux_from_half_state(&states[3], interface_velocity, n_unit, eos.gamma()),
    ];

    // Flux limiter
    let dx = dx_left.dot(n_unit) + dx_right.dot(n_unit);
    let jumps_local = DVec3::new(
        states[1].density() - states[0].density(),
        states[2].density() - states[1].density(),
        states[3].density() - states[2].density(),
    );
    // Cancel out jumps in left and right flux limiter
    let jumps_left = -left_flux_limiter.apply(jumps_local, r);
    let jumps_right = right_flux_limiter.apply(-jumps_local, r);
    let mut phi = wave_speeds;
    if do_limit {
        for i in 0..3 {
            let jumps_local_inv = if jumps_local[i] != 0. {
                1. / jumps_local[i]
            } else {
                0.
            };
            let r = if wave_speeds[i] < 0. {
                jumps_right[i] * jumps_local_inv
            } else {
                jumps_left[i] * jumps_local_inv
            };
            let psi_r = flux_limiter_function.limit(r);
            phi[i] = wave_speeds[i].signum() * (dx - (dx - wave_speeds[i].abs() * dt) * psi_r);
        }
    }

    // Compute WAF flux
    let mut waf_flux = 0.5 * dx * (fluxes[0] + fluxes[3]);
    for i in 1..4 {
        waf_flux -= 0.5 * phi[i - 1] * (fluxes[i] - fluxes[i - 1]);
    }
    waf_flux = 1. / dx * waf_flux;

    assert!(waf_flux.mass().is_finite());
    assert!(waf_flux.momentum().is_finite());
    assert!(waf_flux.energy().is_finite());

    waf_flux
}
