use glam::DVec3;

use crate::physical_quantities::{Primitives, StateGradients};

fn cell_wide_limiter_single_quantity(max: f64, min: f64, emax: f64, emin: f64) -> f64 {
    if emax == 0. || emin == 0. {
        1.
    } else {
        (1.0f64).min((max / emax).min(min / emin))
    }
}

pub fn cell_wide_limiter(
    gradients: StateGradients,
    primitives: Primitives,
    left: Primitives,
    right: Primitives,
    dx_left: DVec3,
    dx_right: DVec3,
) -> StateGradients {
    let max_primitives = left.pairwise_max(right) - primitives;
    let min_primitives = left.pairwise_min(right) - primitives;
    let ext_left: Primitives = gradients.dot(dx_left).into();
    let ext_right: Primitives = gradients.dot(dx_right).into();
    let ext_max = ext_left.pairwise_max(ext_right);
    let ext_min = ext_left.pairwise_min(ext_right);

    let density_alpha = cell_wide_limiter_single_quantity(
        max_primitives.density(),
        min_primitives.density(),
        ext_max.density(),
        ext_min.density(),
    );
    let velocity_alpha_x = cell_wide_limiter_single_quantity(
        max_primitives.velocity().x,
        min_primitives.velocity().x,
        ext_max.velocity().x,
        ext_min.velocity().x,
    );
    let velocity_alpha_y = cell_wide_limiter_single_quantity(
        max_primitives.velocity().y,
        min_primitives.velocity().y,
        ext_max.velocity().y,
        ext_min.velocity().y,
    );
    let velocity_alpha_z = cell_wide_limiter_single_quantity(
        max_primitives.velocity().z,
        min_primitives.velocity().z,
        ext_max.velocity().z,
        ext_min.velocity().z,
    );
    let pressure_alpha = cell_wide_limiter_single_quantity(
        max_primitives.pressure(),
        min_primitives.pressure(),
        ext_max.pressure(),
        ext_min.pressure(),
    );

    StateGradients::new(
        density_alpha * gradients[0],
        velocity_alpha_x * gradients[1],
        velocity_alpha_y * gradients[2],
        velocity_alpha_z * gradients[3],
        pressure_alpha * gradients[4],
    )
}

fn pairwise_limiter_single_quantity(q_l: f64, q_r: f64, q_dash: f64) -> f64 {
    if q_l == q_r {
        return q_l;
    }

    let q_bar = q_l + 0.5 * (q_r - q_l);
    let q_diff = (q_l - q_r).abs();
    let delta1 = 0.5 * q_diff;
    let delta2 = 0.25 * q_diff;

    if q_l < q_r {
        let q_min = q_l.min(q_r);
        let qmin = if (q_min - delta1) * q_min > 0. {
            q_min - delta1
        } else {
            q_min * q_min.abs() / (q_min.abs() + delta1)
        };
        qmin.max((q_bar + delta2).min(q_dash))
    } else {
        let q_max = q_l.max(q_r);
        let qplu = if (q_max + delta1) * q_max > 0. {
            q_max + delta1
        } else {
            q_max * q_max.abs() / (q_max.abs() + delta1)
        };
        qplu.min((q_bar - delta2).max(q_dash))
    }
}

pub fn pairwise_limiter(
    primitives_left: Primitives,
    primitives_right: Primitives,
    primitives_dash: Primitives,
) -> Primitives {
    let mut limited = Primitives::new(
        pairwise_limiter_single_quantity(
            primitives_left.density(),
            primitives_right.density(),
            primitives_dash.density(),
        ),
        DVec3 {
            x: pairwise_limiter_single_quantity(
                primitives_left.velocity().x,
                primitives_right.velocity().x,
                primitives_dash.velocity().x,
            ),
            y: pairwise_limiter_single_quantity(
                primitives_left.velocity().y,
                primitives_right.velocity().y,
                primitives_dash.velocity().y,
            ),
            z: pairwise_limiter_single_quantity(
                primitives_left.velocity().z,
                primitives_right.velocity().z,
                primitives_dash.velocity().z,
            ),
        },
        pairwise_limiter_single_quantity(
            primitives_left.pressure(),
            primitives_right.pressure(),
            primitives_dash.pressure(),
        ),
    );

    limited.check_physical();

    limited
}
