use glam::DVec3;

use crate::physical_quantities::{Primitives, StateGradients};

fn cell_wide_limiter_single_quantity(max: f64, min: f64, e_max: f64, e_min: f64) -> f64 {
    if e_max == 0. || e_min == 0. {
        1.
    } else {
        (1.0f64).min((max / e_max).min(min / e_min))
    }
}

pub fn cell_wide_limiter(
    min: &Primitives,
    max: &Primitives,
    e_min: &Primitives,
    e_max: &Primitives,
    gradients: &mut StateGradients,
) {
    let density_alpha = cell_wide_limiter_single_quantity(
        max.density(),
        min.density(),
        e_max.density(),
        e_min.density(),
    );
    let velocity_alpha_x = cell_wide_limiter_single_quantity(
        max.velocity().x,
        min.velocity().x,
        e_max.velocity().x,
        e_min.velocity().x,
    );
    let velocity_alpha_y = cell_wide_limiter_single_quantity(
        max.velocity().y,
        min.velocity().y,
        e_max.velocity().y,
        e_min.velocity().y,
    );
    let velocity_alpha_z = cell_wide_limiter_single_quantity(
        max.velocity().z,
        min.velocity().z,
        e_max.velocity().z,
        e_min.velocity().z,
    );
    let pressure_alpha = cell_wide_limiter_single_quantity(
        max.pressure(),
        min.pressure(),
        e_max.pressure(),
        e_min.pressure(),
    );

    gradients[0] *= density_alpha;
    gradients[1] *= velocity_alpha_x;
    gradients[2] *= velocity_alpha_y;
    gradients[3] *= velocity_alpha_z;
    gradients[4] *= pressure_alpha;
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
