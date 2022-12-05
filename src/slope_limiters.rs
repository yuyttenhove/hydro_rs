use crate::physical_quantities::Primitives;

fn cell_wide_limiter_single_quantity(max: f64, min: f64, emax: f64, emin: f64) -> f64 {
    if emax == 0. || emin == 0. {
        1.
    } else {
        (1.0f64).min((max / emax).min(min / emin))
    }
}

pub fn cell_wide_limiter(gradients: Primitives, primitives: Primitives, left: Primitives, right: Primitives, dx_left: f64, dx_right: f64) -> Primitives {
    let max_primitives = left.pairwise_max(right) - primitives;
    let min_primitives = left.pairwise_min(right) - primitives;
    let ext_left = dx_left * gradients;
    let ext_right = dx_right * gradients;
    let ext_max = ext_left.pairwise_max(ext_right);
    let ext_min = ext_left.pairwise_min(ext_right);

    let density_alpha = cell_wide_limiter_single_quantity(
        max_primitives.density(), min_primitives.density(), ext_max.density(), ext_min.density()
    );
    let velocity_alpha = cell_wide_limiter_single_quantity(
        max_primitives.velocity(), min_primitives.velocity(), ext_max.velocity(), ext_min.velocity()
    );
    let pressure_alpha = cell_wide_limiter_single_quantity(
        max_primitives.pressure(), min_primitives.pressure(), ext_max.pressure(), ext_min.pressure()
    );

    Primitives::new(
        density_alpha * gradients.density(), 
        velocity_alpha * gradients.velocity(), 
        pressure_alpha * gradients.pressure()
    )
}

fn pairwise_limiter_single_quantity(q_l: f64, q_r: f64, q_dash: f64) -> f64 {
    if q_l == q_r { return q_l; }

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

pub fn pairwise_limiter(primitives_left: Primitives, primitives_right: Primitives, primitives_dash: Primitives) -> Primitives {
    let mut limited = Primitives::new(
        pairwise_limiter_single_quantity(
            primitives_left.density(), primitives_right.density(), primitives_dash.density()
        ),
        pairwise_limiter_single_quantity(
            primitives_left.velocity(), primitives_right.velocity(), primitives_dash.velocity()
        ), 
        pairwise_limiter_single_quantity(
            primitives_left.pressure(), primitives_right.pressure(), primitives_dash.pressure()
        )
    );

    limited.check_physical();

    limited

}