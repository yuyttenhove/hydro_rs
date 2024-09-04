use glam::{DMat3, DVec3};

use crate::{
    physical_quantities::{Gradients, Primitive, State},
    utils::HydroDimension,
};

pub struct GradientData<T> {
    gradients: Gradients<T>,
    matrix_wls: DMat3,
}

impl<T> GradientData<T> {
    pub fn init(dimensionality: HydroDimension) -> Self {
        let matrix_wls = match dimensionality {
            HydroDimension::HydroDimension1D => {
                let mut mat = DMat3::IDENTITY;
                mat.x_axis.x = 0.;
                mat
            }
            HydroDimension::HydroDimension2D => {
                let mut mat = DMat3::ZERO;
                mat.z_axis.z = 1.;
                mat
            }
            HydroDimension::HydroDimension3D => DMat3::ZERO,
        };

        Self {
            gradients: Gradients::zeros(),
            matrix_wls,
        }
    }

    pub fn collect(&mut self, state_left: &State<T>, state_right: &State<T>, w: f64, ds: DVec3) {
        for i in 0..5 {
            self.gradients[i] += w * (state_right[i] - state_left[i]) * ds;
        }

        self.matrix_wls += w * DMat3::from_cols(ds.x * ds, ds.y * ds, ds.z * ds);
    }

    pub fn finalize(mut self) -> Gradients<T> {
        let matrix_wls = self.matrix_wls.inverse();
        for i in 0..5 {
            self.gradients[i] = matrix_wls.mul_vec3(self.gradients[i]);
        }
        if !self.gradients.is_finite() {
            eprintln!(
                "Gradient calculation failed! Falling back to first order for this particle..."
            );
            self.gradients = Gradients::zeros();
        }
        self.gradients
    }
}

pub struct LimiterData<T> {
    pub min: State<T>,
    pub max: State<T>,
    pub e_min: State<T>,
    pub e_max: State<T>,
}

impl<T: Copy> LimiterData<T> {
    pub fn init(state: &State<T>) -> Self {
        Self {
            min: *state,
            max: *state,
            e_min: State::splat(f64::INFINITY).into(),
            e_max: State::splat(f64::NEG_INFINITY).into(),
        }
    }
}

impl<T> LimiterData<T> {
    pub fn collect(&mut self, state: &State<T>, extrapolated: &State<T>) {
        self.min = self.min.pairwise_min(state);
        self.max = self.max.pairwise_max(state);
        self.e_min = self.e_min.pairwise_min(extrapolated);
        self.e_max = self.e_max.pairwise_max(extrapolated);
    }

    fn limit_single_quantity(d_max: f64, d_min: f64, e_max: f64, e_min: f64) -> f64 {
        if e_max == 0. || e_min == 0. {
            1.
        } else {
            (1.0f64).min((d_max / e_max).min(d_min / e_min))
        }
    }

    pub fn limit(&self, gradients: &mut Gradients<T>, state: &State<T>) {
        for i in 0..5 {
            let alpha = Self::limit_single_quantity(
                self.max[i] - state[i],
                self.min[i] - state[i],
                self.e_max[i],
                self.e_min[i],
            );
            gradients[i] *= alpha;
        }
    }
}

fn pairwise_limiter_single_quantity(q_l: f64, q_r: f64, q_dash: f64, dx_fac: f64) -> f64 {
    if q_l == q_r {
        return q_l;
    }

    let q_bar = q_l + dx_fac * (q_r - q_l);
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
    primitives_left: &State<Primitive>,
    primitives_right: &State<Primitive>,
    primitives_dash: &State<Primitive>,
    dx_fac: f64,
) -> State<Primitive> {
    let mut limited = State::vacuum();
    for i in 0..5 {
        limited[i] = pairwise_limiter_single_quantity(
            primitives_left[i],
            primitives_right[i],
            primitives_dash[i],
            dx_fac,
        )
    }
    limited.check_physical();

    limited
}
