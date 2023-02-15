use glam::{DMat3, DVec3};

use crate::{
    physical_quantities::{Primitives, StateGradients, StateVector},
    utils::HydroDimension,
};

pub struct GradientData {
    gradients: StateGradients,
    matrix_wls: DMat3,
}

impl GradientData {
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
            gradients: StateGradients::zeros(),
            matrix_wls,
        }
    }

    pub fn collect(
        &mut self,
        primitives_left: &Primitives,
        primitives_right: &Primitives,
        w: f64,
        ds: DVec3,
    ) {
        fn gradient_est_single_quantity(q_l: f64, q_r: f64, w: f64, ds: DVec3, grad: &mut DVec3) {
            *grad += w * (q_r - q_l) * ds;
        }

        gradient_est_single_quantity(
            primitives_left.density(),
            primitives_right.density(),
            w,
            ds,
            &mut self.gradients[0],
        );
        gradient_est_single_quantity(
            primitives_left.velocity().x,
            primitives_right.velocity().x,
            w,
            ds,
            &mut self.gradients[1],
        );
        gradient_est_single_quantity(
            primitives_left.velocity().y,
            primitives_right.velocity().y,
            w,
            ds,
            &mut self.gradients[2],
        );
        gradient_est_single_quantity(
            primitives_left.velocity().z,
            primitives_right.velocity().z,
            w,
            ds,
            &mut self.gradients[3],
        );
        gradient_est_single_quantity(
            primitives_left.pressure(),
            primitives_right.pressure(),
            w,
            ds,
            &mut self.gradients[4],
        );

        self.matrix_wls += DMat3::from_cols(w * ds.x * ds, w * ds.y * ds, w * ds.z * ds);
    }

    pub fn finalize(mut self) -> StateGradients {
        let matrix_wls = self.matrix_wls.inverse();
        self.gradients[0] = matrix_wls.mul_vec3(self.gradients[0]);
        self.gradients[1] = matrix_wls.mul_vec3(self.gradients[1]);
        self.gradients[2] = matrix_wls.mul_vec3(self.gradients[2]);
        self.gradients[3] = matrix_wls.mul_vec3(self.gradients[3]);
        self.gradients[4] = matrix_wls.mul_vec3(self.gradients[4]);
        self.gradients
    }
}

pub struct LimiterData {
    pub min: Primitives,
    pub max: Primitives,
    pub e_min: Primitives,
    pub e_max: Primitives,
}

impl Default for LimiterData {
    fn default() -> Self {
        Self {
            min: StateVector::splat(f64::INFINITY).into(),
            max: StateVector::splat(f64::NEG_INFINITY).into(),
            e_min: StateVector::splat(f64::INFINITY).into(),
            e_max: StateVector::splat(f64::NEG_INFINITY).into(),
        }
    }
}

impl LimiterData {
    pub fn collect(&mut self, primitives: &Primitives, extrapolated: &Primitives) {
        self.min = self.min.pairwise_min(primitives);
        self.max = self.max.pairwise_max(primitives);
        self.e_min = self.e_min.pairwise_min(extrapolated);
        self.e_max = self.e_max.pairwise_max(extrapolated);
    }

    fn limit_single_quantity(max: f64, min: f64, e_max: f64, e_min: f64) -> f64 {
        if e_max == 0. || e_min == 0. {
            1.
        } else {
            (1.0f64).min((max / e_max).min(min / e_min))
        }
    }

    pub fn limit(&self, gradients: &mut StateGradients, primitives: &Primitives) {
        let density_alpha = Self::limit_single_quantity(
            self.max.density() - primitives.density(),
            self.min.density() - primitives.density(),
            self.e_max.density(),
            self.e_min.density(),
        );
        let velocity_alpha_x = Self::limit_single_quantity(
            self.max.velocity().x - primitives.velocity().x,
            self.min.velocity().x - primitives.velocity().x,
            self.e_max.velocity().x,
            self.e_min.velocity().x,
        );
        let velocity_alpha_y = Self::limit_single_quantity(
            self.max.velocity().y - primitives.velocity().y,
            self.min.velocity().y - primitives.velocity().y,
            self.e_max.velocity().y,
            self.e_min.velocity().y,
        );
        let velocity_alpha_z = Self::limit_single_quantity(
            self.max.velocity().z - primitives.velocity().z,
            self.min.velocity().z - primitives.velocity().z,
            self.e_max.velocity().z,
            self.e_min.velocity().z,
        );
        let pressure_alpha = Self::limit_single_quantity(
            self.max.pressure() - primitives.pressure(),
            self.min.pressure() - primitives.pressure(),
            self.e_max.pressure(),
            self.e_min.pressure(),
        );

        gradients[0] *= density_alpha;
        gradients[1] *= velocity_alpha_x;
        gradients[2] *= velocity_alpha_y;
        gradients[3] *= velocity_alpha_z;
        gradients[4] *= pressure_alpha;
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
    primitives_left: &Primitives,
    primitives_right: &Primitives,
    primitives_dash: &Primitives,
    dx_fac: f64,
) -> Primitives {
    let mut limited = Primitives::new(
        pairwise_limiter_single_quantity(
            primitives_left.density(),
            primitives_right.density(),
            primitives_dash.density(),
            dx_fac,
        ),
        DVec3 {
            x: pairwise_limiter_single_quantity(
                primitives_left.velocity().x,
                primitives_right.velocity().x,
                primitives_dash.velocity().x,
                dx_fac,
            ),
            y: pairwise_limiter_single_quantity(
                primitives_left.velocity().y,
                primitives_right.velocity().y,
                primitives_dash.velocity().y,
                dx_fac,
            ),
            z: pairwise_limiter_single_quantity(
                primitives_left.velocity().z,
                primitives_right.velocity().z,
                primitives_dash.velocity().z,
                dx_fac,
            ),
        },
        pairwise_limiter_single_quantity(
            primitives_left.pressure(),
            primitives_right.pressure(),
            primitives_dash.pressure(),
            dx_fac,
        ),
    );

    limited.check_physical();

    limited
}
