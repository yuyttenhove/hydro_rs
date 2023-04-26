pub trait Kernel {
    fn kernel(&self, r: f64, h: f64) -> f64;

    fn q(r: f64, h: f64) -> f64 {
        3. * r / h
    }
}

pub struct QuinticSpline;

impl Kernel for QuinticSpline {
    fn kernel(&self, r: f64, h: f64) -> f64 {
        let q = Self::q(r, h);
        if q > 3. {
            0.
        } else if q > 2. {
            (3. - q).powi(5)
        } else if q > 1. {
            (3. - q).powi(5) - 6. * (2. - q).powi(5)
        } else {
            (3. - q).powi(5) - 6. * (2. - q).powi(5) + 15. * (1. - q).powi(5)
        }
    }
}

pub struct WendlandC2;

impl Kernel for WendlandC2 {
    fn q(r: f64, h: f64) -> f64 {
        2. * r / h
    }

    fn kernel(&self, r: f64, h: f64) -> f64 {
        let q = Self::q(r, h);
        if q > 2. {
            0.
        } else {
            (1. - 0.5 * q).powi(4) * (2. * q + 1.)
        }
    }
}

pub struct WendlandC4;

impl Kernel for WendlandC4 {
    fn q(r: f64, h: f64) -> f64 {
        2. * r / h
    }

    fn kernel(&self, r: f64, h: f64) -> f64 {
        let q = Self::q(r, h);
        if q > 2. {
            0.
        } else {
            (1. - 0.5 * q).powi(6) * (35. / 12. * q.powi(2) * 3. * q + 1.)
        }
    }
}

pub struct WendlandC6;

impl Kernel for WendlandC6 {
    fn q(r: f64, h: f64) -> f64 {
        2. * r / h
    }

    fn kernel(&self, r: f64, h: f64) -> f64 {
        let q = Self::q(r, h);
        if q > 2. {
            0.
        } else {
            (1. - 0.5 * q).powi(8) * (4. * q.powi(3) + 6.25 * q.powi(2) + 4. * q + 1.)
        }
    }
}

pub struct OneOver(pub i32);

impl Kernel for OneOver {
    fn kernel(&self, r: f64, h: f64) -> f64 {
        if r < h {
            1. / r.powi(self.0)
        } else {
            0.
        }
    }
}
