use glam::DVec3;

use crate::part::Part;

pub trait Round {
    fn round_to(&self, decimal_places: u8) -> Self;
}

impl Round for f64 {
    fn round_to(&self, decimal_places: u8) -> Self {
        let factor = 10.0f64.powi(decimal_places as i32);
        (self * factor).round() as f64 / factor
    }
}

pub fn contains(box_size: DVec3, pos: DVec3) -> bool {
    pos.x >= 0.
        && pos.x < box_size.x
        && pos.y >= 0.
        && pos.y < box_size.y
        && pos.z >= 0.
        && pos.z < box_size.z
}

pub fn box_wrap(box_size: DVec3, pos: &mut DVec3) {
    for i in 0..3 {
        while pos[i] < 0. {
            pos[i] += box_size[i];
        }
        while pos[i] >= box_size[i] {
            pos[i] -= box_size[i];
        }
    }
}

pub fn box_reflect(box_size: DVec3, part: &mut Part) {
    let x_old = part.x;
    for i in 0..3 {
        if part.x[i] < 0. {
            part.x[i] += 2. * part.x[i];
        }
        if part.x[i] > box_size[i] {
            part.x[i] -= 2. * (box_size[i] - part.x[i]);
        }
    }
    debug_assert!(contains(box_size, part.x));
    let dx = part.x - x_old;
    if dx.length_squared() > 0. {
        let normal = dx * dx.length_recip();
        part.reflect_quantities(dx).reflect_gradients(dx);
    }
}