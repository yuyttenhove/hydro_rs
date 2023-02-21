use glam::DVec3;

use crate::part::Particle;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum HydroDimension {
    HydroDimension1D,
    HydroDimension2D,
    HydroDimension3D,
}

impl From<usize> for HydroDimension {
    fn from(u: usize) -> Self {
        match u {
            1 => HydroDimension::HydroDimension1D,
            2 => HydroDimension::HydroDimension2D,
            3 => HydroDimension::HydroDimension3D,
            _ => panic!("Illegal hydro dimension: {}", u),
        }
    }
}

impl From<HydroDimension> for usize {
    fn from(dim: HydroDimension) -> Self {
        match dim {
            HydroDimension::HydroDimension1D => 1,
            HydroDimension::HydroDimension2D => 2,
            HydroDimension::HydroDimension3D => 3,
        }
    }
}

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

pub fn box_wrap(box_size: DVec3, pos: &mut DVec3, dimension: usize) {
    for i in 0..dimension {
        while pos[i] < 0. {
            pos[i] += box_size[i];
        }
        while pos[i] >= box_size[i] {
            pos[i] -= box_size[i];
        }
    }
}

pub fn box_reflect(box_size: DVec3, part: &mut Particle) {
    let x_old = part.x;
    for i in 0..3 {
        if part.x[i] < 0. {
            part.x[i] -= 2. * part.x[i];
        }
        if part.x[i] > box_size[i] {
            part.x[i] -= 2. * (part.x[i] - box_size[i]);
        }
    }
    debug_assert!(contains(box_size, part.x));
    let dx = part.x - x_old;
    if dx.length_squared() > 0. {
        let normal = dx * dx.length_recip();
        *part = part
            .clone()
            .reflect_quantities(normal)
            .reflect_gradients(normal);
    }
}
