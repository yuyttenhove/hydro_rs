use glam::DVec3;
use meshless_voronoi::Dimensionality;


#[derive(Clone, Copy, PartialEq, Eq, num_enum::IntoPrimitive, num_enum::TryFromPrimitive)]
#[repr(usize)]
pub enum HydroDimension {
    HydroDimension1D = 1,
    HydroDimension2D = 2,
    HydroDimension3D = 3,
}

impl From<Dimensionality> for HydroDimension {
    fn from(d: Dimensionality) -> Self {
        match d {
            Dimensionality::OneD => HydroDimension::HydroDimension1D,
            Dimensionality::TwoD => HydroDimension::HydroDimension2D,
            Dimensionality::ThreeD => HydroDimension::HydroDimension3D,
        }
    }
}

impl From<HydroDimension> for Dimensionality {
    fn from(d: HydroDimension) -> Self {
        match d {
            HydroDimension::HydroDimension1D => Dimensionality::OneD,
            HydroDimension::HydroDimension2D => Dimensionality::TwoD,
            HydroDimension::HydroDimension3D => Dimensionality::ThreeD,
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

pub fn contains(box_size: DVec3, pos: DVec3, dimension: usize) -> bool {
    let mut contains = true;
    for i in 0..dimension {
        contains &= pos[i] >= 0. && pos[i] < box_size[i];
    }
    contains
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

pub fn box_reflect(box_size: DVec3, pos: &mut DVec3, dimension: usize) {
    for i in 0..dimension {
        if pos[i] < 0. {
            pos[i] -= 2. * pos[i];
        }
        if pos[i] > box_size[i] {
            pos[i] -= 2. * (pos[i] - box_size[i]);
        }
    }
}
