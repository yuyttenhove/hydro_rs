use glam::DVec3;

pub trait Round {
    #[allow(dead_code)]
    fn round_to(&self, decimal_places: u8) -> Self;
}

impl Round for f64 {
    fn round_to(&self, decimal_places: u8) -> Self {
        let factor = 10.0f64.powi(decimal_places as i32);
        (self * factor).round() / factor
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
