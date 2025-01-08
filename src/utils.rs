use crate::part::Particle;
use crate::physical_quantities::State;
use crate::Boundary;
use glam::DVec3;
use meshless_voronoi::VoronoiFace;

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

pub fn interface_velocity(
    left: DVec3,
    right: DVec3,
    v_l: DVec3,
    v_r: DVec3,
    centroid: DVec3,
) -> DVec3 {
    // Compute interface velocity (Springel (2010), eq. 33):
    let midpoint = 0.5 * (left + right);
    let dx = right - left;
    let fac = (v_r - v_l).dot(centroid - midpoint) / dx.length_squared();
    0.5 * (v_l + v_r) - fac * dx
}

pub fn get_other(face: &VoronoiFace, part_idx: usize) -> Option<usize> {
    if part_idx == face.left() {
        face.right()
    } else {
        Some(face.left())
    }
}

pub fn get_boundary_part(boundary: Boundary, part: &Particle, face: &VoronoiFace) -> Particle {
    match boundary {
        Boundary::Reflective => part
            .reflect(face.centroid(), face.normal())
            .reflect_quantities(face.normal()),
        Boundary::Open => part.reflect(face.centroid(), face.normal()),
        Boundary::Vacuum => {
            let mut reflected = part.reflect(face.centroid(), face.normal());
            reflected.primitives = State::vacuum();
            reflected
        }
        _ => panic!(
            "Trying to create boundary particle with {:?} boundary conditions",
            boundary
        ),
    }
}
