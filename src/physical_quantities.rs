use std::ops::{Add, AddAssign, Index, IndexMut, Mul, Sub, SubAssign};

use glam::DVec3;

use crate::equation_of_state::EquationOfState;

#[derive(Default, Debug, Clone, Copy)]
pub struct StateVector(f64, DVec3, f64);

impl Add for StateVector {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        StateVector(self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2)
    }
}

impl AddAssign for StateVector {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
        self.1 += rhs.1;
        self.2 += rhs.2;
    }
}

impl Sub for StateVector {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        StateVector(self.0 - rhs.0, self.1 - rhs.1, self.2 - rhs.2)
    }
}

impl SubAssign for StateVector {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
        self.1 -= rhs.1;
        self.2 -= rhs.2;
    }
}

impl Mul<StateVector> for f64 {
    type Output = StateVector;

    fn mul(self, rhs: StateVector) -> Self::Output {
        StateVector(self * rhs.0, self * rhs.1, self * rhs.2)
    }
}

impl StateVector {
    pub fn zeros() -> Self {
        StateVector(0., DVec3::ZERO, 0.)
    }

    pub fn splat(value: f64) -> Self {
        Self(value, DVec3::splat(value), value)
    }

    pub fn pairwise_max(&self, other: Self) -> Self {
        StateVector(
            self.0.max(other.0),
            self.1.max(other.1),
            self.2.max(other.2),
        )
    }

    pub fn pairwise_min(&self, other: Self) -> Self {
        StateVector(
            self.0.min(other.0),
            self.1.min(other.1),
            self.2.min(other.2),
        )
    }
}

impl Index<usize> for StateVector {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.0,
            1 => &self.1.x,
            2 => &self.1.y,
            3 => &self.1.z,
            4 => &self.2,
            _ => panic!("Index out of bounds for StateVector!"),
        }
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub struct StateGradients(DVec3, [DVec3; 3], DVec3);

impl StateGradients {
    pub fn zeros() -> Self {
        StateGradients(DVec3::ZERO, [DVec3::ZERO; 3], DVec3::ZERO)
    }

    pub fn dot(&self, dx: DVec3) -> StateVector {
        StateVector(
            self.0.dot(dx),
            DVec3 {
                x: self.1[0].dot(dx),
                y: self.1[1].dot(dx),
                z: self.1[2].dot(dx),
            },
            self.2.dot(dx),
        )
    }

    pub fn is_finite(&self) -> bool {
        self.0.is_finite()
            && self.1[0].is_finite()
            && self.1[1].is_finite()
            && self.1[2].is_finite()
            && self.2.is_finite()
    }

    pub fn div_v(&self) -> f64 {
        self.1[0].x + self.1[1].y + self.1[2].z
    }

    pub fn curl_v(&self) -> DVec3 {
        DVec3::new(
            self.1[2].y - self.1[1].z,
            self.1[0].z - self.1[2].x,
            self.1[1].x - self.1[0].y,
        )
    }
}

impl Index<usize> for StateGradients {
    type Output = DVec3;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.0,
            1 => &self.1[0],
            2 => &self.1[1],
            3 => &self.1[2],
            4 => &self.2,
            _ => panic!("Index out of bounds for StateVector!"),
        }
    }
}

impl IndexMut<usize> for StateGradients {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.0,
            1 => &mut self.1[0],
            2 => &mut self.1[1],
            3 => &mut self.1[2],
            4 => &mut self.2,
            _ => panic!("Index out of bounds for StateVector!"),
        }
    }
}

impl AddAssign for StateGradients {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
        self.1[0] += rhs.1[0];
        self.1[1] += rhs.1[1];
        self.1[2] += rhs.1[2];
        self.2 += rhs.2;
    }
}

#[derive(Default, Debug, Clone, Copy, Add, Sub, AddAssign, SubAssign)]
pub struct Primitives {
    values: StateVector,
}

impl From<StateVector> for Primitives {
    fn from(values: StateVector) -> Self {
        Primitives { values }
    }
}

impl Primitives {
    pub fn density(&self) -> f64 {
        self.values.0
    }

    pub fn velocity(&self) -> DVec3 {
        self.values.1
    }

    pub fn pressure(&self) -> f64 {
        self.values.2
    }

    pub fn vacuum() -> Self {
        Primitives {
            values: StateVector::zeros(),
        }
    }

    pub fn new(density: f64, velocity: DVec3, pressure: f64) -> Self {
        Primitives {
            values: StateVector(density, velocity, pressure),
        }
    }

    pub fn from_conserved(conserved: &Conserved, volume: f64, eos: &EquationOfState) -> Self {
        if conserved.mass() > 0. {
            let m_inv = 1. / conserved.mass();
            let density = conserved.mass() / volume;
            let velocity = conserved.momentum() * m_inv;
            let internal_energy = conserved.internal_energy();
            let pressure = eos.gas_pressure_from_internal_energy(internal_energy, density);
            assert!(density >= 0.);
            assert!(pressure >= 0.);
            Self {
                values: StateVector(density, velocity, pressure),
            }
        } else {
            Self::vacuum()
        }
    }

    pub fn boost(&self, velocity: DVec3) -> Self {
        if self.density() > 0. {
            Self::new(self.density(), self.velocity() + velocity, self.pressure())
        } else {
            debug_assert_eq!(self.density(), 0.);
            debug_assert_eq!(self.pressure(), 0.);
            *self
        }
    }

    pub fn reflect(&self, normal: DVec3) -> Self {
        // Reflect fluid velocity component along normal
        let v = self.velocity() - 2. * self.velocity().dot(normal) * normal;
        Primitives::new(self.density(), v, self.pressure())
    }

    pub fn pairwise_max(&self, other: &Self) -> Self {
        Self {
            values: self.values.pairwise_max(other.values),
        }
    }

    pub fn pairwise_min(&self, other: &Self) -> Self {
        Self {
            values: self.values.pairwise_min(other.values),
        }
    }

    pub fn values(&self) -> StateVector {
        self.values
    }

    pub fn check_physical(&mut self) {
        if self.density() < 0. {
            eprintln!("Negative density encountered, resetting to vacuum!");
            self.values = StateVector::zeros();
        }
        if self.pressure() < 0. {
            eprintln!("Negative density encountered, resetting to vacuum!");
            self.values = StateVector::zeros();
        }

        debug_assert!(
            self.density().is_finite(),
            "Infinite density after extrapolation!"
        );
        debug_assert!(
            self.velocity().is_finite(),
            "Infinite velocity after extrapolation!"
        );
        debug_assert!(
            self.pressure().is_finite(),
            "Infinite pressure after extrapolation!"
        );
    }
}

impl Mul<Primitives> for f64 {
    type Output = Primitives;

    fn mul(self, rhs: Primitives) -> Self::Output {
        Primitives {
            values: self * rhs.values,
        }
    }
}

#[derive(Default, Debug, Clone, Copy, Add, Sub, AddAssign, SubAssign)]
pub struct Conserved {
    values: StateVector,
}

impl Conserved {
    pub fn mass(&self) -> f64 {
        self.values.0
    }

    pub fn momentum(&self) -> DVec3 {
        self.values.1
    }

    pub fn energy(&self) -> f64 {
        self.values.2
    }

    /// returns the specific internal energy e: E = E_kin + E_therm = E_kin + m * e
    pub fn internal_energy(&self) -> f64 {
        let m_inv = 1. / self.mass();
        let thermal_energy = self.energy() - 0.5 * self.momentum().length_squared() * m_inv;
        thermal_energy * m_inv
    }

    pub fn vacuum() -> Self {
        Conserved {
            values: StateVector::zeros(),
        }
    }

    pub fn new(mass: f64, momentum: DVec3, energy: f64) -> Self {
        Conserved {
            values: StateVector(mass, momentum, energy),
        }
    }

    pub fn from_primitives(primitives: &Primitives, volume: f64, eos: &EquationOfState) -> Self {
        let mass = primitives.density() * volume;
        let momentum = mass * primitives.velocity();
        let energy = 0.5 * momentum.dot(primitives.velocity())
            + mass
                * eos.gas_internal_energy_from_pressure(
                    primitives.pressure(),
                    1. / primitives.density(),
                );
        Self {
            values: StateVector(mass, momentum, energy),
        }
    }

    pub fn pairwise_max(&self, other: Self) -> Self {
        Self {
            values: self.values.pairwise_max(other.values),
        }
    }

    pub fn pairwise_min(&self, other: Self) -> Self {
        Self {
            values: self.values.pairwise_min(other.values),
        }
    }

    pub fn values(&self) -> StateVector {
        self.values
    }
}

impl Mul<Conserved> for f64 {
    type Output = Conserved;

    fn mul(self, rhs: Conserved) -> Self::Output {
        Conserved {
            values: self * rhs.values,
        }
    }
}

#[cfg(test)]
mod test {
    use float_cmp::assert_approx_eq;
    use glam::DVec3;
    use yaml_rust::YamlLoader;

    use super::{Conserved, Primitives};
    use crate::equation_of_state::EquationOfState;

    #[test]
    fn test_conversions() {
        let primitives = Primitives::new(
            0.75,
            DVec3 {
                x: 0.4,
                y: 0.,
                z: 0.,
            },
            0.8,
        );
        let volume = 0.1;
        let eos = EquationOfState::new(&YamlLoader::load_from_str("gamma: 1.666667").unwrap()[0])
            .unwrap();
        let conserved = Conserved::from_primitives(&primitives, volume, &eos);
        let primitives_new = Primitives::from_conserved(&conserved, volume, &eos);

        assert_approx_eq!(f64, primitives.density(), primitives_new.density());
        assert_approx_eq!(f64, primitives.velocity().x, primitives_new.velocity().x);
        assert_approx_eq!(f64, primitives.velocity().y, primitives_new.velocity().y);
        assert_approx_eq!(f64, primitives.velocity().z, primitives_new.velocity().z);
        assert_approx_eq!(f64, primitives.pressure(), primitives_new.pressure());
    }
}
