use std::{
    marker::PhantomData,
    ops::{Add, AddAssign, Index, IndexMut, Mul, Sub, SubAssign},
};

use glam::DVec3;

use crate::gas_law::GasLaw;

#[derive(Default, Debug, Clone, Copy)]
pub struct Primitive;
#[derive(Default, Debug, Clone, Copy)]
pub struct Conserved;

#[derive(Default, Debug, Clone, Copy)]
pub struct State<T>(f64, DVec3, f64, PhantomData<T>);

impl<T> Add for State<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2, PhantomData)
    }
}

impl<T> AddAssign for State<T> {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
        self.1 += rhs.1;
        self.2 += rhs.2;
    }
}

impl<T> Sub for State<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0, self.1 - rhs.1, self.2 - rhs.2, PhantomData)
    }
}

impl<T> SubAssign for State<T> {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
        self.1 -= rhs.1;
        self.2 -= rhs.2;
    }
}

impl<T> Mul<State<T>> for f64 {
    type Output = State<T>;

    fn mul(self, rhs: State<T>) -> Self::Output {
        State::<T>(self * rhs.0, self * rhs.1, self * rhs.2, PhantomData)
    }
}

impl<T> State<T> {
    pub(crate) fn splat(value: f64) -> Self {
        Self(value, DVec3::splat(value), value, PhantomData)
    }

    pub fn vacuum() -> Self {
        Self(0., DVec3::ZERO, 0., PhantomData)
    }

    pub fn pairwise_max(&self, other: &Self) -> Self {
        Self(
            self.0.max(other.0),
            self.1.max(other.1),
            self.2.max(other.2),
            PhantomData,
        )
    }

    pub fn pairwise_min(&self, other: &Self) -> Self {
        Self(
            self.0.min(other.0),
            self.1.min(other.1),
            self.2.min(other.2),
            PhantomData,
        )
    }
}

impl<T> Index<usize> for State<T> {
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

impl<T> IndexMut<usize> for State<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.0,
            1 => &mut self.1.x,
            2 => &mut self.1.y,
            3 => &mut self.1.z,
            4 => &mut self.2,
            _ => panic!("Index out of bounds for StateVector!"),
        }
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub struct Gradients<T>(DVec3, [DVec3; 3], DVec3, PhantomData<T>);

impl<T> Gradients<T> {
    pub fn zeros() -> Self {
        Gradients(DVec3::ZERO, [DVec3::ZERO; 3], DVec3::ZERO, PhantomData)
    }

    pub fn dot(&self, dx: DVec3) -> State<T> {
        State::<T>(
            self.0.dot(dx),
            DVec3 {
                x: self.1[0].dot(dx),
                y: self.1[1].dot(dx),
                z: self.1[2].dot(dx),
            },
            self.2.dot(dx),
            PhantomData,
        )
    }

    pub fn is_finite(&self) -> bool {
        self.0.is_finite()
            && self.1[0].is_finite()
            && self.1[1].is_finite()
            && self.1[2].is_finite()
            && self.2.is_finite()
    }
}

impl Gradients<Primitive> {
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

impl<T> Index<usize> for Gradients<T> {
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

impl<T> IndexMut<usize> for Gradients<T> {
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

impl<T> AddAssign for Gradients<T> {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
        self.1[0] += rhs.1[0];
        self.1[1] += rhs.1[1];
        self.1[2] += rhs.1[2];
        self.2 += rhs.2;
    }
}

impl State<Primitive> {
    pub fn new(density: f64, velocity: DVec3, pressure: f64) -> Self {
        Self(density, velocity, pressure, PhantomData)
    }

    pub fn density(&self) -> f64 {
        self.0
    }

    pub fn velocity(&self) -> DVec3 {
        self.1
    }

    pub fn pressure(&self) -> f64 {
        self.2
    }

    pub fn from_conserved(conserved: &State<Conserved>, volume: f64, eos: &GasLaw) -> Self {
        if conserved.mass() > 0. {
            let m_inv = 1. / conserved.mass();
            let density = conserved.mass() / volume;
            let velocity = conserved.momentum() * m_inv;
            let internal_energy = conserved.internal_energy();
            let pressure = eos.gas_pressure_from_internal_energy(internal_energy, density);
            assert!(density >= 0.);
            assert!(pressure >= 0.);
            Self::new(density, velocity, pressure)
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

    /// Reflect velocity component along normal
    pub fn reflect(&self, normal: DVec3) -> Self {
        let v = self.velocity() - 2. * self.velocity().dot(normal) * normal;
        Self::new(self.density(), v, self.pressure())
    }

    /// Resets unphysical values to vacuum and prints an error.
    pub fn check_physical(&mut self) {
        if self.density() < 0. {
            eprintln!("Negative density encountered, resetting to vacuum!");
            *self = Self::vacuum();
        }
        if self.pressure() < 0. {
            eprintln!("Negative pressure encountered, resetting to vacuum!");
            *self = Self::vacuum();
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

impl State<Conserved> {
    pub fn new(mass: f64, momentum: DVec3, energy: f64) -> Self {
        Self(mass, momentum, energy, PhantomData)
    }

    pub fn mass(&self) -> f64 {
        self.0
    }

    pub fn momentum(&self) -> DVec3 {
        self.1
    }

    pub fn energy(&self) -> f64 {
        self.2
    }

    /// returns the specific internal energy e defined by: E = E_kin + E_therm = E_kin + m * e
    pub fn internal_energy(&self) -> f64 {
        let m_inv = 1. / self.mass();
        let thermal_energy = self.energy() - 0.5 * self.momentum().length_squared() * m_inv;
        thermal_energy * m_inv
    }

    pub fn from_primitives(primitives: &State<Primitive>, volume: f64, eos: &GasLaw) -> Self {
        let mass = primitives.density() * volume;
        let momentum = mass * primitives.velocity();
        let energy = 0.5 * momentum.dot(primitives.velocity())
            + mass
                * eos.gas_internal_energy_from_pressure(
                    primitives.pressure(),
                    1. / primitives.density(),
                );
        Self::new(mass, momentum, energy)
    }
}

#[cfg(test)]
mod test {
    use float_cmp::assert_approx_eq;
    use glam::DVec3;

    use super::{Conserved, Primitive};
    use crate::{gas_law::{EquationOfState, GasLaw}, physical_quantities::State};

    #[test]
    fn test_conversions() {
        let primitives = State::<Primitive>::new(
            0.75,
            DVec3 {
                x: 0.4,
                y: 0.,
                z: 0.,
            },
            0.8,
        );
        let volume = 0.1;
        let eos = GasLaw::new(5. / 3., EquationOfState::Ideal);
        let conserved = State::<Conserved>::from_primitives(&primitives, volume, &eos);
        let primitives_new = State::<Primitive>::from_conserved(&conserved, volume, &eos);

        assert_approx_eq!(f64, primitives.density(), primitives_new.density());
        assert_approx_eq!(f64, primitives.velocity().x, primitives_new.velocity().x);
        assert_approx_eq!(f64, primitives.velocity().y, primitives_new.velocity().y);
        assert_approx_eq!(f64, primitives.velocity().z, primitives_new.velocity().z);
        assert_approx_eq!(f64, primitives.pressure(), primitives_new.pressure());
    }
}
