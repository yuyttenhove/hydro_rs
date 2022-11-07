use std::ops::{Add, AddAssign, Index, Mul, Sub, SubAssign};

use crate::equation_of_state::EquationOfState;

#[derive(Default, Debug, Clone, Copy)]
pub struct Vec3f64(f64, f64, f64);

impl Add for Vec3f64 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Vec3f64(self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2)
    }
}

impl AddAssign for Vec3f64 {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
        self.1 += rhs.1;
        self.2 += rhs.2;
    }
}

impl Sub for Vec3f64 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Vec3f64(self.0 - rhs.0, self.1 - rhs.1, self.2 - rhs.2)
    }
}

impl SubAssign for Vec3f64 {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
        self.1 -= rhs.1;
        self.2 -= rhs.2;
    }
}

impl Mul<Vec3f64> for f64 {
    type Output = Vec3f64;

    fn mul(self, rhs: Vec3f64) -> Self::Output {
        Vec3f64(self * rhs.0, self * rhs.1, self * rhs.2)
    }
}

impl Vec3f64 {
    pub fn zeros() -> Self {
        Vec3f64(0., 0., 0.)
    }

    pub fn pairwise_max(&self, other: Self) -> Self {
        Vec3f64(
            self.0.max(other.0),
            self.1.max(other.1),
            self.2.max(other.2),
        )
    }

    pub fn pairwise_min(&self, other: Self) -> Self {
        Vec3f64(
            self.0.min(other.0),
            self.1.min(other.1),
            self.2.min(other.2),
        )
    }
}

impl Index<usize> for Vec3f64 {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.0,
            1 => &self.1,
            2 => &self.2,
            _ => panic!("Index out of bounds for Vec3f64!"),
        }
    }
}

#[derive(Default, Debug, Clone, Copy, Add, Sub, AddAssign, SubAssign)]
pub struct Primitives {
    values: Vec3f64,
}

impl Primitives {
    pub fn density(&self) -> f64 {
        self.values.0
    }

    pub fn velocity(&self) -> f64 {
        self.values.1
    }

    pub fn pressure(&self) -> f64 {
        self.values.2
    }

    pub fn vacuum() -> Self {
        Primitives {
            values: Vec3f64::zeros(),
        }
    }

    pub fn new(density: f64, velocity: f64, pressure: f64) -> Self {
        Primitives {
            values: Vec3f64(density, velocity, pressure),
        }
    }

    pub fn from_conserved(conserved: &Conserved, volume: f64, eos: EquationOfState) -> Self {
        if conserved.mass() > 0. {
            let density = conserved.mass() / volume;
            let velocity = conserved.momentum() / conserved.mass();
            let internal_energy = conserved.energy() - 0.5 * conserved.momentum() * velocity;
            let pressure = eos.gas_pressure_from_internal_energy(internal_energy, volume);
            Self {
                values: Vec3f64(density, velocity, pressure),
            }
        } else {
            Self::vacuum()
        }
    }

    pub fn boost(&self, velocity: f64) -> Self {
        if self.density() > 0. {
            Self::new(self.density(), self.velocity() + velocity, self.pressure())
        } else {
            debug_assert_eq!(self.velocity(), 0.);
            debug_assert_eq!(self.velocity(), 0.);
            *self
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

    pub fn values(&self) -> Vec3f64 {
        self.values
    }

    pub fn check_physical(&mut self) {
        if self.density() < 0. {
            eprintln!("Negative density encountered, resetting to vacuum!");
            self.values = Vec3f64::zeros();
        }
        if self.pressure() < 0. {
            eprintln!("Negative density encountered, resetting to vacuum!");
            self.values = Vec3f64::zeros();
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
    values: Vec3f64,
}

impl Conserved {
    pub fn mass(&self) -> f64 {
        self.values.0
    }

    pub fn momentum(&self) -> f64 {
        self.values.1
    }

    pub fn energy(&self) -> f64 {
        self.values.2
    }

    pub fn vacuum() -> Self {
        Conserved {
            values: Vec3f64::zeros(),
        }
    }

    pub fn new(mass: f64, momentum: f64, energy: f64) -> Self {
        Conserved {
            values: Vec3f64(mass, momentum, energy),
        }
    }

    pub fn from_primitives(primitives: &Primitives, volume: f64, eos: EquationOfState) -> Self {
        let mass = primitives.density() * volume;
        let momentum = mass * primitives.velocity();
        let energy = 0.5 * momentum * primitives.velocity()
            + eos.gas_internal_energy_from_pressure(primitives.pressure(), volume);
        Self {
            values: Vec3f64(mass, momentum, energy),
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
    use crate::equation_of_state::EquationOfState;
    use crate::utils::Round;

    use super::{Conserved, Primitives};

    #[test]
    fn test_conversions() {
        let primitives = Primitives::new(0.75, 0.4, 0.8);
        let volume = 0.1;
        let eos = EquationOfState::Ideal { gamma: 5. / 3. };
        let conserved = Conserved::from_primitives(&primitives, volume, eos);
        let primitives_new = Primitives::from_conserved(&conserved, volume, eos);

        assert_eq!(
            primitives.density().round_to(15),
            primitives_new.density().round_to(15)
        );
        assert_eq!(
            primitives.velocity().round_to(15),
            primitives_new.velocity().round_to(15)
        );
        assert_eq!(
            primitives.pressure().round_to(15),
            primitives_new.pressure().round_to(15)
        );
    }
}
