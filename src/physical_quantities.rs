use std::ops::{Add, AddAssign, Mul, SubAssign};

use crate::equation_of_state::EquationOfState;


#[derive(Default, Debug, Clone, Copy)]
struct Vec3f64(f64, f64, f64);

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
}

#[derive(Default, Debug, Clone, Copy, Add, AddAssign, SubAssign)]
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
        Primitives { values: Vec3f64::zeros() }
    }

    pub fn new(density: f64, velocity: f64, pressure: f64) -> Self {
        Primitives { values: Vec3f64(density, velocity, pressure) }
    }

    pub fn from_conserved(conserved: &Conserved, volume: f64, eos: EquationOfState) -> Self {
        if conserved.mass() > 0. {
            let density = conserved.mass() / volume;
            let velocity = conserved.momentum() / conserved.mass();
            let pressure = eos.gas_pressure_from_energy(
                conserved.energy() - 0.5 * conserved.momentum() * velocity, volume);
            Self { values: Vec3f64(density, velocity, pressure)}
        } else {
            Self::vacuum()
        }
    }

    pub fn boost(&self, velocity: f64) -> Self {
        Self::new(self.density(), self.velocity() + velocity, self.pressure())
    }
}

impl Mul<Primitives> for f64 {
    type Output = Primitives;

    fn mul(self, rhs: Primitives) -> Self::Output {
        Primitives { values: self * rhs.values }
    }
}

#[derive(Default, Debug, Clone, Copy, Add, AddAssign, SubAssign)]
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
        Conserved { values: Vec3f64::zeros() }
    }

    pub fn new(mass: f64, momentum: f64, energy: f64) -> Self {
        Conserved { values: Vec3f64(mass, momentum, energy) }
    }

    pub fn from_primitives(primitives: &Primitives, volume: f64, eos: EquationOfState) -> Self {
        let mass = primitives.density() * volume;
        let momentum = mass * primitives.velocity();
        let energy = 0.5 * momentum * primitives.velocity() 
                          + eos.gas_energy_from_pressure(primitives.pressure(), volume);
        Self { values: Vec3f64(mass, momentum, energy) }
    }
}

impl Mul<Conserved> for f64 {
    type Output = Conserved;

    fn mul(self, rhs: Conserved) -> Self::Output {
        Conserved { values: self * rhs.values }
    }
}


#[cfg(test)]
mod test {
    use crate::equation_of_state::EquationOfState;
    use crate::utils::Round;

    use super::{Primitives, Conserved};


    #[test]
    fn test_conversions() {
        let primitives = Primitives::new(0.75, 0.4, 0.8);
        let volume = 0.1;
        let eos = EquationOfState::Ideal { gamma: 5. / 3. };
        let conserved = Conserved::from_primitives(&primitives, volume, eos);
        let primitives_new = Primitives::from_conserved(&conserved, volume, eos);

        assert_eq!(primitives.density().round_to(15), primitives_new.density().round_to(15));
        assert_eq!(primitives.velocity().round_to(15), primitives_new.velocity().round_to(15));
        assert_eq!(primitives.pressure().round_to(15), primitives_new.pressure().round_to(15));
    }
}