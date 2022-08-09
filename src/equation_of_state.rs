#[derive(Clone, Copy)]
pub enum EquationOfState {
    Ideal { gamma: f64 },
    Isothermal,
}

impl EquationOfState {
    pub fn gas_energy_from_pressure(&self, pressure: f64, volume: f64) -> f64 {
        match self {
            EquationOfState::Ideal { gamma } => pressure * volume / (gamma - 1.),
            _ => todo!()
        }
    }

    pub fn gas_pressure_from_energy(&self, energy: f64, volume: f64) -> f64 {
        match self {
            EquationOfState::Ideal { gamma } => (gamma - 1.) * energy / volume,
            _ => todo!()
        }
    }

    pub fn sound_speed(&self, pressure: f64, density_inv: f64) -> f64 {
        match self {
            EquationOfState::Ideal { gamma } => (gamma * pressure * density_inv).sqrt(),
            _ => unimplemented!()
        }
    }

    pub fn gas_entropy_from_internal_energy(&self, internal_energy: f64, density: f64) -> f64 {
        match self {
            EquationOfState::Ideal { gamma } => (gamma - 1.) * internal_energy * density.powf(1. - gamma),
            _ => unimplemented!()
        }
    }
}