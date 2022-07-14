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
}