#[derive(Debug, Default, Clone, Copy)]
pub struct AdiabaticIndex {
    gamma: f64,
    gamma_inv: f64,
    odgm1: f64,
    odgp1: f64,
}

impl From<f64> for AdiabaticIndex {
    fn from(value: f64) -> Self {
        AdiabaticIndex {
            gamma: value,
            gamma_inv: 1. / value,
            odgm1: 1. / (value - 1.),
            odgp1: 1. / (value + 1.),
        }
    }
}

impl From<AdiabaticIndex> for f64 {
    fn from(value: AdiabaticIndex) -> Self {
        value.gamma
    }
}

impl AdiabaticIndex {
    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    pub fn gp1dg(&self) -> f64 {
        (self.gamma + 1.) * self.gamma_inv
    }

    pub fn gm1d2g(&self) -> f64 {
        0.5 * (self.gamma - 1.) * self.gamma_inv
    }

    pub fn gm1dgp1(&self) -> f64 {
        (self.gamma - 1.) * self.odgp1
    }

    pub fn odgm1(&self) -> f64 {
        self.odgm1
    }

    pub fn tdgm1(&self) -> f64 {
        2. * self.odgm1
    }

    pub fn tdgp1(&self) -> f64 {
        2. * self.odgp1
    }
}

#[derive(Debug, Clone, Copy)]
pub enum EquationOfState {
    Ideal,
    Isothermal { isothermal_internal_energy: f64 },
}

#[derive(Debug, Clone, Copy)]
pub struct GasLaw {
    gamma: AdiabaticIndex,
    eos: EquationOfState,
}

impl GasLaw {
    pub fn new(gamma: f64, eos: EquationOfState) -> Self {
        Self {
            gamma: gamma.into(),
            eos,
        }
    }

    pub fn gamma(&self) -> &AdiabaticIndex {
        &self.gamma
    }

    /// Specific internal energy
    pub fn gas_internal_energy_from_pressure(&self, pressure: f64, density_inv: f64) -> f64 {
        match self.eos {
            EquationOfState::Ideal => pressure * density_inv * self.gamma.odgm1(),
            EquationOfState::Isothermal {
                isothermal_internal_energy,
            } => isothermal_internal_energy,
        }
    }

    pub fn gas_pressure_from_internal_energy(&self, internal_energy: f64, density: f64) -> f64 {
        match self.eos {
            EquationOfState::Ideal => (self.gamma.gamma - 1.) * internal_energy * density,
            EquationOfState::Isothermal {
                isothermal_internal_energy,
            } => (self.gamma.gamma - 1.) * isothermal_internal_energy * density,
        }
    }

    pub fn sound_speed(&self, pressure: f64, density_inv: f64) -> f64 {
        match self.eos {
            EquationOfState::Ideal => (self.gamma.gamma * pressure * density_inv).sqrt(),
            EquationOfState::Isothermal {
                isothermal_internal_energy,
            } => f64::sqrt(isothermal_internal_energy * self.gamma.gamma * (self.gamma.gamma - 1.)),
        }
    }

    pub fn gas_entropy_from_internal_energy(&self, internal_energy: f64, density: f64) -> f64 {
        match self.eos {
            EquationOfState::Ideal => {
                (self.gamma.gamma - 1.) * internal_energy * density.powf(1. - self.gamma.gamma)
            }
            EquationOfState::Isothermal {
                isothermal_internal_energy,
            } => {
                (self.gamma.gamma - 1.)
                    * isothermal_internal_energy
                    * density.powf(1. - self.gamma.gamma)
            }
        }
    }
}
