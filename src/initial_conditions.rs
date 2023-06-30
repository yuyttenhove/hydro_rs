use std::{error::Error, path::Path};

use glam::DVec3;
use yaml_rust::Yaml;

use meshless_voronoi::VoronoiIntegrator;
use meshless_voronoi::integrals::VolumeIntegral;

use crate::{
    equation_of_state::EquationOfState,
    errors::ConfigError,
    macros::{create_attr, create_dataset},
    part::Particle,
    physical_quantities::Conserved,
    utils::{HydroDimension, HydroDimension::*},
};

macro_rules! conv1d {
    ($x:expr, $density:expr, $velocity:expr, $pressure:expr, $box_size:expr, $num_part_inv: expr, $eos:expr) => {
        Particle::from_ic(
            DVec3 {
                x: $x,
                y: 0.,
                z: 0.,
            },
            $density * $box_size.x * $box_size.y * $box_size.z * $num_part_inv,
            DVec3 {
                x: $velocity,
                y: 0.,
                z: 0.,
            },
            $eos.gas_internal_energy_from_pressure($pressure, 1. / $density),
        )
    };
}

macro_rules! cfg_ics2vec {
    ($yaml:expr, $prop:expr, $count:expr) => {
        match $yaml[$prop].as_vec() {
            Some(arr) if arr.len() == $count => arr
                .into_iter()
                .map(|y| match y {
                    Yaml::Real(s) => Ok(s.parse::<f64>().unwrap()),
                    Yaml::Integer(i) => Ok(*i as f64),
                    _ => Err(ConfigError::InvalidArrayFormat($yaml[$prop].clone())),
                })
                .collect::<Result<Vec<_>, _>>(),
            Some(arr) => Err(ConfigError::InvalidArrayLength($count, arr.len())),
            None => Err(ConfigError::MissingParameter(format!(
                "initial_conditions:particles:{}",
                $prop
            ))),
        }
    };
}

fn sod_shock(num_part: usize, box_size: DVec3, eos: &EquationOfState) -> Vec<Particle> {
    let mut ic = Vec::<Particle>::with_capacity(num_part);
    let num_part_inv = 1. / (num_part as f64);

    for idx in 0..num_part {
        let x = (idx as f64 + 0.5) * num_part_inv;
        let density = if idx < num_part / 2 { 1. } else { 0.125 };
        let velocity = 0.;
        let pressure = if idx < num_part / 2 { 1. } else { 0.1 };

        ic.push(conv1d!(
            x * box_size.x,
            density,
            velocity,
            pressure,
            box_size,
            num_part_inv,
            eos
        ));
    }

    ic
}

fn noh(num_part: usize, box_size: DVec3, eos: &EquationOfState) -> Vec<Particle> {
    let mut ic = Vec::<Particle>::with_capacity(num_part);
    let num_part_inv = 1. / (num_part as f64);

    for idx in 0..num_part {
        let x = (idx as f64 + 0.5) * num_part_inv;
        let density = 1.;
        let velocity = if x < 0.5 { 1. } else { -1. };
        let pressure = 1.0e-6;
        ic.push(conv1d!(
            x * box_size.x,
            density,
            velocity,
            pressure,
            box_size,
            num_part_inv,
            eos
        ));
    }

    ic
}

fn toro(num_part: usize, box_size: DVec3, eos: &EquationOfState) -> Vec<Particle> {
    let mut ic = Vec::<Particle>::with_capacity(num_part);
    let num_part_inv = 1. / (num_part as f64);

    for idx in 0..num_part {
        let x = (idx as f64 + 0.5) * num_part_inv;
        let density = 1.;
        let velocity = if x < 0.5 { 2. } else { -2. };
        let pressure = 0.4;
        ic.push(conv1d!(
            x * box_size.x,
            density,
            velocity,
            pressure,
            box_size,
            num_part_inv,
            eos
        ));
    }

    ic
}

fn vacuum(num_part: usize, box_size: DVec3, eos: &EquationOfState) -> Vec<Particle> {
    let mut ic = Vec::<Particle>::with_capacity(num_part);
    let num_part_inv = 1. / (num_part as f64);

    for idx in 0..num_part {
        let x = (idx as f64 + 0.5) * num_part_inv;
        let density = if x < 0.5 { 1. } else { 0. };
        let velocity = 0.;
        let pressure = if x < 0.5 { 1. } else { 0. };
        ic.push(conv1d!(
            x * box_size.x,
            density,
            velocity,
            pressure,
            box_size,
            num_part_inv,
            eos
        ));
    }

    ic
}

/// Generates a spherically symmetric 1/r density profile
fn evrard(num_part: usize, box_size: DVec3, eos: &EquationOfState) -> Vec<Particle> {
    let mut ic = Vec::<Particle>::with_capacity(num_part);
    let num_part_inv = 1. / (num_part as f64);

    let u0 = 0.05;
    let mut m_tot = 0.;
    for idx in 0..num_part {
        let x = (idx as f64 + 0.5) * num_part_inv;
        let density = 1. / x;
        let r_in = idx as f64 * num_part_inv;
        let r_out = (idx as f64 + 1.) * num_part_inv;
        m_tot += density * 4. * std::f64::consts::FRAC_PI_3 * (r_out.powi(3) - r_in.powi(3));
        let velocity = 0.;
        let pressure = eos.gas_pressure_from_internal_energy(u0, density);
        ic.push(conv1d!(
            x * box_size.x,
            density,
            velocity,
            pressure,
            box_size,
            num_part_inv,
            eos
        ));
    }

    let correction = 1. / m_tot;
    for part in ic.iter_mut() {
        part.conserved = Conserved::new(
            correction * part.conserved.mass(),
            part.conserved.momentum(),
            part.conserved.energy(),
        );
    }

    ic
}

fn constant(num_part: usize, box_size: DVec3, eos: &EquationOfState) -> Vec<Particle> {
    let mut ic = Vec::<Particle>::with_capacity(num_part);
    let num_part_inv = 1. / (num_part as f64);

    for idx in 0..num_part {
        let x = (idx as f64 + 0.5) * num_part_inv;
        let density = 1.;
        let velocity = 0.;
        let pressure = 1.;
        ic.push(conv1d!(
            x * box_size.x,
            density,
            velocity,
            pressure,
            box_size,
            num_part_inv,
            eos
        ));
    }

    ic
}

fn square(num_part: usize, box_size: DVec3, eos: &EquationOfState) -> Vec<Particle> {
    let mut ic = Vec::<Particle>::with_capacity(num_part);
    let num_part_left = (0.8 * num_part as f64) as usize;
    let num_part_right = num_part - num_part_left;

    let density_low = 1.;
    let density_high = 4.;
    let square = |x: f64| {
        if x < 0.25 || x > 0.75 {
            density_low
        } else {
            density_high
        }
    };

    // left particles at four times the number density
    let num_part_left_inv = 0.5 / (num_part_left as f64);
    for idx in 0..num_part_left {
        let x = (idx as f64 + 0.5) * num_part_left_inv;
        let density = square(x);
        let velocity = 1.;
        let pressure = 1.;
        ic.push(conv1d!(
            x * box_size.x,
            density,
            velocity,
            pressure,
            box_size
                * DVec3 {
                    x: 0.5,
                    y: 1.,
                    z: 1.
                },
            num_part_left_inv,
            eos
        ));
    }

    // right particles
    let num_part_right_inv = 0.5 / (num_part_right as f64);
    for idx in 0..num_part_right {
        let x = 0.5 + (idx as f64 + 0.5) * num_part_right_inv;
        let density = square(x);
        let velocity = 1.;
        let pressure = 1.;
        ic.push(conv1d!(
            x * box_size.x,
            density,
            velocity,
            pressure,
            box_size
                * DVec3 {
                    x: 0.5,
                    y: 1.,
                    z: 1.
                },
            num_part_right_inv,
            eos
        ));
    }

    ic
}

fn read_parts_from_cfg(ic_cfg: &Yaml, num_part: usize) -> Result<Vec<Particle>, ConfigError> {
    let positions = cfg_ics2vec!(ic_cfg, "x", num_part)?;
    let masses = cfg_ics2vec!(ic_cfg, "mass", num_part)?;
    let velocities = cfg_ics2vec!(ic_cfg, "velocity", num_part)?;
    let internal_energy = cfg_ics2vec!(ic_cfg, "internal_energy", num_part)?;

    let mut ic = Vec::with_capacity(num_part);
    for i in 0..num_part {
        ic.push(Particle::from_ic(
            DVec3 {
                x: positions[i],
                y: 0.,
                z: 0.,
            },
            masses[i],
            velocities[i] * DVec3::X,
            internal_energy[i],
        ))
    }
    Ok(ic)
}

pub struct HydroICBuilder {
    coordinates: Vec<DVec3>,
    masses: Vec<f64>,
    velocities: Vec<DVec3>,
    internal_energies: Vec<f64>,
    volumes: Option<Vec<f64>>,
    gamma: f64,
    num_part: usize,
    periodic: bool,
    dimension: usize,
    box_size: DVec3,
}

impl HydroICBuilder {
    pub fn empty(
        num_part: usize,
        dimension: usize,
        box_size: DVec3,
        periodic: Option<bool>,
        gamma: Option<f64>,
    ) -> Self {
        let gamma = gamma.unwrap_or(5. / 3.);
        let periodic = periodic.unwrap_or(true);
        Self {
            coordinates: vec![],
            masses: vec![],
            velocities: vec![],
            internal_energies: vec![],
            volumes: None,
            gamma,
            num_part,
            periodic,
            dimension,
            box_size,
        }
    }

    fn init_volumes(&mut self) {
        self.volumes.get_or_insert_with(|| {
            assert_eq!(
                self.coordinates.len(),
                self.num_part,
                "Invalid lenght of coordinates!"
            );
            VoronoiIntegrator::build(
                &self.coordinates,
                None,
                DVec3::ZERO,
                self.box_size,
                self.dimension,
                self.periodic,
            )
            .compute_cell_integrals::<VolumeIntegral>()
            .iter()
            .map(|int| int.volume)
            .collect::<Vec<_>>()
        });
    }

    fn get_volumes(&self) -> &[f64] {
        self.volumes.as_ref().unwrap()
    }

    pub fn set_coordinates(mut self, coordinates: Vec<DVec3>) -> Self {
        self.coordinates = coordinates;
        self
    }

    pub fn set_masses(mut self, masses: Vec<f64>) -> Self {
        self.masses = masses;
        self
    }

    pub fn set_internal_energies(mut self, internal_energies: Vec<f64>) -> Self {
        self.internal_energies = internal_energies;
        self
    }

    pub fn set_densities(mut self, densities: &[f64]) -> Self {
        // Set masses
        self.masses = densities
            .iter()
            .zip(self.get_volumes().iter())
            .map(|(rho, vol)| rho * vol)
            .collect();

        self
    }

    pub fn set_pressures(mut self, pressures: &[f64]) -> Self {
        assert_eq!(
            self.masses.len(),
            self.num_part,
            "Invalid lenght of masses!"
        );
        assert_eq!(
            pressures.len(),
            self.num_part,
            "Invalid lenght of pressures!"
        );

        self.init_volumes();
        let inv_densities = self
            .masses
            .iter()
            .zip(self.get_volumes().iter())
            .map(|(m, vol)| vol / m);
        let odgm1 = 1. / (self.gamma - 1.);
        self.internal_energies = pressures
            .iter()
            .zip(inv_densities)
            .map(|(pres, rho_inv)| *pres * rho_inv * odgm1)
            .collect();

        self
    }

    pub fn set_momenta(mut self, momenta: &[DVec3]) -> Self {
        assert_eq!(
            self.masses.len(),
            self.num_part,
            "Invalid lenght of masses!"
        );
        assert_eq!(momenta.len(), self.num_part, "Invalid lenght of momenta!");

        self.velocities = self
            .masses
            .iter()
            .zip(momenta.iter())
            .map(|(m, p)| *p / *m)
            .collect();
        self
    }

    pub fn save<P: AsRef<Path>>(&self, save_name: P) -> Result<(), hdf5::Error> {
        let file = hdf5::File::create(save_name)?;

        // Write header
        let header = file.create_group("Header")?;
        create_attr!(header, self.box_size.to_array(), "BoxSize")?;
        create_attr!(header, [self.num_part, 0, 0, 0, 0], "Numpart_Total")?;
        create_attr!(header, [self.dimension], "Dimension")?;

        // Write particle data
        let part_data = file.create_group("PartType0")?;
        create_dataset!(
            part_data,
            self.coordinates.iter().map(|x| x.to_array()),
            "Coordinates"
        )?;
        part_data
            .new_dataset_builder()
            .with_data(&self.masses)
            .create("Masses")?;
        create_dataset!(
            part_data,
            self.velocities.iter().map(|v| v.to_array()),
            "Velocities"
        )?;
        part_data
            .new_dataset_builder()
            .with_data(&self.internal_energies)
            .create("InternalEnergy")?;

        Ok(())
    }
}

pub struct InitialConditions {
    particles: Vec<Particle>,
    box_size: DVec3,
    dimensionality: HydroDimension,
}

impl InitialConditions {
    pub fn new(ic_cfg: &Yaml, eos: &EquationOfState) -> Result<Self, Box<dyn Error>> {
        let kind = ic_cfg["type"]
            .as_str()
            .ok_or(ConfigError::MissingParameter(
                "initial_conditions:type".to_string(),
            ))?
            .to_string();

        match kind.as_str() {
            "file" => {
                let filename = ic_cfg["filename"]
                    .as_str()
                    .ok_or(ConfigError::MissingParameter(
                        "initial_conditions:filename".to_string(),
                    ))?
                    .to_string();
                Self::from_hdf5(filename)
            }
            _ => {
                let num_part = ic_cfg["num_part"].as_i64().unwrap_or(100) as usize;
                let box_size = ic_cfg["box_size"].as_vec().map_or(
                    Ok(DVec3::from_array([1., 1., 1.])),
                    |v| match &v[..] {
                        [Yaml::Real(a), Yaml::Real(b), Yaml::Real(c)] => Ok(DVec3 {
                            x: a.parse().unwrap(),
                            y: b.parse().unwrap(),
                            z: c.parse().unwrap(),
                        }),
                        _ => Err(ConfigError::IllegalBoxSize(format!(
                            "{:?}",
                            ic_cfg["box_size"]
                        ))),
                    },
                )?;

                let parts = match kind.as_str() {
                    "config" => read_parts_from_cfg(&ic_cfg["particles"], num_part),
                    "sodshock" => Ok(sod_shock(num_part, box_size, eos)),
                    "noh" => Ok(noh(num_part, box_size, eos)),
                    "toro" => Ok(toro(num_part, box_size, eos)),
                    "vacuum" => Ok(vacuum(num_part, box_size, eos)),
                    "evrard" => Ok(evrard(num_part, box_size, eos)),
                    "constant" => Ok(constant(num_part, box_size, eos)),
                    "square" => Ok(square(num_part, box_size, eos)),
                    _ => Err(ConfigError::UnknownICs(kind.to_string())),
                }?;

                Ok(Self {
                    particles: parts,
                    box_size,
                    dimensionality: HydroDimension1D,
                })
            }
        }
    }

    fn from_hdf5<P: AsRef<Path>>(filename: P) -> Result<Self, Box<dyn Error>> {
        let file = hdf5::File::open(filename)?;

        // Read some properties from the file
        let header = file.group("Header")?;
        let num_parts = header.attr("NumPart_Total")?.read_raw::<usize>()?[0];
        let dimension = header.attr("Dimension")?.read_raw::<usize>()?[0];
        let mut box_size = header.attr("BoxSize")?.read_raw::<f64>()?;
        for i in dimension..3 {
            if box_size.len() < i + 1 {
                box_size.push(1.)
            } else {
                box_size[i] = 1.;
            }
        }
        let box_size = DVec3::from_slice(&box_size);

        // Read the particle data from the file
        let data = file.group("PartType0")?;
        let coordinates = data.dataset("Coordinates")?.read_raw::<f64>()?;
        let masses = data.dataset("Masses")?.read_raw::<f64>()?;
        let velocities = data.dataset("Velocities")?.read_raw::<f64>()?;
        let internal_energy = data.dataset("InternalEnergy")?.read_raw::<f64>()?;

        file.close()?;

        // Construct the actual particles
        assert_eq!(
            3 * num_parts,
            coordinates.len(),
            "Incorrect lenght of coordinates vector!"
        );
        assert_eq!(
            num_parts,
            masses.len(),
            "Incorrect lenght of masses vector!"
        );
        assert_eq!(
            3 * num_parts,
            velocities.len(),
            "Incorrect lenght of velocities vector!"
        );
        assert_eq!(
            num_parts,
            internal_energy.len(),
            "Incorrect lenght of internal energies vector!"
        );
        let mut parts = Vec::with_capacity(num_parts);
        for i in 0..num_parts {
            let x = DVec3::from_slice(&coordinates[3 * i..3 * i + 3]);
            let velocity = DVec3::from_slice(&velocities[3 * i..3 * i + 3]);
            parts.push(Particle::from_ic(
                x,
                masses[i],
                velocity,
                internal_energy[i],
            ));
        }

        Ok(Self {
            particles: parts,
            box_size,
            dimensionality: dimension.into(),
        })
    }

    /// Create an initial condition from a mapper that maps a particle's `position` to its `(density, velocity, pressure)`.
    pub fn from_fn<F>(
        mut box_size: DVec3,
        num_part: usize,
        dimensionality: usize,
        eos: &EquationOfState,
        mut mapper: F,
    ) -> Self
    where
        F: FnMut(DVec3) -> (f64, DVec3, f64),
    {
        if dimensionality < 2 {
            box_size.y = 1.;
        }
        if dimensionality < 3 {
            box_size.z = 1.;
        }

        let volume = box_size.x * box_size.y * box_size.z;
        let num_part_per_unit_length = (num_part as f64 * volume).powf(1. / dimensionality as f64);
        let num_part_x = (box_size.x * num_part_per_unit_length) as usize;
        let num_part_y = if dimensionality > 1 {
            (box_size.y * num_part_per_unit_length) as usize
        } else {
            1
        };
        let num_part_z = if dimensionality > 2 {
            (box_size.z * num_part_per_unit_length) as usize
        } else {
            1
        };
        let num_part = num_part_x * num_part_y * num_part_z;
        let volume_per_part = volume / num_part as f64;

        let particles = (0..num_part)
            .map(|n| {
                let i = n / (num_part_y * num_part_z);
                let j = (n - i * num_part_y * num_part_z) / num_part_z;
                let k = n - i * num_part_y * num_part_z - j * num_part_z;

                let position = DVec3 {
                    x: (i as f64 + 0.5) / num_part_x as f64 * box_size.x,
                    y: (j as f64 + 0.5) / num_part_y as f64 * box_size.y,
                    z: (k as f64 + 0.5) / num_part_z as f64 * box_size.z,
                };

                let (density, velocity, pressure) = mapper(position);

                Particle::from_ic(
                    position,
                    density * volume_per_part,
                    velocity,
                    eos.gas_internal_energy_from_pressure(pressure, 1. / density),
                )
            })
            .collect();

        Self {
            particles,
            box_size,
            dimensionality: dimensionality.into(),
        }
    }

    pub fn box_size(&self) -> DVec3 {
        self.box_size
    }

    pub fn into_parts(self) -> Vec<Particle> {
        self.particles
    }

    pub fn dimensionality(&self) -> HydroDimension {
        self.dimensionality
    }
}

impl From<HydroICBuilder> for InitialConditions {
    fn from(ic: HydroICBuilder) -> Self {
        let particles = (0..ic.num_part)
            .map(|i| {
                Particle::from_ic(
                    ic.coordinates[i],
                    ic.masses[i],
                    ic.velocities[i],
                    ic.internal_energies[i],
                )
            })
            .collect();
        Self {
            particles,
            box_size: ic.box_size,
            dimensionality: ic.dimension.into(),
        }
    }
}
