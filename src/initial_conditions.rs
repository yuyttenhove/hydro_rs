use std::{error::Error, path::Path};

use glam::DVec3;
use rand::Rng;

use meshless_voronoi::integrals::VolumeIntegral;
use meshless_voronoi::VoronoiIntegrator;

use crate::{
    gas_law::GasLaw,
    macros::{create_attr, create_dataset},
    part::Particle,
    Dimensionality,
};

#[derive(Clone, Default)]
pub struct InitialConditions {
    pub coordinates: Vec<DVec3>,
    pub masses: Vec<f64>,
    pub velocities: Vec<DVec3>,
    pub internal_energies: Vec<f64>,
    pub num_part: usize,
    pub periodic: bool,
    pub dimensionality: Dimensionality,
    pub box_size: DVec3,
    volumes: Option<Vec<f64>>,
    smoothing_lengths: Option<Vec<f64>>,
    ids: Option<Vec<u64>>,
}

impl InitialConditions {
    pub fn empty(
        num_part: usize,
        dimension: Dimensionality,
        box_size: DVec3,
        periodic: bool,
    ) -> Self {
        Self {
            num_part,
            periodic,
            dimensionality: dimension
                .try_into()
                .expect("Dimension must be from 1 to 3!"),
            box_size,
            ..Self::default()
        }
    }

    pub fn from_hdf5<P: AsRef<Path>>(filename: P) -> Result<Self, Box<dyn Error>> {
        let file = hdf5::File::open(filename)?;

        // Read some properties from the file
        print!(" - Reading header...");
        let header = file.group("Header")?;
        let num_parts = header.attr("NumPart_Total")?.read_raw::<usize>()?[0];
        let dimension = header.attr("Dimension")?.read_raw::<usize>()?[0];
        let periodic = match header.attr("Periodic") {
            Ok(attr) => attr.read_raw::<bool>()?[0],
            Err(_) => false,
        };
        let mut box_size = header.attr("BoxSize")?.read_raw::<f64>()?;
        for i in dimension..3 {
            if box_size.len() < i + 1 {
                box_size.push(1.)
            } else {
                box_size[i] = 1.;
            }
        }
        let box_size = DVec3::from_slice(&box_size);
        let dimension = dimension.try_into().expect("Dimension must be <= 3!");
        println!("✅");

        // Read the particle data from the file
        print!(" - Reading raw particle data...");
        let data = file.group("PartType0")?;
        let coordinates = data.dataset("Coordinates")?.read_raw::<f64>()?;
        let masses = data.dataset("Masses")?.read_raw::<f64>()?;
        let velocities = data.dataset("Velocities")?.read_raw::<f64>()?;
        let internal_energy = data.dataset("InternalEnergy")?.read_raw::<f64>()?;
        println!("✅");

        file.close()?;

        // Construct the actual particles
        print!(" - Contructing particle ICs...");
        assert_eq!(
            3 * num_parts,
            coordinates.len(),
            "Incorrect length of coordinates vector!"
        );
        assert_eq!(
            num_parts,
            masses.len(),
            "Incorrect length of masses vector!"
        );
        assert_eq!(
            3 * num_parts,
            velocities.len(),
            "Incorrect length of velocities vector!"
        );
        assert_eq!(
            num_parts,
            internal_energy.len(),
            "Incorrect length of internal energies vector!"
        );
        let coordinates = coordinates
            .chunks(3)
            .map(|c| DVec3::from_slice(c))
            .collect();
        let velocities = velocities.chunks(3).map(|c| DVec3::from_slice(c)).collect();

        let ics = Self::empty(num_parts, dimension, box_size, periodic)
            .set_coordinates(coordinates)
            .set_masses(masses)
            .set_velocities(velocities)
            .set_internal_energies(internal_energy);

        Ok(ics)
    }

    /// Create an initial condition from a mapper that maps a particle's `position` to its `(density, velocity, pressure)`.
    /// The positions are normalized to the interval [0.; 1.] in all dimensions.
    pub fn from_fn<F>(
        mut box_size: DVec3,
        num_part: usize,
        dimensionality: Dimensionality,
        periodic: bool,
        eos: &GasLaw,
        perturbations: Option<f64>,
        mut mapper: F,
    ) -> Self
    where
        F: FnMut(DVec3) -> (f64, DVec3, f64),
    {
        let mut perturbation = DVec3::splat(perturbations.unwrap_or_default());
        let dimension: usize = dimensionality.into();
        if dimension < 2 {
            box_size.y = 1.;
            perturbation.y = 0.;
        }
        if dimension < 3 {
            box_size.z = 1.;
            perturbation.z = 0.;
        }
        let mut rng = rand::thread_rng();

        let volume = box_size.x * box_size.y * box_size.z;
        let num_part_per_unit_length = (num_part as f64 / volume).powf(1. / dimension as f64);
        let num_part_x = (box_size.x * num_part_per_unit_length).round() as usize;
        let num_part_y = if dimension > 1 {
            (box_size.y * num_part_per_unit_length).round() as usize
        } else {
            1
        };
        let num_part_z = if dimension > 2 {
            (box_size.z * num_part_per_unit_length).round() as usize
        } else {
            1
        };
        let num_part = num_part_x * num_part_y * num_part_z;
        let volume_per_part = volume / num_part as f64;

        let mut ics = Self::empty(num_part, dimensionality, box_size, periodic);
        for n in 0..num_part {
            let i = n / (num_part_y * num_part_z);
            let j = (n - i * num_part_y * num_part_z) / num_part_z;
            let k = n - i * num_part_y * num_part_z - j * num_part_z;

            let x_pert = perturbation.x * rng.gen::<f64>();
            let y_pert = perturbation.y * rng.gen::<f64>();
            let z_pert = perturbation.z * rng.gen::<f64>();

            let coordinate = DVec3 {
                x: (i as f64 + 0.5 + x_pert) / num_part_x as f64 * box_size.x,
                y: (j as f64 + 0.5 + y_pert) / num_part_y as f64 * box_size.y,
                z: (k as f64 + 0.5 + z_pert) / num_part_z as f64 * box_size.z,
            };

            let (density, velocity, pressure) = mapper(coordinate / box_size);
            ics.coordinates.push(coordinate);
            ics.masses.push(density * volume_per_part);
            ics.velocities.push(velocity);
            ics.internal_energies
                .push(eos.gas_internal_energy_from_pressure(pressure, 1. / density));
        }

        ics
    }

    fn postprocess_positions(&mut self) {
        let dimension_fac = match self.dimensionality {
            Dimensionality::OneD => DVec3::new(1., 0., 0.),
            Dimensionality::TwoD => DVec3::new(1., 1., 0.),
            Dimensionality::ThreeD => DVec3::new(1., 1., 1.),
        };
        for pos in self.coordinates.iter_mut() {
            *pos *= dimension_fac;
        }
    }

    fn init_volumes(&mut self) -> &[f64] {
        self.volumes.get_or_insert_with(|| {
            assert_eq!(
                self.coordinates.len(),
                self.num_part,
                "Invalid length of coordinates!"
            );
            VoronoiIntegrator::build(
                &self.coordinates,
                None,
                DVec3::ZERO,
                self.box_size,
                self.dimensionality.into(),
                self.periodic,
            )
            .compute_cell_integrals::<VolumeIntegral>()
            .iter()
            .map(|int| int.volume)
            .collect::<Vec<_>>()
        })
    }

    fn get_volumes(&self) -> &[f64] {
        self.volumes.as_ref().unwrap()
    }

    pub fn set_coordinates(mut self, coordinates: Vec<DVec3>) -> Self {
        self.coordinates = coordinates;
        assert_eq!(
            self.coordinates.len(),
            self.num_part,
            "Invalid length of coordinates!"
        );
        self
    }

    pub fn set_masses(mut self, masses: Vec<f64>) -> Self {
        self.masses = masses;
        assert_eq!(
            self.masses.len(),
            self.num_part,
            "Invalid length of masses!"
        );
        self
    }

    pub fn set_velocities(mut self, velocities: Vec<DVec3>) -> Self {
        self.velocities = velocities;
        assert_eq!(
            self.velocities.len(),
            self.num_part,
            "Invalid length of velocities!"
        );
        self
    }

    pub fn set_internal_energies(mut self, internal_energies: Vec<f64>) -> Self {
        self.internal_energies = internal_energies;
        assert_eq!(
            self.internal_energies.len(),
            self.num_part,
            "Invalid length of internal energies!"
        );
        self
    }

    pub fn set_densities(mut self, densities: &[f64]) -> Self {
        // Set masses
        assert_eq!(
            densities.len(),
            self.num_part,
            "Invalid length of densities!"
        );
        self.init_volumes();
        self.masses = densities
            .iter()
            .zip(self.get_volumes().iter())
            .map(|(rho, vol)| rho * vol)
            .collect();

        self
    }

    pub fn set_pressures(mut self, pressures: &[f64], eos: &GasLaw) -> Self {
        assert_eq!(
            self.masses.len(),
            self.num_part,
            "Invalid length of masses!"
        );
        assert_eq!(
            pressures.len(),
            self.num_part,
            "Invalid length of pressures!"
        );

        self.init_volumes();
        let inv_densities = self
            .masses
            .iter()
            .zip(self.get_volumes().iter())
            .map(|(m, vol)| vol / m);
        self.internal_energies = pressures
            .iter()
            .zip(inv_densities)
            .map(|(&pres, rho_inv)| eos.gas_internal_energy_from_pressure(pres, rho_inv))
            .collect();

        self
    }

    pub fn set_momenta(mut self, momenta: &[DVec3]) -> Self {
        assert_eq!(
            self.masses.len(),
            self.num_part,
            "Invalid length of masses!"
        );
        assert_eq!(momenta.len(), self.num_part, "Invalid length of momenta!");

        self.velocities = self
            .masses
            .iter()
            .zip(momenta.iter())
            .map(|(m, p)| *p / *m)
            .collect();
        self
    }

    pub fn set_smoothing_lengths(mut self, smoothing_lengths: Vec<f64>) -> Self {
        self.smoothing_lengths = Some(smoothing_lengths);
        self
    }

    pub fn set_ids(mut self, ids: Vec<u64>) -> Self {
        self.ids = Some(ids);
        self
    }

    pub fn save<P: AsRef<Path>>(&mut self, save_name: P) -> Result<(), hdf5::Error> {
        let file = hdf5::File::create(save_name)?;

        // Do we need to compute smoothing lengths?
        if self.smoothing_lengths.is_none() {
            self.init_volumes();
        }
        let smoothing_lengths: &Vec<f64> = self.smoothing_lengths.get_or_insert_with(|| {
            self.volumes
                .as_ref()
                .expect("Volumes have to be initialized at this point.")
                .iter()
                .map(|vol| {
                    let radius = match self.dimensionality {
                        Dimensionality::OneD => 0.5 * *vol,
                        Dimensionality::TwoD => (std::f64::consts::FRAC_1_PI * vol).sqrt(),
                        Dimensionality::ThreeD => {
                            (0.25 * 3. * std::f64::consts::FRAC_1_PI * vol).powf(1. / 3.)
                        }
                    };
                    1.5 * radius
                })
                .collect()
        });

        // Do we need to set ids?
        let ids: &Vec<u64> = self
            .ids
            .get_or_insert_with(|| (1..=self.num_part as u64).collect());

        // Write header
        let header = file.create_group("Header")?;
        create_attr!(header, self.box_size.to_array(), "BoxSize")?;
        create_attr!(header, [self.num_part, 0, 0, 0, 0], "NumPart_Total")?;
        create_attr!(
            header,
            [Into::<usize>::into(self.dimensionality)],
            "Dimension"
        )?;

        // Some unused values, necessary for swift compatibility
        create_attr!(header, [0], "Flag_Entropy_ICs")?;
        create_attr!(header, [0, 0, 0, 0, 0], "NumPart_Total_HighWord")?;

        // Write particle data
        let part_data = file.create_group("PartType0")?;
        create_dataset!(
            part_data,
            self.coordinates.iter().map(|x| x.to_array()).flatten(),
            "Coordinates"
        )?;
        part_data
            .new_dataset_builder()
            .with_data(&self.masses)
            .create("Masses")?;
        create_dataset!(
            part_data,
            self.velocities.iter().map(|v| v.to_array()).flatten(),
            "Velocities"
        )?;
        part_data
            .new_dataset_builder()
            .with_data(&self.internal_energies)
            .create("InternalEnergy")?;
        part_data
            .new_dataset_builder()
            .with_data(smoothing_lengths)
            .create("SmoothingLength")?;
        part_data
            .new_dataset_builder()
            .with_data(ids)
            .create("ParticleIDs")?;
        if let Some(volumes) = self.volumes.as_ref() {
            part_data
                .new_dataset_builder()
                .with_data(volumes)
                .create("Volumes")?;
        }

        Ok(())
    }

    pub fn box_size(&self) -> DVec3 {
        self.box_size
    }

    pub(crate) fn into_parts(mut self) -> Vec<Particle> {
        self.postprocess_positions();
        self.coordinates
            .iter()
            .zip(self.masses.iter())
            .zip(self.velocities.iter())
            .zip(self.internal_energies.iter())
            .map(|(((&coordinate, &mass), &velocity), &internal_energy)| {
                Particle::from_ic(coordinate, mass, velocity, internal_energy)
            })
            .collect()
    }

    pub fn dimensionality(&self) -> Dimensionality {
        self.dimensionality
    }
}
