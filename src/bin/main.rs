use clap::Parser;
use glam::DVec3;
use mvmm_hydro::{
    finite_volume_solver::{EulerEqnsFvs, FiniteVolumeSolver}, gas_law::{EquationOfState, GasLaw}, gravity::{ExternalPotentialGravity, GravitySolver, KeplerianPotential, Potential, SelfGravity}, hydrodynamics::OptimalOrderRunner, riemann_solver::{
        AIRiemannSolver, ExactRiemannSolver, HLLCRiemannSolver, PVRiemannSolver,
        TRRiemannSolver, TSRiemannSolver,
    }, Engine, InitialConditions, ParticleMotion, Space
};
use std::path;

use std::{error::Error, fmt::Display, fs, path::PathBuf};

use mvmm_hydro::{Boundary, Dimensionality};
use yaml_rust::{Yaml, YamlLoader};

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

fn sod_shock(coordinate: DVec3) -> (f64, DVec3, f64) {
    let density = if coordinate.x < 0.5 { 1. } else { 0.125 };
    let velocity = DVec3::ZERO;
    let pressure = if coordinate.x < 0.5 { 1. } else { 0.1 };
    (density, velocity, pressure)
}

fn noh(coordinate: DVec3) -> (f64, DVec3, f64) {
    let density = 1.;
    let velocity = if coordinate.x < 0.5 {
        DVec3::X
    } else {
        -DVec3::X
    };
    let pressure = 1.0e-6;
    (density, velocity, pressure)
}

fn toro(coordinate: DVec3) -> (f64, DVec3, f64) {
    let density = 1.;
    let velocity = if coordinate.x < 0.5 { 2. } else { -2. };
    let pressure = 0.4;
    (density, velocity * DVec3::X, pressure)
}

fn vacuum_expansion(coordinate: DVec3) -> (f64, DVec3, f64) {
    let density = if coordinate.x < 0.5 { 1. } else { 0. };
    let velocity = DVec3::ZERO;
    let pressure = if coordinate.x < 0.5 { 1. } else { 0. };
    (density, velocity, pressure)
}

// /// Generates a spherically symmetric 1/r density profile
// fn _evrard(_num_part: usize, _box_size: DVec3, _eos: &GasLaw) -> Vec<Particle> {
//     let mut ic = Vec::<Particle>::with_capacity(num_part);
//     let num_part_inv = 1. / (num_part as f64);

//     let u0 = 0.05;
//     let mut m_tot = 0.;
//     for idx in 0..num_part {
//         let x = (idx as f64 + 0.5) * num_part_inv;
//         let density = 1. / x;
//         let r_in = idx as f64 * num_part_inv;
//         let r_out = (idx as f64 + 1.) * num_part_inv;
//         m_tot += density * 4. * std::f64::consts::FRAC_PI_3 * (r_out.powi(3) - r_in.powi(3));
//         let velocity = 0.;
//         let pressure = eos.gas_pressure_from_internal_energy(u0, density);
//         ic.push(conv1d!(
//             x * box_size.x,
//             density,
//             velocity,
//             pressure,
//             box_size,
//             num_part_inv,
//             eos
//         ));
//     }

//     let correction = 1. / m_tot;
//     for part in ic.iter_mut() {
//         part.conserved = State::<Conserved>::new(
//             correction * part.conserved.mass(),
//             part.conserved.momentum(),
//             part.conserved.energy(),
//         );
//     }

//     ic
// }

fn constant(_coordinate: DVec3) -> (f64, DVec3, f64) {
    let density = 1.;
    let velocity = DVec3::ZERO;
    let pressure = 1.;
    (density, velocity, pressure)
}

fn square(coordinate: DVec3) -> (f64, DVec3, f64) {
    let density_low = 1.;
    let density_high = 4.;
    let density = if coordinate.x < 0.25 || coordinate.x > 0.75 {
        density_low
    } else {
        density_high
    };
    let velocity = DVec3::X;
    let pressure = 1.;
    (density, velocity, pressure)
}

pub fn ics_from_preset(
    name: String,
    num_part: usize,
    perturbations: Option<f64>,
    dimensionality: Dimensionality,
    box_size: DVec3,
    periodic: bool,
    eos: &GasLaw,
) -> Result<InitialConditions, ConfigError> {
    let ics = match name.as_str() {
        "sodshock" => InitialConditions::from_fn(
            box_size,
            num_part,
            dimensionality,
            periodic,
            eos,
            perturbations,
            sod_shock,
        ),
        "noh" => InitialConditions::from_fn(
            box_size,
            num_part,
            dimensionality,
            periodic,
            eos,
            perturbations,
            noh,
        ),
        "toro" => InitialConditions::from_fn(
            box_size,
            num_part,
            dimensionality,
            periodic,
            eos,
            perturbations,
            toro,
        ),
        "vacuum-expansion" => InitialConditions::from_fn(
            box_size,
            num_part,
            dimensionality,
            periodic,
            eos,
            perturbations,
            vacuum_expansion,
        ),
        "evrard" => todo!(),
        "constant" => InitialConditions::from_fn(
            box_size,
            num_part,
            dimensionality,
            periodic,
            eos,
            perturbations,
            constant,
        ),
        "square-advection" => InitialConditions::from_fn(
            box_size,
            num_part,
            dimensionality,
            periodic,
            eos,
            perturbations,
            square,
        ),
        _ => return Err(ConfigError::UnknownICs(name)),
    };

    Ok(ics)
}

#[derive(Debug)]
pub enum ConfigError {
    MissingParameter(String),
    UnknownRunner(String),
    UnknownICs(String),
    UnknownGravity(String),
    UnknownParticleMotion(String),
    UnknownBoundaryConditions(String),
    UnknownEOS(String),
    UnknownRiemannSolver(String),
    IllegalDVec3(String),
    InvalidArrayFormat(Yaml),
    InvalidArrayLength(usize, usize),
}

impl Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::MissingParameter(name) => {
                write!(f, "Missing required parameter in configuration: {}", name)
            }
            ConfigError::UnknownRunner(name) => {
                write!(f, "Unknown type of runner configured: {}", name)
            }
            ConfigError::UnknownGravity(name) => {
                write!(f, "Unknown type of gravity configured: {name}")
            }
            ConfigError::UnknownICs(name) => {
                write!(f, "Unknown type of initial conditions configured: {}", name)
            }
            ConfigError::UnknownParticleMotion(name) => {
                write!(f, "Unknown type of particle motion configured: {}", name)
            }
            ConfigError::UnknownBoundaryConditions(name) => {
                write!(f, "Unknown type of boundary condition configured: {}", name)
            }
            ConfigError::UnknownEOS(name) => {
                write!(f, "Unknown type of equation of stated configured: {}", name)
            }
            ConfigError::UnknownRiemannSolver(name) => {
                write!(f, "Unknown type of Riemann solver configured: {}", name)
            }
            ConfigError::IllegalDVec3(name) => {
                write!(f, "Illegal DVec3 format: {}!", name)
            }
            ConfigError::InvalidArrayFormat(value) => {
                write!(f, "Expected array but found: {:?}", value)
            }
            ConfigError::InvalidArrayLength(a, b) => {
                write!(f, "Expected array of lenght {}, but found {}", a, b)
            }
        }
    }
}

impl Error for ConfigError {}

fn parse_dvec3(yaml: &Yaml) -> Result<DVec3, ()> {
    let yaml_vec = yaml.as_vec().ok_or(())?;
    match &yaml_vec[..] {
        [Yaml::Real(a), Yaml::Real(b), Yaml::Real(c)] => Ok(DVec3 {
            x: a.parse().unwrap(),
            y: b.parse().unwrap(),
            z: c.parse().unwrap(),
        }),
        _ => Err(()),
    }
}

struct SpaceCfg {
    boundary: Boundary,
    max_top_level_cells: usize,
}

impl SpaceCfg {
    fn parse(yaml: &Yaml) -> Result<Self, ConfigError> {
        let max_top_level_cells = yaml["max_top_level_cells"].as_i64().unwrap_or(12) as usize;
        let boundary = yaml["boundary"]
            .as_str()
            .unwrap_or("reflective")
            .to_string();
        let boundary = match boundary.as_str() {
            "periodic" => Boundary::Periodic,
            "reflective" => Boundary::Reflective,
            "open" => Boundary::Open,
            "vacuum" => Boundary::Vacuum,
            _ => return Err(ConfigError::UnknownBoundaryConditions(boundary)),
        };
        Ok(Self {
            boundary,
            max_top_level_cells,
        })
    }
}

enum InitialConditionsCfg {
    File {
        filename: String,
    },
    Config {
        num_part: usize,
        dimensionality: Dimensionality,
        box_size: DVec3,
        periodic: bool,
        x: Vec<f64>,
        mass: Vec<f64>,
        velocity: Vec<f64>,
        internal_energy: Vec<f64>,
    },
    Preset {
        name: String,
        num_part: usize,
        dimensionality: Dimensionality,
        box_size: DVec3,
        periodic: bool,
        perturbations: Option<f64>,
    },
}

impl InitialConditionsCfg {
    fn parse(yaml: &Yaml) -> Result<Self, ConfigError> {
        let kind = yaml["kind"].as_str().ok_or(ConfigError::MissingParameter(
            "initial_conditions:kind".to_string(),
        ))?;

        Ok(match kind {
            "file" => {
                let filename = yaml["filename"]
                    .as_str()
                    .ok_or(ConfigError::MissingParameter(
                        "initial_conditions:filename".to_string(),
                    ))?
                    .to_string();
                Self::File { filename }
            }
            _ => {
                // Not reading from file, continue parsing the configuration
                let num_part = yaml["num_part"].as_i64().unwrap_or(100) as usize;
                let box_size = parse_dvec3(&yaml["box_size"])
                    .map_err(|_| ConfigError::IllegalDVec3(format!("{:?}", yaml["box_size"])))?;
                let periodic = yaml["periodic"].as_bool().unwrap_or(false);
                let dimensionality = (yaml["dimensionality"].as_i64().unwrap_or(3) as usize)
                    .try_into()
                    .expect("Hydro dimensionality must be <= 3!");
                match kind {
                    "config" => {
                        let x = cfg_ics2vec!(yaml, "x", num_part)?;
                        let mass = cfg_ics2vec!(yaml, "mass", num_part)?;
                        let velocity = cfg_ics2vec!(yaml, "velocity", num_part)?;
                        let internal_energy = cfg_ics2vec!(yaml, "internal_energy", num_part)?;
                        Self::Config {
                            num_part,
                            dimensionality,
                            box_size,
                            periodic,
                            x,
                            mass,
                            velocity,
                            internal_energy,
                        }
                    }
                    _ => Self::Preset {
                        name: kind.to_string(),
                        num_part,
                        dimensionality,
                        box_size,
                        periodic,
                        perturbations: yaml["perturbations"].as_f64(),
                    },
                }
            }
        })
    }
}

struct HydroCfg {
    gas_law: GasLaw,
    cfl: f64,
    riemann: RiemannCfg,
}

impl HydroCfg {
    fn parse(yaml: &Yaml) -> Result<Self, ConfigError> {
        let gamma = yaml["gamma"].as_f64().ok_or(ConfigError::MissingParameter(
            "hydrodynamics: gamma".to_string(),
        ))?;
        let equation_of_state = yaml["equation_of_state"]
            .as_str()
            .ok_or(ConfigError::MissingParameter(
                "hydrodynamics: equation_of_state".to_string(),
            ))?
            .to_string();
        let equation_of_state = match equation_of_state.as_str() {
            "Ideal" => EquationOfState::Ideal,
            "Isothermal" => {
                let isothermal_internal_energy = yaml["isothermal_internal_energy"]
                    .as_f64()
                    .ok_or(ConfigError::MissingParameter(
                        "hydrodynamics: isothermal_internal_energy".to_string(),
                    ))?;
                EquationOfState::Isothermal {
                    isothermal_internal_energy,
                }
            }
            _ => return Err(ConfigError::UnknownEOS(equation_of_state)),
        };
        let cfl = yaml["cfl_criterion"].as_f64().ok_or(ConfigError::MissingParameter(
            "hydrodynamics: cfl_criterion".to_string(),
        ))?;
        let riemann = RiemannCfg::parse(&yaml["riemann_solver"])?;
        Ok(Self {
            gas_law: GasLaw::new(gamma, equation_of_state),
            cfl,
            riemann,
        })
    }
}

struct RiemannCfg {
    kind: String,
    threshold: Option<f64>,
}

impl RiemannCfg {
    fn parse(yaml: &Yaml) -> Result<Self, ConfigError> {
        Ok(Self {
            kind: yaml["kind"]
                .as_str()
                .ok_or(ConfigError::MissingParameter(
                    "riemann_solver: kind".to_string(),
                ))?
                .to_string(),
            threshold: yaml["threshold"].as_f64(),
        })
    }
}

struct PotentialCfg {
    kind: String,
    acceleration: Option<DVec3>,
    position: Option<DVec3>,
}

impl PotentialCfg {
    fn parse(yaml: &Yaml) -> Result<Self, ConfigError> {
        let kind = yaml["kind"].as_str().ok_or(ConfigError::MissingParameter("gravity: potential: kind".to_string()))?;
        let acceleration = &yaml["acceleration"];
        let acceleration = if acceleration.is_badvalue() {
            None
        } else {
            Some(parse_dvec3(acceleration).map_err(
                |_| ConfigError::IllegalDVec3("gravity:potential:acceleration".to_string())
            )?)
        };
        let position = &yaml["position"];
        let position = if position.is_badvalue() {
            None
        } else {
            Some(parse_dvec3(position).map_err(
                |_| ConfigError::IllegalDVec3("gravity:potential:position".to_string())
            )?)
        };
        Ok(Self { kind: kind.to_string(), acceleration, position })
    }
}

struct GravityCfg {
    kind: String,
    softening_length: Option<f64>,
    potential: Option<PotentialCfg>,
}

impl GravityCfg {
    fn parse(yaml: &Yaml) -> Result<Self, ConfigError> {
        let kind = yaml["kind"].as_str().unwrap_or("none").to_string();
        let softening_length = yaml["softening_length"].as_f64();
        let potential = if yaml["potential"].is_badvalue() {
            None
        } else {
            Some(PotentialCfg::parse(&yaml["potential"])?)
        };
        
        Ok(Self { kind, softening_length, potential })
    }
}

struct EngingeCfg {
    runner: String,
    dt_min: f64,
    dt_max: f64,
    t_end: f64,
    sync_timesteps: bool,
    dt_snap: f64,
    prefix: String,
    save_faces: bool,
    dt_status: f64,
    particle_motion: ParticleMotion,
}

impl EngingeCfg {
    fn parse(
        yaml_engine: &Yaml,
        yaml_time_integration: &Yaml,
        yaml_snapshots: &Yaml,
    ) -> Result<Self, ConfigError> {
        let dt_status = yaml_engine["dt_status"]
            .as_f64()
            .ok_or(ConfigError::MissingParameter(
                "engine:dt_status".to_string(),
            ))?;
        let runner = yaml_engine["runner"]
            .as_str()
            .ok_or(ConfigError::MissingParameter("engine: runner".to_string()))?;
        let particle_motion = yaml_engine["particle_motion"].as_str().unwrap_or("fluid");
        let particle_motion = match particle_motion {
            "fixed" => ParticleMotion::Fixed,
            "steer" => ParticleMotion::Steer,
            "steer_pakmor" => ParticleMotion::SteerPakmor,
            "fluid" => ParticleMotion::Fluid,
            _ => {
                return Err(ConfigError::UnknownParticleMotion(
                    particle_motion.to_string(),
                ))
            }
        };

        let dt_min =
            yaml_time_integration["dt_min"]
                .as_f64()
                .ok_or(ConfigError::MissingParameter(
                    "time_integration:t_end".to_string(),
                ))?;
        let dt_max =
            yaml_time_integration["dt_max"]
                .as_f64()
                .ok_or(ConfigError::MissingParameter(
                    "time_integration:dt_max".to_string(),
                ))?;
        let t_end =
            yaml_time_integration["t_end"]
                .as_f64()
                .ok_or(ConfigError::MissingParameter(
                    "time_integration:t_end".to_string(),
                ))?;
        let sync_timesteps = yaml_time_integration["sync_timesteps"]
            .as_bool()
            .unwrap_or(false);

        let dt_snap = yaml_snapshots["dt_snap"]
            .as_f64()
            .ok_or(ConfigError::MissingParameter(
                "snapshots:dt_snap".to_string(),
            ))?;
        let prefix = yaml_snapshots["prefix"]
            .as_str()
            .ok_or(ConfigError::MissingParameter(
                "snapshots:prefix".to_string(),
            ))?;
        let save_faces = yaml_snapshots["save_faces"].as_bool().unwrap_or(false);

        Ok(Self {
            runner: runner.to_string(),
            dt_min,
            dt_max,
            t_end,
            sync_timesteps,
            dt_snap,
            prefix: prefix.to_string(),
            save_faces,
            dt_status,
            particle_motion,
        })
    }
}

struct Config {
    hydro: HydroCfg,
    gravity: GravityCfg,
    engine: EngingeCfg,
    initial_conditions: InitialConditionsCfg,
    space: SpaceCfg,
}

impl Config {
    fn parse(file: PathBuf) -> Result<Self, Box<dyn Error>> {
        let docs = YamlLoader::load_from_str(&fs::read_to_string(file)?)?;
        let config_yml = &docs[0];

        Ok(Self {
            hydro: HydroCfg::parse(&config_yml["hydrodynamics"])?,
            gravity: GravityCfg::parse(&config_yml["gravity"])?,
            engine: EngingeCfg::parse(
                &config_yml["engine"],
                &config_yml["time_integration"],
                &config_yml["snapshots"],
            )?,
            initial_conditions: InitialConditionsCfg::parse(&config_yml["initial_conditions"])?,
            space: SpaceCfg::parse(&config_yml["space"])?,
        })
    }
}

#[derive(Parser)]
pub struct Cli {
    /// The path to the config file to read
    #[clap(parse(from_os_str))]
    pub config: path::PathBuf,
}

macro_rules! get_hydro_solver {
    (kind:expr, gas_law:expr, cfl:expr) => {
        match kind.as_str() {
            "HLLC" 
        }
    };
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // parse command line parameters
    let args = Cli::parse();

    // read configuration
    let config = Config::parse(args.config)?;

    // Setup simulation
    let gas_law = config.hydro.gas_law;
    let cfl = config.hydro.cfl;
    let finite_volume_solver: Box<dyn FiniteVolumeSolver> = match config.hydro.riemann.kind.as_str() {
        "HLLC" => Box::new(EulerEqnsFvs::new(HLLCRiemannSolver, cfl, gas_law)),
        "PVRS" => Box::new(EulerEqnsFvs::new(PVRiemannSolver, cfl, gas_law)),
        "Exact" => Box::new(EulerEqnsFvs::new(ExactRiemannSolver, cfl, gas_law)),
        "TSRS" => Box::new(EulerEqnsFvs::new(TSRiemannSolver, cfl, gas_law)),
        "TRRS" => Box::new(EulerEqnsFvs::new(TRRiemannSolver, cfl, gas_law)),
        "AIRS" => {
            let threshold = config.hydro.riemann.threshold.ok_or(ConfigError::MissingParameter("hydrodynamics: riemann_solver: threshold".to_string()))?;
            Box::new(EulerEqnsFvs::new(AIRiemannSolver::new(threshold), cfl, gas_law))
        }
        _ => Err(ConfigError::UnknownRiemannSolver(
            config.hydro.riemann.kind,
        ))?,
    };
    let gravity_solver: Option<Box<dyn GravitySolver>> = match config.gravity.kind.as_str() {
        "none" => None,
        "self_gravity" => {
            let softening_length = config.gravity.softening_length.ok_or(ConfigError::MissingParameter("gravity: softening_length".to_string()))?;
            Some(Box::new(SelfGravity::new(softening_length)))
        }
        "external_potential" => {
            let potential_cfg = config.gravity.potential.ok_or(ConfigError::MissingParameter("gravity: potential".to_string()))?;
            let potential = match potential_cfg.kind.as_str() {
                "constant" => {
                    let acceleration = potential_cfg.acceleration.ok_or(ConfigError::MissingParameter("gravity: potential: acceleration".to_string()))?;
                    Potential::Constant { acceleration }
                }
                "keplerian_disc" => {
                    let position = potential_cfg.position.ok_or(ConfigError::MissingParameter("gravity: potential: position".to_string()))?;
                    let softening_length = config.gravity.softening_length.ok_or(ConfigError::MissingParameter("gravity: softening_length".to_string()))?;
                    Potential::Keplerian(KeplerianPotential::new(position, softening_length))
                }
                _ => return Err(Box::new(ConfigError::UnknownGravity(format!("gravity: potential: kind: {:}", potential_cfg.kind).to_string()))),
            };
            Some(Box::new(ExternalPotentialGravity::new(potential)))
        }
        _ => return Err(Box::new(ConfigError::UnknownGravity(config.gravity.kind))),
    };
    let runner = match config.engine.runner.as_str() {
        "OptimalOrder" => Box::new(OptimalOrderRunner),
        _ => Err(ConfigError::UnknownRunner(
            config.engine.runner,
        ))?,
    };
    let mut engine = Engine::new(
        runner,
        finite_volume_solver,
        gravity_solver, 
        config.engine.t_end,
        config.engine.dt_min,
        config.engine.dt_max,
        config.engine.sync_timesteps,
        config.engine.dt_snap,
        &config.engine.prefix,
        config.engine.dt_status,
        config.engine.save_faces,
        config.engine.particle_motion,
    );
    

    // Setup ICs and construct space
    let ic = match config.initial_conditions {
        InitialConditionsCfg::File { filename } => InitialConditions::from_hdf5(filename)?,
        InitialConditionsCfg::Config {
            num_part,
            dimensionality,
            box_size,
            periodic,
            x,
            mass,
            velocity,
            internal_energy,
        } => InitialConditions::empty(num_part, dimensionality, box_size, periodic)
            .set_coordinates(x.into_iter().map(|x| x * DVec3::X).collect())
            .set_masses(mass)
            .set_velocities(velocity.into_iter().map(|v| v * DVec3::X).collect())
            .set_internal_energies(internal_energy),
        InitialConditionsCfg::Preset {
            name,
            num_part,
            dimensionality,
            box_size,
            periodic,
            perturbations,
        } => ics_from_preset(
            name,
            num_part,
            perturbations,
            dimensionality,
            box_size,
            periodic,
            &gas_law,
        )?,
    };
    let mut space = Space::initialize(
        ic,
        config.space.max_top_level_cells,
        config.space.boundary,
        gas_law,
    );

    // run
    engine.run(&mut space)?;

    println!("Done!");
    Ok(())
}
