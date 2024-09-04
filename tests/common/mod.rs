use hydro_rs::{gas_law::GasLaw, Engine, InitialConditions, Space};
use yaml_rust::YamlLoader;

pub const _CONFIG: &'static str = r###"
time_integration:
  dt_min: 1e-11
  dt_max: 1e-2
  t_min: 1e-8
  t_end: 0.5
  cfl_criterion: 0.3

snapshots:
  t_between_snaps: 0.05
  prefix: "sedov_2D_"

engine:
  t_status: 0.005
  particle_motion: "fluid"
  runner: "OptimalOrder"

gravity:
  kind: "none"
  potential:
    kind: "keplerian"
    position: [0., 0., 0.]
    softening-length: 1e-3

hydrodynamics:
  solver: "Exact"
  threshold: 2.

space:
  boundary: "periodic"

initial_conditions:
  type: "file"
  box_size: [2., 1., 1.]
  num_part: 200
  filename: "ICs/sedov_2D.hdf5"

equation_of_state:
  gamma: 1.66666666667
"###;

pub const TIME_INTEGRATION_CONFIG: &'static str = r##"
dt_min: 1e-11
dt_max: 1e-2
t_min: 1e-8
t_end: 0.5
cfl_criterion: 0.3
"##;

pub const SNAPSHOTS_CONFIG: &'static str = r##"
t_between_snaps: 0.05
prefix: "sedov_2D_"
"##;

pub const _SPACE_CONFIG: &'static str = r##"
boundary: "periodic"
"##;

pub const ENGINE_CONFIG: &'static str = r##"
t_status: 0.005
particle_motion: "fluid"
runner: "OptimalOrder"
"##;

pub const HYDRO_CONFIG: &'static str = r##"
solver: "Exact"
"##;

pub const GRAVITY_CONFIG: &'static str = r##"
kind: "none"
"##;

pub const EOS_CONFIG: &'static str = r##"
gamma: 1.66666667
equation_of_state: 
  kind: "Ideal"
"##;

pub const _IC_CONFIG: &'static str = r##"
type: "file"
box_size: [2., 1., 1.]
num_part: 200
filename: "ICs/sedov_2D.hdf5"
"##;

pub fn get_eos(cfg: &str) -> GasLaw {
    GasLaw::init(&YamlLoader::load_from_str(cfg).expect("Error loading EOS cfg!")[0])
        .expect("Error creating GasLaw!")
}

pub fn _get_ic(cfg: &str, eos: &GasLaw) -> InitialConditions {
    InitialConditions::init(
        &YamlLoader::load_from_str(cfg).expect("Error loading ICs cfg!")[0],
        eos,
    )
    .expect("Error creating Initial conditions")
}

pub fn get_space(cfg: &str, ic: InitialConditions, eos: GasLaw) -> Space {
    Space::from_ic(
        ic,
        &YamlLoader::load_from_str(cfg).expect("Error loading space cfg!")[0],
        eos,
    )
    .expect("Error creating Space")
}

pub fn get_engine(
    engine_cfg: &str,
    time_integration_cfg: &str,
    snapshots_cfg: &str,
    hydro_solver_cfg: &str,
    gravity_solver_cfg: &str,
) -> Engine {
    Engine::init(
        &YamlLoader::load_from_str(engine_cfg).expect("Error loading engine cfg!")[0],
        &YamlLoader::load_from_str(time_integration_cfg).expect("Error loading engine cfg!")[0],
        &YamlLoader::load_from_str(snapshots_cfg).expect("Error loading engine cfg!")[0],
        &YamlLoader::load_from_str(hydro_solver_cfg).expect("Error loading engine cfg!")[0],
        &YamlLoader::load_from_str(gravity_solver_cfg).expect("Error loading engine cfg!")[0],
    )
    .expect("Error initializing engine!")
}
