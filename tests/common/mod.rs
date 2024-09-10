use std::f64;

use mvmm_hydro::{
    gas_law::GasLaw,
    riemann_solver::RiemannFluxSolver,
    Engine, InitialConditions, ParticleMotion, Runner, Space,
};

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

pub fn get_space(ic: InitialConditions, eos: GasLaw) -> Space {
    Space::from_ic(
        ic,
        1,
        mvmm_hydro::Boundary::Periodic,
        eos,
    )
}

pub fn get_testing_engine(
    runner: Runner,
    riemann_solver: Box<dyn RiemannFluxSolver>,
    t_end: f64,
    dt_min: f64,
    dt_max: f64,
    sync_all: bool,
    cfl_criterion: f64,
    particle_motion: ParticleMotion,
) -> Engine {
    Engine::new(
        runner,
        riemann_solver,
        None,
        t_end,
        dt_min,
        dt_max,
        sync_all,
        cfl_criterion,
        f64::INFINITY,
        "none",
        f64::INFINITY,
        false,
        particle_motion,
    )
}
