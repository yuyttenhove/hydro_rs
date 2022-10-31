extern crate yaml_rust;
#[macro_use]
extern crate derive_more;

use std::fs;

use clap::Parser;
use cli::Cli;
use engine::Engine;
use errors::ConfigError;
use space::{Boundary, Space};
use yaml_rust::YamlLoader;

use crate::{
    initial_conditions::create_ics, riemann_solver::get_solver, time_integration::get_runner,
};

mod cli;
mod engine;
mod equation_of_state;
mod errors;
mod initial_conditions;
mod part;
mod physical_constants;
mod physical_quantities;
mod riemann_solver;
mod slope_limiters;
mod space;
mod time_integration;
mod timeline;
mod utils;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // parse command line parameters
    let args = Cli::parse();

    // read configuration
    let docs = YamlLoader::load_from_str(&fs::read_to_string(args.config)?)?;
    let config = &docs[0];

    let ic_kind = config["initial_conditions"]["type"]
        .as_str()
        .ok_or(ConfigError::MissingParameter("initial_conditions:type"))?
        .to_string();
    let num_part = config["initial_conditions"]["num_part"]
        .as_i64()
        .unwrap_or(100) as usize;
    let box_size = config["initial_conditions"]["box_size"]
        .as_f64()
        .unwrap_or(1.);
    let dt_min = config["time_integration"]["dt_min"]
        .as_f64()
        .ok_or(ConfigError::MissingParameter("time_integration:t_end"))?;
    let dt_max = config["time_integration"]["dt_max"]
        .as_f64()
        .ok_or(ConfigError::MissingParameter("time_integration:t_end"))?;
    let t_end = config["time_integration"]["t_end"]
        .as_f64()
        .ok_or(ConfigError::MissingParameter("time_integration:t_end"))?;
    let cfl_criterion = config["time_integration"]["cfl_criterion"].as_f64().ok_or(
        ConfigError::MissingParameter("time_integration:cfl_criterion"),
    )?;
    let runner_kind = config["time_integration"]["runner"]
        .as_str()
        .ok_or(ConfigError::MissingParameter("time_integration: runner"))?
        .to_string();
    let t_between_snaps = config["snapshots"]["t_between_snaps"]
        .as_f64()
        .ok_or(ConfigError::MissingParameter("snapshots:t_between_snaps"))?;
    let prefix = config["snapshots"]["prefix"]
        .as_str()
        .ok_or(ConfigError::MissingParameter("snapshots:t_between_snaps"))?;
    let t_status = config["engine"]["t_status"]
        .as_f64()
        .ok_or(ConfigError::MissingParameter("engine:t_status"))?;
    let solver_kind = config["engine"]["solver"]
        .as_str()
        .ok_or(ConfigError::MissingParameter("engine:solver"))?
        .to_string();
    let periodic = config["initial_conditions"]["periodic"]
        .as_bool()
        .unwrap_or(true);
    let gamma = config["equation_of_state"]["gamma"]
        .as_f64()
        .unwrap_or(5. / 3.);

    // Setup simulation
    let ic = create_ics(ic_kind, num_part, box_size)?;
    let boundary = if periodic {
        Boundary::Periodic
    } else {
        Boundary::Reflective
    };
    let mut space = Space::from_ic(&ic, boundary, box_size, gamma);

    let runner = get_runner(runner_kind)?;
    let solver = get_solver(solver_kind, gamma)?;
    let mut engine = Engine::init(
        runner,
        solver,
        cfl_criterion,
        dt_min,
        dt_max,
        t_end,
        t_between_snaps,
        t_status,
        prefix,
    );

    // run
    engine.run(&mut space)?;

    println!("Done!");
    Ok(())
}
