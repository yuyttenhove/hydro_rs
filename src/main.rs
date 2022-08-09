extern crate yaml_rust;
#[macro_use]
extern crate derive_more;

use std::fs;

use clap::Parser;
use cli::Cli;
use engine::Engine;
use errors::ConfigError;
use initial_conditions::sod_shock;
use time_integration::{DefaultRunner, OptimalRunner};
use space::{Boundary, Space};
use yaml_rust::YamlLoader;

mod equation_of_state;
mod part;
mod riemann_solver;
mod physical_constants;
mod physical_quantities;
mod space;
mod utils;
mod engine;
mod initial_conditions;
mod errors;
mod cli;
mod time_integration;
mod slope_limiters;

fn main() -> Result<(), Box<dyn std::error::Error>> {

    // parse command line parameters
    let args = Cli::parse();

    // read configuration 
    let docs = YamlLoader::load_from_str(
        &fs::read_to_string(args.config)?
    )?;
    let config = &docs[0];

    let num_part = config["initial_conditions"]["num_part"].as_i64().unwrap_or(100) as usize;
    let box_size = config["initial_conditions"]["box_size"].as_f64().unwrap_or(1.);
    let t_end = config["time_integration"]["t_end"].as_f64().ok_or(ConfigError::MissingParameter("time_integration:t_end"))?;
    let cfl_criterion = config["time_integration"]["cfl_criterion"].as_f64().ok_or(ConfigError::MissingParameter("time_integration:cfl_criterion"))?;
    let t_between_snaps = config["snapshots"]["t_between_snaps"].as_f64().ok_or(ConfigError::MissingParameter("snapshots:t_between_snaps"))?;
    let prefix = config["snapshots"]["prefix"].as_str().ok_or(ConfigError::MissingParameter("snapshots:t_between_snaps"))?;
    let periodic = config["initial_conditions"]["periodic"].as_bool().unwrap_or(true);
    let gamma = config["equation_of_state"]["gamma"].as_f64().unwrap_or(5. / 3.);

    // Setup simulation
    let ic = sod_shock(num_part, box_size);
    let boundary = if periodic { Boundary::Periodic } else { Boundary::Reflective };
    let mut space = Space::from_ic(&ic, boundary, box_size, gamma);
    let runner = DefaultRunner::new(cfl_criterion);
    let mut engine = Engine::init(&runner, gamma, t_end, t_between_snaps, prefix);
    
    // run
    engine.run(&mut space)?;

    Ok(())
}
