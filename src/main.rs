extern crate yaml_rust;
#[macro_use]
extern crate derive_more;

use std::fs;

use clap::Parser;
use cli::Cli;
use engine::Engine;
use space::Space;
use yaml_rust::YamlLoader;

use crate::{equation_of_state::EquationOfState, initial_conditions::InitialConditions};

mod cli;
mod engine;
mod equation_of_state;
mod errors;
mod flux;
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

    // Setup simulation
    let eos = EquationOfState::new(&config["equation_of_state"])?;
    let mut engine = Engine::init(
        &config["engine"],
        &config["time_integration"],
        &config["snapshots"],
        &eos,
    )?;
    let ic = InitialConditions::new(&config["initial_conditions"], &eos)?;
    let mut space = Space::from_ic(ic, &config["space"], eos)?;

    // run
    engine.run(&mut space)?;

    println!("Done!");
    Ok(())
}
