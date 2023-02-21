use std::fs;

use clap::Parser;
use cli::Cli;
use hydro_rs::{Engine, EquationOfState, InitialConditions, Space};
use yaml_rust::YamlLoader;

mod cli;

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
        &config["hydrodynamics"],
        &config["gravity"],
    )?;
    let ic = InitialConditions::new(&config["initial_conditions"], &eos)?;
    let mut space = Space::from_ic(ic, &config["space"], eos)?;

    // run
    engine.run(&mut space)?;

    println!("Done!");
    Ok(())
}
