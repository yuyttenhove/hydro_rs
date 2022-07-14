extern crate yaml_rust;

use std::fs;

use engine::Engine;
use errors::ConfigError;
use initial_conditions::sod_shock;
use space::Boundary;
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // read configuration 
    let docs = YamlLoader::load_from_str(
        &fs::read_to_string("config.yml")?
    )?;
    let config = &docs[0];

    let num_part = config["num_part"].as_i64().unwrap_or(100) as usize;
    let box_size = config["box_size"].as_f64().unwrap_or(1.);
    let t_max = config["t_max"].as_f64().ok_or(ConfigError::MissingParameter("t_max"))?;
    let periodic = config["periodic"].as_bool().unwrap_or(true);

    // Setup simulation
    let ic = sod_shock(num_part, box_size);
    let boundary = if periodic { Boundary::Periodic } else { Boundary::Reflective };
    let mut engine = Engine::new(&ic, boundary, box_size, t_max);

    // run
    engine.run();

    Ok(())
}
