use yaml_rust::Yaml;

use crate::errors::ConfigError;

fn sod_shock(num_part: usize) -> Vec<(f64, f64, f64, f64)> {
    let mut ic = Vec::<(f64, f64, f64, f64)>::new();
    let num_part_inv = 1. / (num_part as f64);

    for idx in 0..num_part {
        let x = (idx as f64 + 0.5) * num_part_inv;
        let density = if idx < num_part / 2 { 1. } else { 0.125 };
        let velocity = 0.;
        let pressure = if idx < num_part / 2 { 1. } else { 0.1 };
        ic.push((x, density, velocity, pressure));
    }

    ic
}

fn noh(num_part: usize) -> Vec<(f64, f64, f64, f64)> {
    let mut ic = Vec::<(f64, f64, f64, f64)>::new();
    let num_part_inv = 1. / (num_part as f64);

    for idx in 0..num_part {
        let x = (idx as f64 + 0.5) * num_part_inv;
        let density = 1.;
        let velocity = if x < 0.5 { 1. } else { -1. };
        let pressure = 1.0e-6;
        ic.push((x, density, velocity, pressure));
    }

    ic
}

fn toro(num_part: usize) -> Vec<(f64, f64, f64, f64)> {
    let mut ic = Vec::<(f64, f64, f64, f64)>::new();
    let num_part_inv = 1. / (num_part as f64);

    for idx in 0..num_part {
        let x = (idx as f64 + 0.5) * num_part_inv;
        let density = 1.;
        let velocity = if x < 0.5 { 2. } else { -2. };
        let pressure = 0.4;
        ic.push((x, density, velocity, pressure));
    }

    ic
}

fn vacuum(num_part: usize) -> Vec<(f64, f64, f64, f64)> {
    let mut ic = Vec::<(f64, f64, f64, f64)>::new();
    let num_part_inv = 1. / (num_part as f64);

    for idx in 0..num_part {
        let x = (idx as f64 + 0.5) * num_part_inv;
        let density = if x < 0.5 { 1. } else { 0. };
        let velocity = 0.;
        let pressure = if x < 0.5 { 1. } else { 0. };
        ic.push((x, density, velocity, pressure));
    }

    ic
}

pub fn create_ics(ic_cfg: &Yaml) -> Result<Vec<(f64, f64, f64, f64)>, ConfigError> {
    let kind = ic_cfg["type"]
        .as_str()
        .ok_or(ConfigError::MissingParameter("initial_conditions:type".to_string()))?
        .to_string();
    let num_part = ic_cfg["num_part"]
        .as_i64()
        .unwrap_or(100) as usize;

    match kind.as_str() {
        "sodshock" => Ok(sod_shock(num_part)),
        "noh" => Ok(noh(num_part)),
        "toro" => Ok(toro(num_part)),
        "vacuum" => Ok(vacuum(num_part)),
        _ => Err(ConfigError::UnknownICs(kind)),
    }
}
