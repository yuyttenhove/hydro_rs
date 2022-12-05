use yaml_rust::Yaml;

use crate::{errors::ConfigError, equation_of_state::EquationOfState};

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

/// Generates a spherically symmetric 1/r density profile
fn evrard(num_part: usize) -> Vec<(f64, f64, f64, f64)> {
    assert!(cfg!(dimensionality="3D"));

    let mut ic = Vec::<(f64, f64, f64, f64)>::new();
    let num_part_inv = 1. / (num_part as f64);

    let mut m_tot = 0.;
    for idx in 0..num_part {
        let x = (idx as f64 + 0.5) * num_part_inv;
        let density = 1. / x;
        let r_in = idx as f64 * num_part_inv;
        let r_out = (idx as f64 + 1.) * num_part_inv;
        m_tot += density * 4. * std::f64::consts::FRAC_PI_3 * (r_out.powi(3) - r_in.powi(3));
        let velocity = 0.;
        let pressure = 1.;
        ic.push((x, density, velocity, pressure));
    }

    let u0 = 0.05;
    let correction = 1. / m_tot;
    let eos = EquationOfState::Ideal { gamma: 5. / 3. };
    for part in ic.iter_mut() {
        part.1 *= correction;
        part.3 = eos.gas_pressure_from_internal_energy(u0, part.1);
    }

    ic
}

fn constant(num_part: usize) -> Vec<(f64, f64, f64, f64)> {
    let mut ic = Vec::<(f64, f64, f64, f64)>::new();
    let num_part_inv = 1. / (num_part as f64);

    for idx in 0..num_part {
        let x = (idx as f64 + 0.5) * num_part_inv;
        let density = 1.;
        let velocity = 0.;
        let pressure = 1.;
        ic.push((x, density, velocity, pressure));
    }

    ic
}

fn square(num_part: usize) -> Vec<(f64, f64, f64, f64)> {
    let mut ic = Vec::<(f64, f64, f64, f64)>::new();
    let num_part_left = (0.8 * num_part as f64) as usize;
    let num_part_right = num_part - num_part_left;

    let density_low = 1.;
    let density_high = 4.;
    let square = |x: f64| {
        if x < 0.25 || x > 0.75 {
            density_low
        } else {
            density_high
        }
    };

    // left particles at four times the number density
    let num_part_left_inv = 0.5 / (num_part_left as f64);
    for idx in 0..num_part_left {
        let x = (idx as f64 + 0.5) * num_part_left_inv;
        let density = square(x);
        let velocity = 1.;
        let pressure = 1.;
        ic.push((x, density, velocity, pressure));
    }

    // right particles
    let num_part_right_inv = 0.5 / (num_part_right as f64);
    for idx in 0..num_part_right {
        let x = 0.5 + (idx as f64 + 0.5) * num_part_right_inv;
        let density = square(x);
        let velocity = 1.;
        let pressure = 1.;
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
        "evrard" => Ok(evrard(num_part)),
        "constant" => Ok(constant(num_part)),
        "square" => Ok(square(num_part)),
        _ => Err(ConfigError::UnknownICs(kind)),
    }
}
