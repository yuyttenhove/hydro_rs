use crate::errors::ConfigError;

fn sod_shock(num_part: usize, box_size: f64) -> Vec<(f64, f64, f64, f64)> {

    let mut ic = Vec::<(f64, f64, f64, f64)>::new();
    let num_part_inv = 1. / (num_part as f64);
    
    for idx in 0..num_part {
        let x = (idx as f64 + 0.5) * box_size * num_part_inv;
        let density = if idx < num_part / 2 { 1. } else { 0.125 };
        let velocity = 0.;
        let pressure = if idx < num_part / 2 { 1. } else { 0.1 };
        ic.push((x, density, velocity, pressure));
    }

    ic
}

fn noh(num_part: usize, box_size: f64) -> Vec<(f64, f64, f64, f64)> {
    let mut ic = Vec::<(f64, f64, f64, f64)>::new();
    let num_part_inv = 1. / (num_part as f64);

    for idx in 0..num_part {
        let x = (idx as f64 + 0.5) * box_size * num_part_inv;
        let density = 1.;
        let velocity = if x < 0.5 * box_size { 1. } else { -1. };
        let pressure = 1.0e-6;
        ic.push((x, density, velocity, pressure));
    }

    ic
}

fn toro(num_part: usize, box_size: f64) -> Vec<(f64, f64, f64, f64)> {
    let mut ic = Vec::<(f64, f64, f64, f64)>::new();
    let num_part_inv = 1. / (num_part as f64);

    for idx in 0..num_part {
        let x = (idx as f64 + 0.5) * box_size * num_part_inv;
        let density = 1.;
        let velocity = if x < 0.5 * box_size { 2. } else { -2. };
        let pressure = 0.4;
        ic.push((x, density, velocity, pressure));
    }

    ic
}


pub fn create_ics<'a>(kind: String, num_part: usize, box_size: f64) -> Result<Vec<(f64, f64, f64, f64)>, ConfigError<'a>> {
    match kind.as_str() {
        "sodshock" => Ok(sod_shock(num_part, box_size)),
        "noh" => Ok(noh(num_part, box_size)),
        "toro" => Ok(toro(num_part, box_size)),
        _ => Err(ConfigError::UnknownICs(kind.to_string()))
    }
}