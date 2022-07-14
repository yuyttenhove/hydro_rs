pub fn sod_shock(num_part: usize, box_size: f64) -> Vec<(f64, f64, f64, f64)> {
    
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