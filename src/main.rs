use engine::Engine;
use initial_conditions::sod_shock;
use space::Boundary;

mod equation_of_state;
mod part;
mod riemann_solver;
mod physical_constants;
mod physical_quantities;
mod space;
mod utils;
mod engine;
mod initial_conditions;

fn main() {
    let num_part = 100;
    let box_size = 2.;
    let t_max = 1.;

    let ic = sod_shock(num_part, box_size);
    let boundary = Boundary::Periodic;
    let mut engine = Engine::new(&ic, boundary, box_size, t_max);

    engine.run();
}
