use crate::{space::{Space, Boundary}, equation_of_state::EquationOfState, riemann_solver::{RiemannSolver, self}};

pub struct Engine {
    space: Space,
    eos: EquationOfState,
    solver: RiemannSolver,
    t_max: f64,
}

impl Engine {
    /// Setup a simulation by initializing a new engine struct for initial conditions
    pub fn new(ic: &[(f64, f64, f64, f64)], boundary: Boundary, box_size: f64, t_max: f64) -> Self {
        let gamma = 5. / 3.;
        let eos = EquationOfState::ideal(gamma);
        let solver = RiemannSolver::new(gamma);
        let space = Space::from_ic(ic, boundary, box_size, &eos);
        Self { space, eos, solver, t_max }
    }

    /// Run this simulation 
    pub fn run(&mut self) {
        todo!()
    }
}