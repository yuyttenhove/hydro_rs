use std::{fs::File, io::{BufWriter, Error as IoError}};

use crate::{space::{Space}, riemann_solver::RiemannSolver, time_integration::Runner};

pub struct Engine<'a> {
    runner: &'a dyn Runner,
    pub solver: RiemannSolver,
    t_end: f64,
    t_current: f64,
    ti_current: u64,
    t_dump: f64,
    t_between_snaps: f64,
    snap: u16,
    snapshot_prefix: String,
}

impl<'a> Engine<'a> {
    /// Setup a simulation by initializing a new engine struct for initial conditions
    pub fn init(runner: &'a dyn Runner, gamma: f64, t_end: f64, t_between_snaps: f64, snapshot_prefix: &str) -> Self {
        let solver = RiemannSolver::new(gamma);
        Self { 
            runner, solver, t_end , t_current: 0.0, ti_current: 0, t_between_snaps, t_dump: 0.0, snap: 0, 
            snapshot_prefix: snapshot_prefix.to_string(),
        }
    }

    /// Run this simulation 
    pub fn run(&mut self, space: &mut Space) -> Result<(), IoError> {
        // Start by saving the initial state
        self.dump(space)?; 

        while self.t_current < self.t_end {
            self.step(space);
            // Do we need to save a snapshot?
            if self.t_current > self.t_dump {
                self.dump(space)?;
            }
        }

        // Save the final state of the simulation
        self.dump(space)?;

        Ok(())
    }

    fn step(&mut self, space: &mut Space) {
        let timestep = self.runner.step(self, space);
        self.t_current += timestep;
    }

    fn dump(&mut self, space: &mut Space) -> Result<(), IoError> {
        let f = File::create(&format!("output/{}{:04}.txt", self.snapshot_prefix, self.snap))?;
        let mut f = BufWriter::new(f);
        space.dump(&mut f)?;
        self.t_dump += self.t_between_snaps;
        self.snap += 1;

        Ok(())
    }
}