use std::{
    fs::File,
    io::{BufWriter, Error as IoError},
};

use crate::{
    riemann_solver::RiemannSolver, space::Space, time_integration::Runner,
    timeline::*,
};

pub struct Engine<'a> {
    runner: &'a dyn Runner,
    pub solver: RiemannSolver,
    t_end: f64,
    t_current: f64,
    ti_old: IntegerTime,
    ti_current: IntegerTime,
    time_base: f64,
    time_base_inv: f64,
    cfl_criterion: f64,
    t_dump: f64,
    t_between_snaps: f64,
    snap: u16,
    snapshot_prefix: String,
}

impl<'a> Engine<'a> {
    /// Setup a simulation by initializing a new engine struct for initial conditions
    pub fn init(
        runner: &'a dyn Runner,
        gamma: f64,
        cfl_criterion: f64,
        t_end: f64,
        t_between_snaps: f64,
        snapshot_prefix: &str,
    ) -> Self {
        let solver = RiemannSolver::new(gamma);
        let time_base = t_end / MAX_NR_TIMESTEPS as f64;
        Self {
            runner,
            solver,
            t_end,
            t_current: 0.0,
            ti_old: 0,
            ti_current: 0,
            time_base,
            time_base_inv: 1. / time_base,
            cfl_criterion,
            t_between_snaps,
            t_dump: 0.0,
            snap: 0,
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
        let ti_next = self.runner.step(self, space);
        let dti = ti_next - self.ti_current;
        self.ti_old = self.ti_current;
        self.ti_current = ti_next;
        self.t_current += make_timestep(dti, self.time_base);
    }

    fn dump(&mut self, space: &mut Space) -> Result<(), IoError> {
        let f = File::create(&format!(
            "output/{}{:04}.txt",
            self.snapshot_prefix, self.snap
        ))?;
        let mut f = BufWriter::new(f);
        space.dump(&mut f)?;
        self.t_dump += self.t_between_snaps;
        self.snap += 1;

        Ok(())
    }

    pub fn ti_current(&self) -> IntegerTime {
        self.ti_current
    }

    pub fn time_base_inv(&self) -> f64 {
        self.time_base_inv
    }

    pub fn time_base(&self) -> f64 {
        self.time_base
    }

    pub fn dt(&self) -> f64 {
        make_timestep(self.ti_current - self.ti_old, self.time_base)
    }

    pub fn cfl_criterion(&self) -> f64 {
        self.cfl_criterion
    }
}
