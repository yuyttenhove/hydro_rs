use std::{
    fs::File,
    io::{BufWriter, Error as IoError},
};

use yaml_rust::Yaml;

use crate::{
    equation_of_state::EquationOfState,
    errors::ConfigError,
    gravity::GravitySolver,
    riemann_solver::{get_solver, RiemannFluxSolver},
    space::Space,
    time_integration::Runner,
    timeline::*,
};

pub enum ParticleMotion {
    FIXED,
    STEER,
    FLUID,
}

impl ParticleMotion {
    fn new(kind: &str) -> Result<Self, ConfigError> {
        match kind {
            "fixed" => Ok(ParticleMotion::FIXED),
            "steer" => Ok(ParticleMotion::STEER),
            "fluid" => Ok(ParticleMotion::FLUID),
            _ => Err(ConfigError::UnknownParticleMotion(kind.to_string())),
        }
    }
}

pub struct Engine {
    runner: Runner,
    pub hydro_solver: Box<dyn RiemannFluxSolver>,
    pub gravity_solver: Option<GravitySolver>,
    t_end: f64,
    t_current: f64,
    ti_old: IntegerTime,
    ti_current: IntegerTime,
    ti_next: IntegerTime,
    time_base: f64,
    time_base_inv: f64,
    cfl_criterion: f64,
    dt_min: f64,
    dt_max: f64,
    ti_snap: IntegerTime,
    ti_between_snaps: IntegerTime,
    ti_status: IntegerTime,
    ti_between_status: IntegerTime,
    snap: u16,
    snapshot_prefix: String,
    pub particle_motion: ParticleMotion,
}

impl Engine {
    /// Setup a simulation by initializing a new engine struct for initial conditions
    pub fn init(
        engine_cfg: &Yaml,
        time_integration_cfg: &Yaml,
        snapshots_cfg: &Yaml,
        hydro_solver_cfg: &Yaml,
        gravity_solver_cfg: &Yaml,
        eos: &EquationOfState,
    ) -> Result<Self, ConfigError> {
        // Read config
        let t_status = engine_cfg["t_status"]
            .as_f64()
            .ok_or(ConfigError::MissingParameter("engine:t_status".to_string()))?;
        let runner_kind = engine_cfg["runner"]
            .as_str()
            .ok_or(ConfigError::MissingParameter(
                "time_integration: runner".to_string(),
            ))?;
        let particle_motion = engine_cfg["particle_motion"].as_str().unwrap_or("fluid");
        let dt_min =
            time_integration_cfg["dt_min"]
                .as_f64()
                .ok_or(ConfigError::MissingParameter(
                    "time_integration:t_end".to_string(),
                ))?;
        let dt_max =
            time_integration_cfg["dt_max"]
                .as_f64()
                .ok_or(ConfigError::MissingParameter(
                    "time_integration:dt_max".to_string(),
                ))?;
        let t_end = time_integration_cfg["t_end"]
            .as_f64()
            .ok_or(ConfigError::MissingParameter(
                "time_integration:t_end".to_string(),
            ))?;
        let cfl_criterion =
            time_integration_cfg["cfl_criterion"]
                .as_f64()
                .ok_or(ConfigError::MissingParameter(
                    "time_integration:cfl_criterion".to_string(),
                ))?;
        let t_between_snaps =
            snapshots_cfg["t_between_snaps"]
                .as_f64()
                .ok_or(ConfigError::MissingParameter(
                    "snapshots:t_between_snaps".to_string(),
                ))?;
        let prefix = snapshots_cfg["prefix"]
            .as_str()
            .ok_or(ConfigError::MissingParameter(
                "snapshots:t_between_snaps".to_string(),
            ))?;

        // Setup members
        let runner = Runner::new(runner_kind)?;
        let hydro_solver = get_solver(hydro_solver_cfg, eos)?;
        let gravity_solver = GravitySolver::init(gravity_solver_cfg)?;
        let particle_motion = ParticleMotion::new(particle_motion)?;
        let time_base = t_end / MAX_NR_TIMESTEPS as f64;
        let time_base_inv = 1. / time_base;
        Ok(Self {
            runner,
            hydro_solver,
            gravity_solver,
            t_end,
            t_current: 0.0,
            ti_old: 0,
            ti_current: 0,
            ti_next: 0,
            time_base,
            time_base_inv,
            cfl_criterion,
            dt_min,
            dt_max,
            ti_snap: 0,
            ti_between_snaps: (t_between_snaps * time_base_inv) as IntegerTime,
            ti_status: 0,
            ti_between_status: (t_status * time_base_inv) as IntegerTime,
            snap: 0,
            snapshot_prefix: prefix.to_string(),
            particle_motion,
        })
    }

    /// Run this simulation
    pub fn run(&mut self, space: &mut Space) -> Result<(), IoError> {
        // Start by saving the initial state
        self.dump(space)?;

        while self.t_current < self.t_end {
            // Print info?
            if self.ti_between_status == 0 || self.ti_current >= self.ti_status {
                println!(
                    "Running at t={:.4e}. Stepping forward in time by: {:.4e}",
                    self.t_current,
                    make_timestep(self.ti_current - self.ti_old, self.time_base)
                );
                while self.ti_between_status != 0 && self.ti_current >= self.ti_status {
                    self.ti_status += self.ti_between_status;
                }
            }

            // Do we need to save a snapshot?
            if self.ti_next > self.ti_snap {
                if self.ti_current < self.ti_snap {
                    // Drift to snapshot time
                    self.step(space, self.ti_snap)
                }
                self.dump(space)?;
            }

            // take a step
            self.step(space, self.ti_next);
        }

        // Save the final state of the simulation
        self.dump(space)?;

        Ok(())
    }

    fn step(&mut self, space: &mut Space, ti_next: IntegerTime) {
        let dti = ti_next - self.ti_current;

        if self.runner.use_half_step() {
            assert!(dti % 2 == 0, "Integer timestep not divisable by 2!");
            let half_dti = dti / 2;
            self.ti_old = self.ti_current;
            self.ti_current += half_dti;
            self.runner.half_step1(self, space);
            self.ti_current += half_dti;
            self.ti_next = self.runner.half_step2(self, space);
        } else {
            self.ti_old = self.ti_current;
            self.ti_current = ti_next;
            self.ti_next = self.runner.step(self, space);
        }

        let dt = make_timestep(dti, self.time_base);
        self.t_current += dt;
    }

    fn dump(&mut self, space: &mut Space) -> Result<(), IoError> {
        println!("Writing snapshot at t={}!", self.t_current);
        let f = File::create(&format!(
            "output/{}{:04}.txt",
            self.snapshot_prefix, self.snap
        ))?;
        let mut f = BufWriter::new(f);
        space.dump(&mut f)?;
        self.ti_snap += self.ti_between_snaps;
        self.snap += 1;

        Ok(())
    }

    pub fn ti_current(&self) -> IntegerTime {
        self.ti_current
    }

    pub fn ti_old(&self) -> IntegerTime {
        self.ti_old
    }

    pub fn runner(&self) -> &Runner {
        &self.runner
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

    pub fn dt_min(&self) -> f64 {
        self.dt_min
    }

    pub fn dt_max(&self) -> f64 {
        self.dt_max
    }

    pub fn with_gravity(&self) -> bool {
        self.gravity_solver.is_some()
    }
}
