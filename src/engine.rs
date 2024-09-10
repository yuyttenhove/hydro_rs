use std::collections::VecDeque;

use crate::{
    gravity::GravitySolver,
    part::Particle,
    riemann_solver::RiemannFluxSolver,
    space::Space,
    time_integration::{Iact, Runner},
    timeline::*,
};

pub enum ParticleMotion {
    Fixed,
    Steer,
    SteerPakmor,
    Fluid,
}

enum SyncPointType {
    Dump,
    HalfStep1,
    HalfStep2,
    Step,
}

struct SyncPoint {
    kind: SyncPointType,
    ti: IntegerTime,
}

impl SyncPoint {
    fn new(ti: IntegerTime, kind: SyncPointType) -> Self {
        Self { ti, kind }
    }
}

pub struct Engine {
    runner: Runner,
    pub(crate) riemann_solver: Box<dyn RiemannFluxSolver>,
    pub(crate) gravity_solver: Option<GravitySolver>,
    t_end: f64,
    t_current: f64,
    ti_old: IntegerTime,
    ti_current: IntegerTime,
    step_count: usize,
    ti_next: IntegerTime,
    sync_points: VecDeque<SyncPoint>,
    time_base: f64,
    time_base_inv: f64,
    cfl_criterion: f64,
    dt_min: f64,
    dt_max: f64,
    sync_all: bool,
    ti_snap: IntegerTime,
    ti_between_snaps: IntegerTime,
    ti_status: IntegerTime,
    ti_between_status: IntegerTime,
    snap: u32,
    snapshot_prefix: String,
    save_faces: bool,
    pub particle_motion: ParticleMotion,
}

impl Engine {
    pub fn new(
        runner: Runner,
        riemann_solver: Box<dyn RiemannFluxSolver>,
        gravity_solver: Option<GravitySolver>,
        t_end: f64,
        dt_min: f64,
        dt_max: f64,
        sync_all: bool,
        cfl_criterion: f64,
        dt_snap: f64,
        snapshot_prefix: &str,
        dt_status: f64,
        save_faces: bool,
        particle_motion: ParticleMotion,
    ) -> Self {
        let time_base = t_end / MAX_NR_TIMESTEPS as f64;
        let time_base_inv = 1. / time_base;

        // Setup intial sync points
        let mut sync_points = VecDeque::new();
        if runner.use_half_step() {
            sync_points.push_back(SyncPoint::new(0, SyncPointType::HalfStep1));
            sync_points.push_back(SyncPoint::new(0, SyncPointType::HalfStep2));
        } else {
            sync_points.push_back(SyncPoint::new(0, SyncPointType::Step));
        }

        Self {
            runner,
            riemann_solver,
            gravity_solver,
            t_end,
            t_current: 0.0,
            ti_old: 0,
            ti_current: 0,
            step_count: 0,
            ti_next: 0,
            sync_points,
            time_base,
            time_base_inv,
            cfl_criterion,
            dt_min,
            dt_max,
            sync_all,
            ti_snap: 0,
            ti_between_snaps: (dt_snap * time_base_inv) as IntegerTime,
            ti_status: 0,
            ti_between_status: (dt_status * time_base_inv) as IntegerTime,
            snap: 0,
            snapshot_prefix: snapshot_prefix.to_string(),
            save_faces,
            particle_motion,
        }
    }

    /// Run this simulation
    pub fn run(&mut self, space: &mut Space) -> Result<(), hdf5::Error> {
        println!("Started running!");

        // Print status line
        println!("timestep \t #Particles \t t_current \t ti_current \t dt \t #Active particles \t min_dt \t max_dt \t total mass \t total energy");

        // run
        while self.t_current < self.t_end {
            // Get next sync point
            let sync_point = self
                .sync_points
                .pop_front()
                .expect("sync_points cannot be empty before end of simulation");

            // Drift to next sync point
            assert!(
                sync_point.ti >= self.ti_current,
                "Trying to drift backwards in time!"
            );
            let dti = sync_point.ti - self.ti_current;
            if dti > 0 {
                self.runner().drift(dti, self, space);
                self.ti_current += dti;
                self.t_current = make_timestep(self.ti_current, self.time_base);
            }

            // What to do at this sync point?
            match sync_point.kind {
                SyncPointType::Dump => self.dump(space)?,
                SyncPointType::HalfStep1 => self.half_step1(space),
                SyncPointType::HalfStep2 => self.half_step2(space),
                SyncPointType::Step => self.step(space),
            };
        }

        // Save the final state of the simulation
        self.dump(space)?;

        Ok(())
    }

    fn step(&mut self, space: &mut Space) {
        assert!(!self.runner.use_half_step());
        let ti_next = self.runner.step(self, space);
        self.queue_step(ti_next, SyncPointType::Step);
        self.update_ti_next(ti_next, space);
    }

    fn half_step1(&self, space: &mut Space) {
        debug_assert!(self.runner.use_half_step());
        self.runner.half_step1(self, space);
    }

    fn half_step2(&mut self, space: &mut Space) {
        debug_assert!(self.runner.use_half_step());
        let ti_next = self.runner.half_step2(self, space);
        let dti = ti_next - self.ti_current;
        assert_eq!(dti % 2, 0, "Encountered indivisible time-step!");
        let ti_half_next = self.ti_current + dti / 2;
        self.queue_step(ti_half_next, SyncPointType::HalfStep1);
        self.queue_step(ti_next, SyncPointType::HalfStep2);
        self.update_ti_next(ti_next, space);
    }

    fn dump(&mut self, space: &mut Space) -> Result<(), hdf5::Error> {
        println!("Writing snapshot at t={}!", self.t_current);
        let filename = format!(
            "output/{}_{}_{:04}.hdf5",
            self.snapshot_prefix,
            self.runner.label(),
            self.snap
        );
        space.dump(&self, &filename)?;
        self.snap += 1;

        Ok(())
    }

    fn status(&mut self, space: &Space) {
        if self.ti_status < self.ti_next {
            println!(
                "{}\t{}\t{:.8e}\t{}\t{:.8e}\t{}",
                self.step_count,
                space.parts().len(),
                self.t_current,
                self.ti_current,
                make_timestep(self.ti_next - self.ti_old, self.time_base),
                space.status(&self),
            );

            while self.ti_between_status != 0 && self.ti_next >= self.ti_status {
                self.ti_status += self.ti_between_status;
            }
        }
    }

    fn queue_step(&mut self, ti: IntegerTime, kind: SyncPointType) {
        while self.ti_snap < ti {
            self.sync_points
                .push_back(SyncPoint::new(self.ti_snap, SyncPointType::Dump));
            self.ti_snap += self.ti_between_snaps;
        }
        self.sync_points.push_back(SyncPoint::new(ti, kind));
    }

    fn update_ti_next(&mut self, ti_next: IntegerTime, space: &Space) {
        debug_assert_eq!(
            self.ti_current, self.ti_next,
            "Updating ti_next while not at the end of a timestep!"
        );
        self.ti_old = self.ti_next;
        self.ti_next = ti_next;
        self.step_count += 1;
        self.status(space);
    }

    pub(crate) fn part_is_active(&self, part: &Particle, iact: Iact) -> bool {
        self.runner.part_is_active(part, iact, self)
    }

    pub(crate) fn ti_current(&self) -> IntegerTime {
        self.ti_current
    }

    pub(crate) fn ti_old(&self) -> IntegerTime {
        self.ti_old
    }

    pub(crate) fn runner(&self) -> &Runner {
        &self.runner
    }

    pub(crate) fn time_base_inv(&self) -> f64 {
        self.time_base_inv
    }

    pub(crate) fn time_base(&self) -> f64 {
        self.time_base
    }

    pub(crate) fn dt(&self, dti: IntegerTime) -> f64 {
        make_timestep(dti, self.time_base)
    }

    pub(crate) fn cfl_criterion(&self) -> f64 {
        self.cfl_criterion
    }

    pub(crate) fn dt_min(&self) -> f64 {
        self.dt_min
    }

    pub(crate) fn dt_max(&self) -> f64 {
        self.dt_max
    }

    pub(crate) fn with_gravity(&self) -> bool {
        self.gravity_solver.is_some()
    }

    pub(crate) fn save_faces(&self) -> bool {
        self.save_faces
    }

    pub(crate) fn sync_all(&self) -> bool {
        self.sync_all
    }
}
