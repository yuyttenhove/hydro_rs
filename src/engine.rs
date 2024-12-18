use std::collections::VecDeque;

use crate::{
    finite_volume_solver::FiniteVolumeSolver, gravity::GravitySolver, runner::Runner, space::Space, timeline::*
};

#[derive(Copy, Clone)]
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

pub struct TimestepInfo {
    pub(crate) ti_old: IntegerTime,
    pub ti_current: IntegerTime,
    ti_next: IntegerTime,
    pub dt_min: f64,
    pub dt_max: f64,
    pub(crate) time_base: f64,
    pub(crate) time_base_inv: f64,
}

impl TimestepInfo {
    pub fn init(time_base: f64, dt_min: f64, dt_max: f64) -> Self {
        Self {
            ti_old: 0,
            ti_current: 0,
            ti_next: 0,
            time_base,
            time_base_inv: 1. / time_base,
            dt_min, dt_max,
        }
    }

    pub(crate) fn dt_from_dti(&self, dti: IntegerTime) -> f64 {
        make_timestep(dti, self.time_base)
    }
    pub(crate) fn dt_from_bin(&self, timebin: Timebin) -> f64 {
        self.dt_from_dti(get_integer_timestep(timebin))
    }

    /// Must be called at the end of the timestep
    pub(crate) fn bin_is_ending(&self, timebin: Timebin) -> bool {
        timebin <= get_max_active_bin(self.ti_current)
    }

    pub(crate) fn bin_is_halfway(&self, timebin: Timebin) -> bool {
        let dti = self.ti_current - self.ti_old;
        !self.bin_is_ending(timebin)
            && timebin <= get_max_active_bin(self.ti_old + 2 * dti)
    }

    /// must be called at the beginning of a timestep (e.g. kick1, limiter)
    pub(crate) fn bin_is_starting(&self, timebin: Timebin) -> bool {
        timebin <= get_max_active_bin(self.ti_current)
    }

    pub(crate) fn get_integer_time_end(&self, timebin: Timebin) -> IntegerTime {
        get_integer_time_end(self.ti_current, timebin)
    }
}

pub struct Engine {
    runner: Box<dyn Runner>,
    finite_volume_solver: Box<dyn FiniteVolumeSolver>,
    gravity_solver: Option<Box<dyn GravitySolver>>,
    pub(crate) timestep_info: TimestepInfo,
    t_current: f64,
    t_end: f64,
    step_count: usize,
    sync_points: VecDeque<SyncPoint>,
    sync_all: bool,
    ti_snap: IntegerTime,
    ti_between_snaps: IntegerTime,
    ti_status: IntegerTime,
    ti_between_status: IntegerTime,
    snap: u32,
    snapshot_prefix: String,
    save_faces: bool,
    pub(crate) particle_motion: ParticleMotion,
}

impl Engine {
    pub fn new(
        runner: Box<dyn Runner>,
        finite_volume_solver: Box<dyn FiniteVolumeSolver>,
        gravity_solver: Option<Box<dyn GravitySolver>>,
        t_end: f64,
        dt_min: f64,
        dt_max: f64,
        sync_all: bool,
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
            finite_volume_solver, 
            gravity_solver,
            t_end,
            t_current: 0.0,
            timestep_info: TimestepInfo::init(time_base, dt_min, dt_max),
            step_count: 0,
            sync_points,
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
                sync_point.ti >= self.timestep_info.ti_current,
                "Trying to drift backwards in time!"
            );
            let dti = sync_point.ti - self.timestep_info.ti_current;
            if dti > 0 {
                let dt = self.timestep_info.dt_from_dti(dti);
                space.drift(dt);
                self.timestep_info.ti_current += dti;
                self.t_current = make_timestep(self.timestep_info.ti_current, self.time_base());
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
        let ti_next = self.runner.step(space, &self.finite_volume_solver, &self.gravity_solver, &self.timestep_info, self.sync_all, self.particle_motion);
        self.queue_step(ti_next, SyncPointType::Step);
        self.update_ti_next(ti_next, space);
    }

    fn half_step1(&self, space: &mut Space) {
        assert!(self.runner.use_half_step());
        self.runner.half_step1(space, &self.finite_volume_solver, &self.gravity_solver, &self.timestep_info, self.sync_all, self.particle_motion);
    }

    fn half_step2(&mut self, space: &mut Space) {
        assert!(self.runner.use_half_step());
        let ti_next = self.runner.half_step2(space, &self.finite_volume_solver, &self.gravity_solver, &self.timestep_info, self.sync_all, self.particle_motion);
        let dti = ti_next - self.ti_current();
        assert_eq!(dti % 2, 0, "Encountered indivisible time-step!");
        let ti_half_next = self.ti_current() + dti / 2;
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
        space.dump(&self.timestep_info, self.save_faces, filename)?;
        self.snap += 1;

        Ok(())
    }

    fn status(&mut self, space: &Space) {
        if self.ti_status < self.ti_next() {
            println!(
                "{}\t{}\t{:.8e}\t{}\t{:.8e}\t{}",
                self.step_count,
                space.parts().len(),
                self.t_current,
                self.ti_current(),
                make_timestep(self.ti_next() - self.ti_old(), self.time_base()),
                space.status(&self.timestep_info),
            );

            while self.ti_between_status != 0 && self.ti_next() >= self.ti_status {
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
            self.ti_current(),
            self.ti_next(),
            "Updating ti_next while not at the end of a timestep!"
        );
        self.timestep_info.ti_old = self.ti_next();
        self.timestep_info.ti_next = ti_next;
        self.step_count += 1;
        self.status(space);
    }

    pub(crate) fn ti_next(&self) -> IntegerTime {
        self.timestep_info.ti_next
    }

    pub(crate) fn ti_current(&self) -> IntegerTime {
        self.timestep_info.ti_current
    }

    pub(crate) fn ti_old(&self) -> IntegerTime {
        self.timestep_info.ti_old
    }

    pub(crate) fn time_base(&self) -> f64 {
        self.timestep_info.time_base
    }

    pub fn sync_all(&self) -> bool {
        self.sync_all
    }
    
    pub fn timestep_info(&self) -> &TimestepInfo {
        &self.timestep_info
    }
    
    pub fn particle_motion(&self) -> &ParticleMotion {
        &self.particle_motion
    }
}
