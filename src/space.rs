use std::{
    fs::File,
    io::{BufWriter, Error as IoError, Write},
};

use glam::{DMat3, DVec3};
use meshless_voronoi::{Voronoi, VoronoiFace};
use yaml_rust::Yaml;

use crate::{
    engine::Engine,
    equation_of_state::EquationOfState,
    errors::ConfigError,
    part::Part,
    physical_quantities::{Primitives, StateGradients},
    slope_limiters::pairwise_limiter,
    timeline::{
        get_integer_time_end, make_integer_timestep, make_timestep, IntegerTime, MAX_NR_TIMESTEPS,
        NUM_TIME_BINS, TIME_BIN_NEIGHBOUR_MAX_DELTA_BIN,
    },
    utils::{box_reflect, box_wrap, contains, HydroDimension}, initial_conditions::InitialConditions,
};

#[derive(Clone, Copy, Debug)]
pub enum Boundary {
    Periodic,
    Reflective,
    Open,
    Vacuum,
}

pub struct Space {
    parts: Vec<Part>,
    boundary: Boundary,
    box_size: DVec3,
    voronoi_faces: Vec<VoronoiFace>,
    eos: EquationOfState,
    dimensionality: HydroDimension,
}

impl Space {
    /// Constructs a space from given ic's (partially initialised particles) and
    /// boundary conditions.
    /// 
    /// Particles typically will only have conserved quantities and coordinates set at this point.
    pub fn from_ic(
        initial_conditions: InitialConditions,
        space_cfg: &Yaml,
        eos: EquationOfState,
    ) -> Result<Self, ConfigError> {
        // read config
        let box_size = initial_conditions.box_size();
        let boundary = space_cfg["boundary"]
            .as_str()
            .unwrap_or("reflective")
            .to_string();

        // create space
        let boundary = match boundary.as_str() {
            "periodic" => Boundary::Periodic,
            "reflective" => Boundary::Reflective,
            "open" => Boundary::Open,
            "vacuum" => Boundary::Vacuum,
            _ => return Err(ConfigError::UnknownBoundaryConditions(boundary)),
        };
        let mut space = Space {
            boundary,
            box_size,
            eos,
            dimensionality: initial_conditions.dimensionality(),
            parts: initial_conditions.into_parts(),
            voronoi_faces: vec![],
        };

        // Set up the conserved quantities
        space.first_init_parts();

        // return
        Ok(space)
    }

    fn get_boundary_part(&self, part: &Part, face: &VoronoiFace) -> Part {
        let mut reflected = part.reflect(face.midpoint(), face.normal());
        match self.boundary {
            Boundary::Reflective => {
                reflected
                    .reflect_quantities(face.normal())
                    .reflect_gradients(face.normal());
            }
            Boundary::Open => {
                reflected.reflect_gradients(face.normal());
            }
            Boundary::Vacuum => {
                reflected.primitives = Primitives::vacuum();
                reflected.gradients = StateGradients::zeros();
            }
            _ => panic!(
                "Trying to create boundary particle with {:?} boundary conditions",
                self.boundary
            ),
        }

        reflected
    }

    /// Do the volume calculation for all the active parts in the space
    pub fn volume_calculation(&mut self, engine: &Engine) {
        let generators: Vec<_> = self.parts.iter().map(|p| p.loc()).collect();
        let voronoi = Voronoi::build(&generators, DVec3::ZERO, self.box_size, self.dimensionality.into());
        for (part, voronoi_cell) in self.parts.iter_mut().zip(voronoi.cells().iter()) {
            if !part.is_active_flux(engine) {
                continue;
            }
            part.volume = voronoi_cell.volume();
            part.set_centroid(voronoi_cell.centroid());
            debug_assert!(part.volume >= 0.);
        }
        self.voronoi_faces = voronoi.into_faces();
    }

    pub fn convert_conserved_to_primitive(&mut self, engine: &Engine) {
        let eos = self.eos;
        for part in self.parts.iter_mut() {
            if part.is_active_primitive_calculation(engine) {
                part.convert_conserved_to_primitive(&eos);

                // This also invalidates the extrapolations
                part.reset_gradients();
            }
        }
    }

    /// Convert the primitive quantities to conserved quantities. This is only done when creating the space from ic's.
    fn first_init_parts(&mut self) {
        // Calculate the volume of *all* particles
        let generators: Vec<_> = self.parts.iter().map(|p| p.loc()).collect();
        let voronoi = Voronoi::build(&generators, DVec3::ZERO, self.box_size, self.dimensionality.into());
        for (part, voronoi_cell) in self.parts.iter_mut().zip(voronoi.cells().iter()) {
            part.volume = voronoi_cell.volume();
            part.set_centroid(voronoi_cell.centroid());
            debug_assert!(part.volume >= 0.);
        }
        self.voronoi_faces = voronoi.into_faces();

        // Calculate the conserved quantities
        let eos = self.eos;
        for part in self.parts.iter_mut() {
            part.first_init(&eos);
        }
    }

    /// Do flux exchange between neighbouring particles
    pub fn flux_exchange(&mut self, engine: &Engine) {
        for face in self.voronoi_faces.iter() {
            let left = &self.parts[face.left()];
            let _reflected;
            let right = match face.right() {
                Some(idx) => &self.parts[idx],
                None => {
                    _reflected = self.get_boundary_part(left, face);
                    &_reflected
                }
            };

            // anything to do here?
            // Since we do the flux exchange symmetrically, we only want to do it when the particle with the
            // smallest timestep is active for the flux exchange. This is important for the half drift case,
            // since then the half of the longer timestep might coincide with the full smaller timestep and we
            // do not want to do the flux exchange in that case.
            let dt = if left.is_active_flux(engine) && left.dt <= right.dt {
                left.dt
            } else if right.is_active_flux(engine) && right.dt <= left.dt {
                right.dt
            } else {
                continue;
            };

            // The particle with the smallest timestep is active: Update the fluxes of both (symmetrically)
            // We extrapolate from the centroid of the particles (test).
            let dx_left = face.midpoint() - left.centroid;
            let dx_right = face.midpoint() - right.centroid;
            let dx = right.centroid - left.centroid;

            // Calculate the maximal signal velocity (used for timestep computation)
            let mut v_max = 0.;
            if left.primitives.density() > 0. {
                v_max += self
                    .eos
                    .sound_speed(left.primitives.pressure(), 1. / left.primitives.density());
            }
            if right.primitives.density() > 0. {
                v_max += self
                    .eos
                    .sound_speed(right.primitives.pressure(), 1. / right.primitives.density());
            }
            v_max -= (right.primitives.velocity() - left.primitives.velocity())
                .dot(dx)
                .min(0.);

            // Gradient extrapolation
            let primitives_left = pairwise_limiter(
                left.primitives,
                right.primitives,
                left.primitives + left.gradients.dot(dx_left).into() + left.extrapolations,
            );
            let primitives_right = pairwise_limiter(
                right.primitives,
                left.primitives,
                right.primitives + right.gradients.dot(dx_right).into() + right.extrapolations,
            );

            // Compute interface velocity (Springel (2010), eq. 33):
            let midpoint = 0.5 * (left.x + right.x);
            let fac = (right.v - left.v).dot(face.midpoint() - midpoint) / dx.length_squared();
            let v_face = 0.5 * (left.v + right.v) - fac * dx;

            // Calculate fluxes
            let fluxes = engine.solver.solve_for_flux(
                &primitives_left.boost(-v_face),
                &primitives_right.boost(-v_face),
                v_face,
                face.normal(),
                &self.eos,
            );

            // Reborrow the particles one at a time as mutable
            // Update the flux accumulators
            {
                let left = &mut self.parts[face.left()];
                left.fluxes -= dt * fluxes;
                left.gravity_mflux -= dx * fluxes.mass();
                if left.is_active_flux(engine) {
                    left.max_signal_velocity = v_max.max(left.max_signal_velocity);
                }
            }
            if let Some(idx) = face.right() {
                let right = &mut self.parts[idx];
                right.fluxes += dt * fluxes;
                right.gravity_mflux -= dx * fluxes.mass();
                if right.is_active_flux(engine) {
                    right.max_signal_velocity = v_max.max(right.max_signal_velocity);
                }
            }
        }
    }

    /// Apply the accumulated fluxes to all active particles
    pub fn apply_flux(&mut self, engine: &Engine) {
        for part in self.parts.iter_mut() {
            if part.is_active(engine) {
                part.apply_flux();
            }
        }
    }

    /// drift all particles foward over the given timestep
    pub fn drift(&mut self, dt_drift: f64, dt_extrapolate: f64) {
        let eos = self.eos;
        // drift all particles, including boundary particles
        for part in self.parts.iter_mut() {
            part.drift(dt_drift, dt_extrapolate, &eos);
        }

        // Handle particles that left the box.
        let mut to_remove = vec![];
        for (idx, part) in self.parts.iter_mut().enumerate() {
            if !contains(self.box_size, part.x) {
                match self.boundary {
                    Boundary::Reflective => box_reflect(self.box_size, part),
                    Boundary::Open | Boundary::Vacuum => to_remove.push(idx),
                    Boundary::Periodic => box_wrap(self.box_size, &mut part.x),
                }
            }
        }
        // remove particles if needed
        for idx in to_remove {
            self.parts.remove(idx);
        }
    }

    /// Estimate the gradients for all particles
    pub fn gradient_estimate(&mut self, engine: &Engine) {
        // Loop over the voronoi faces and do the gradient interactions
        for face in self.voronoi_faces.iter() {
            let left = &self.parts[face.left()];
            let _reflected;
            let right = match face.right() {
                Some(idx) => &self.parts[idx],
                None => {
                    _reflected = self.get_boundary_part(left, face);
                    &_reflected
                }
            };

            // Anything to do here?
            if !left.is_active_primitive_calculation(engine)
                && !right.is_active_primitive_calculation(engine)
            {
                continue;
            }

            // Some common calculations
            let ds = right.centroid - left.centroid;
            let w = face.area() / ds.length_squared();
            let matrix_wls = DMat3::from_cols(w * ds.x * ds, w * ds.y * ds, w * ds.z * ds);
            let primitives_left = left.primitives;
            let primitives_right = right.primitives;

            // Reborrow the particles one at a time as mutable
            {
                let left = &mut self.parts[face.left()];
                if left.is_active_primitive_calculation(engine) {
                    left.gradient_estimate(&primitives_right, w, ds, matrix_wls);
                }
            }
            if let Some(idx) = face.right() {
                let right = &mut self.parts[idx];
                if right.is_active_primitive_calculation(engine) {
                    right.gradient_estimate(&primitives_left, w, -ds, matrix_wls);
                }
            }
        }

        // Now loop over the parts and finalize the gradient estimation
        for part in self.parts.iter_mut() {
            if part.is_active_primitive_calculation(engine) {
                part.gradient_finalize()
            }
        }

        // Now, loop again over the faces to collect information for the cell wide limiter
        for face in self.voronoi_faces.iter() {
            let left = &self.parts[face.left()];
            let _reflected;
            let right = match face.right() {
                Some(idx) => &self.parts[idx],
                None => {
                    _reflected = self.get_boundary_part(left, face);
                    &_reflected
                }
            };

            // Anything to do here?
            if !left.is_active_primitive_calculation(engine)
                && !right.is_active_primitive_calculation(engine)
            {
                continue;
            }

            let primitives_left = left.primitives;
            let primitives_right = right.primitives;
            let extrapolated_left =
                left.primitives + left.gradients.dot(face.midpoint() - left.centroid).into();
            let extrapolated_right =
                right.primitives + right.gradients.dot(face.midpoint() - right.centroid).into();

            // Reborrow the particles one at a time as mutable
            {
                let left = &mut self.parts[face.left()];
                if left.is_active_primitive_calculation(engine) {
                    left.gradient_limiter_collect(&primitives_right, &extrapolated_left);
                }
            }
            if let Some(idx) = face.right() {
                let right = &mut self.parts[idx];
                if right.is_active_primitive_calculation(engine) {
                    right.gradient_limiter_collect(&primitives_left, &extrapolated_right);
                }
            }
        }

        // Finally, loop once more over the particles to apply the cell wide limiter
        for part in self.parts.iter_mut() {
            if part.is_active_primitive_calculation(engine) {
                part.gradient_limit();
            }
        }
    }

    /// Calculate the next timestep for all active particles
    pub fn timestep(&mut self, engine: &Engine) -> IntegerTime {
        // Some useful variables
        let ti_current = engine.ti_current();
        let mut ti_end_min = MAX_NR_TIMESTEPS;

        for part in self.parts.iter_mut() {
            if part.is_ending(engine) {
                // Compute new timestep
                let mut dt =
                    part.timestep(engine.cfl_criterion(), &self.eos, &engine.particle_motion, self.dimensionality);
                dt = dt.min(engine.dt_max());
                assert!(
                    dt > engine.dt_min(),
                    "Error: particle requested dt ({}) below dt_min ({})",
                    dt,
                    engine.dt_min()
                );
                let dti = make_integer_timestep(
                    dt,
                    part.timebin,
                    /*TODO*/ NUM_TIME_BINS,
                    ti_current,
                    engine.time_base_inv(),
                );
                part.dt = make_timestep(dti, engine.time_base());
                part.set_timebin(dti);

                ti_end_min = ti_end_min.min(ti_current + dti);
            } else {
                ti_end_min = ti_end_min.min(get_integer_time_end(ti_current, part.timebin));
            }
        }

        debug_assert!(
            ti_end_min > ti_current,
            "Next sync point before current time!"
        );
        ti_end_min
    }

    /// Apply the timestep limiter to the particles
    pub fn timestep_limiter(&mut self, engine: &Engine) {
        // Loop over all the voronoi faces to collect info about the timesteps of the neighbouring particles
        for face in self.voronoi_faces.iter() {
            // Get the two neighbouring parts of the face
            let left = &self.parts[face.left()];
            let _reflected;
            let right = match face.right() {
                Some(idx) => &self.parts[idx],
                None => {
                    _reflected = self.get_boundary_part(left, face);
                    &_reflected
                }
            };

            let left_is_ending = left.is_ending(engine);
            let right_is_ending = right.is_ending(engine);
            let left_timebin = left.timebin;
            let right_timebin = right.timebin;

            // Reborrow the particles one at a time as mutable
            if right_is_ending {
                let left = &mut self.parts[face.left()];
                left.wakeup = left
                    .wakeup
                    .min(right_timebin + TIME_BIN_NEIGHBOUR_MAX_DELTA_BIN);
            }
            if let Some(idx) = face.right() {
                if left_is_ending {
                    let right = &mut self.parts[idx];
                    right.wakeup = right
                        .wakeup
                        .min(left_timebin + TIME_BIN_NEIGHBOUR_MAX_DELTA_BIN);
                }
            }
        }

        // Now apply the limiter
        for part in self.parts.iter_mut() {
            part.timestep_limit(engine);
        }
    }

    /// Collect the gravitational accellerations in the particles
    pub fn self_gravity(&mut self, engine: &Engine) {
        if !engine.with_gravity {
            return;
        }

        unimplemented!()
    }

    /// Apply the first half kick (gravity) to the particles
    pub fn kick1(&mut self, engine: &Engine) {
        if !engine.with_gravity {
            return;
        }
        for part in self.parts.iter_mut() {
            if part.is_active(engine) {
                part.grav_kick();
            }
        }
    }

    /// Apply the second half kick (gravity) to the particles
    pub fn kick2(&mut self, engine: &Engine) {
        if !engine.with_gravity {
            return;
        }
        for part in self.parts.iter_mut() {
            if part.is_active(engine) {
                part.grav_kick();
                part.reset_fluxes();
            }
        }
    }

    /// Prepare the particles for the next timestep
    pub fn prepare(&mut self, engine: &Engine) {
        for part in self.parts.iter_mut() {
            if part.is_active(engine) {
                part.reset_fluxes();
            }
        }
    }

    /// Dump snapshot of space at the current time
    pub fn dump(&mut self, f: &mut BufWriter<File>) -> Result<(), IoError> {
        writeln!(f, "## Particles:")?;
        writeln!(
            f,
            "#p\tx (m)\ty\tz\trho (kg m^-3)\tvx (m s^-1)\tvy\tvz\tP (kg m^-1 s^-2)\tu (J / kg)\tS\ttime (s)"
        )?;
        for part in self.parts.iter() {
            let internal_energy = part.internal_energy();
            let density = part.primitives.density();
            let entropy = self
                .eos
                .gas_entropy_from_internal_energy(internal_energy, density);
            writeln!(
                f,
                "p\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                part.x.x,
                part.x.y,
                part.x.z,
                density,
                part.primitives.velocity().x,
                part.primitives.velocity().y,
                part.primitives.velocity().z,
                part.primitives.pressure(),
                internal_energy,
                entropy,
                part.dt,
            )?;
        }

        if self.dimensionality == HydroDimension::HydroDimension2D {
            writeln!(f, "## Voronoi faces:")?;
            writeln!(f, "#f\ta_x\ta_y\tb_x\t_by")?;
            for face in self.voronoi_faces.iter() {
                let d = face.normal().cross(DVec3::Z);
                let a = face.midpoint() + 0.5 * face.area() * d;
                let b = face.midpoint() - 0.5 * face.area() * d;
                writeln!(f, "f\t{}\t{}\t{}\t{}", a.x, a.y, b.x, b.y)?;
            }
        }

        Ok(())
    }

    pub fn self_check(&self) {
        for part in self.parts.iter() {
            debug_assert!(part.x.is_finite());
            debug_assert!(part.primitives.density().is_finite());
            debug_assert!(part.primitives.velocity().is_finite());
            debug_assert!(part.primitives.pressure().is_finite());
            debug_assert!(part.conserved.mass().is_finite());
            debug_assert!(part.conserved.momentum().is_finite());
            debug_assert!(part.conserved.energy().is_finite());
            debug_assert!(part.fluxes.mass().is_finite());
            debug_assert!(part.fluxes.momentum().is_finite());
            debug_assert!(part.fluxes.energy().is_finite());
            debug_assert!(part.conserved.mass() >= 0.);
            debug_assert!(part.conserved.energy() >= 0.);
        }
    }
}

#[cfg(test)]
mod test {
    use float_cmp::assert_approx_eq;
    use yaml_rust::YamlLoader;

    use crate::equation_of_state::EquationOfState;

    use super::Space;

    const GAMMA: f64 = 5. / 3.;
    const CFG_STR: &'_ str = "boundary: \"reflective\"";
    const IC_CFG: &'_ str = 
r###"type: "config"
num_part: 4
box_size: [1., 1., 1.]
particles:
    x: [0.25, 0.65, 0.75, 0.85]
    mass: [1., 1.125, 1.125, 1.125]
    velocity: [0.5, 0.5, 0.5, 0.5]
    internal_energy: [1., 0.1, 0.1, 0.1]"###;

    #[test]
    fn test_init_1d() {
        use crate::initial_conditions::InitialConditions;

        let eos = EquationOfState::Ideal { gamma: GAMMA };
        let space_config = &YamlLoader::load_from_str(CFG_STR).unwrap()[0];
        let ic_config = &YamlLoader::load_from_str(IC_CFG).unwrap()[0];
        let ics = InitialConditions::new(ic_config, &eos).unwrap();
        let space = Space::from_ic(ics, space_config, eos).expect("Config should be valid!");
        assert_eq!(space.parts.len(), 4);
        // Check volumes
        assert_approx_eq!(f64, space.parts[0].volume, 0.45);
        assert_approx_eq!(f64, space.parts[1].volume, 0.25);
        assert_approx_eq!(f64, space.parts[2].volume, 0.1);
        assert_approx_eq!(f64, space.parts[3].volume, 0.2);
        // Check faces
        assert_eq!(space.voronoi_faces.len(), 21);
        assert_approx_eq!(f64, space.voronoi_faces[0].area(), 1.);
        assert_approx_eq!(f64, space.voronoi_faces[5].area(), 1.);
        assert_approx_eq!(f64, space.voronoi_faces[10].area(), 1.);
        assert_approx_eq!(f64, space.voronoi_faces[15].area(), 1.);
        assert_approx_eq!(f64, space.voronoi_faces[16].area(), 1.);
        assert_approx_eq!(f64, space.voronoi_faces[0].midpoint().x, 0.0);
        assert_approx_eq!(f64, space.voronoi_faces[5].midpoint().x, 0.45);
        assert_approx_eq!(f64, space.voronoi_faces[10].midpoint().x, 0.7);
        assert_approx_eq!(f64, space.voronoi_faces[15].midpoint().x, 0.8);
        assert_approx_eq!(f64, space.voronoi_faces[16].midpoint().x, 1.);
    }
}
