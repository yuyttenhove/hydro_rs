use std::{
    fs::File,
    io::{BufWriter, Error as IoError, Write},
};

use yaml_rust::Yaml;

use crate::{
    engine::Engine,
    equation_of_state::EquationOfState,
    part::Part,
    physical_quantities::{Conserved, Primitives},
    slope_limiters::{cell_wide_limiter, pairwise_limiter},
    timeline::{
        get_integer_time_end, make_integer_timestep, make_timestep, IntegerTime, MAX_NR_TIMESTEPS,
        NUM_TIME_BINS, TIME_BIN_NEIGHBOUR_MAX_DELTA_BIN,
    }, physical_constants::NEWTON_G_IN_SI,
};

#[derive(Clone, Copy)]
pub enum Boundary {
    Periodic,
    Reflective,
    Open,
    Vacuum,
    Spherical,
}

pub struct Space {
    parts: Vec<Part>,
    boundary: Boundary,
    box_size: f64,
    num_parts: usize,
    eos: EquationOfState,
}

impl Space {
    pub fn parts(&self) -> &[Part] {
        &self.parts[1..self.num_parts + 1]
    }

    pub fn parts_mut(&mut self) -> &mut [Part] {
        &mut self.parts[1..self.num_parts + 1]
    }

    /// Constructs a space from given ic's (tuples of position, density, velocity and pressure) and
    /// boundary conditions.
    pub fn from_ic(ic: &[(f64, f64, f64, f64)], space_cfg: &Yaml, eos: EquationOfState) -> Self {
        // read config
        let box_size = space_cfg["box_size"].as_f64().unwrap_or(1.);
        let periodic = space_cfg["periodic"].as_bool().unwrap_or(true);

        // Get our own mutable copy of the ICs
        let mut ic = ic.to_vec();
        // Wrap particles to be inside box:
        for properties in ic.iter_mut() {
            properties.0 *= box_size;
            while properties.0 < 0. {
                properties.0 += box_size
            }
            while properties.0 >= box_size {
                properties.0 -= box_size
            }
        }

        // create vector of parts for space
        let mut parts = vec![Part::default(); ic.len() + 2];
        // Copy positions and primitives to particles
        for (idx, properties) in ic.iter().enumerate() {
            let part = &mut parts[idx + 1];
            part.x = properties.0;
            part.primitives = Primitives::new(properties.1, properties.2, properties.3);
        }

        // create space
        #[cfg(dimensionality = "1D")]
        let boundary = if periodic {
            Boundary::Periodic
        } else {
            Boundary::Reflective
        };
        #[cfg(any(dimensionality = "2D", dimensionality = "3D"))]
        let boundary = Boundary::Spherical;
        let mut space = Space {
            parts,
            boundary,
            box_size,
            num_parts: ic.len(),
            eos,
        };
        // sort particles
        space.sort();
        // Set up the primitive variables of the boundary particles
        space.apply_boundary_condition();
        // Set up the conserved quantities
        space.first_init_parts();

        // return
        space
    }

    pub fn apply_boundary_condition(&mut self) {
        match self.boundary {
            Boundary::Periodic => {
                self.parts[0] = self.parts[self.num_parts].clone();
                self.parts[0].x -= self.box_size;
                self.parts[0].centroid -= self.box_size;
                self.parts[self.num_parts + 1] = self.parts[1].clone();
                self.parts[self.num_parts + 1].x += self.box_size;
                self.parts[self.num_parts + 1].centroid += self.box_size;
            }
            Boundary::Reflective => {
                self.parts[0] = self.parts[1].clone();
                self.parts[0].reflect(0.).reflect_quantities();

                self.parts[self.num_parts + 1] = self.parts[self.num_parts].clone();
                self.parts[self.num_parts + 1].reflect(self.box_size).reflect_quantities();
            }
            Boundary::Spherical => {
                // Reflective at center, vacuum at edge
                self.parts[0] = self.parts[1].clone();
                self.parts[0].reflect(0.).reflect_quantities();

                self.parts[self.num_parts + 1] = self.parts[self.num_parts].clone();
                self.parts[self.num_parts + 1].reflect(self.box_size);
                self.parts[self.num_parts + 1].primitives = Primitives::vacuum();
                self.parts[self.num_parts + 1].conserved = Conserved::vacuum();
            }
            _ => unimplemented!(),
        }
    }

    /// Do the volume calculation for all the active parts in the space
    pub fn volume_calculation(&mut self, engine: &Engine) {
        for i in 1..self.num_parts + 1 {
            let part = &self.parts[i];
            if !part.is_active_flux(engine) {
                continue;
            }
            let x_left = self.parts[i - 1].x;
            let x_right = self.parts[i + 1].x;
            let part = &mut self.parts[i];
            part.volume = 0.5 * (x_right - x_left);
            part.centroid = 0.25 * (x_left + 2. * part.x + x_right);
            debug_assert!(self.parts[i].volume >= 0.);
        }
    }

    pub fn convert_conserved_to_primitive(&mut self, engine: &Engine) {
        let eos = self.eos;
        for part in self.parts_mut() {
            if part.is_active_primitive_calculation(engine) {
                part.convert_conserved_to_primitive(eos);

                // This also invalidates the extrapolations
                part.extrapolations = Primitives::vacuum();
            }
        }
    }

    /// Convert the primitive quantities to conserved quantities. This is only done when creating the space from ic's.
    fn first_init_parts(&mut self) {
        // Calculate the volume of *all* particles
        for i in 1..self.num_parts + 1 {
            let x_left = self.parts[i - 1].x;
            let x_right = self.parts[i + 1].x;
            let part = &mut self.parts[i];
            part.volume = 0.5 * (x_right - x_left);
            part.centroid = 0.25 * (x_left + 2. * part.x + x_right);
            debug_assert!(self.parts[i].volume >= 0.);
        }

        // Calculate the conserved quantities
        let eos = self.eos;
        for part in self.parts_mut() {
            part.conserved = Conserved::from_primitives(&part.primitives, part.physical_volume(), eos);
        }

        // re-apply BC, now with updated conserved quantities
        self.apply_boundary_condition();
    }

    /// Sort the parts in space according to their x coordinate
    pub fn sort(&mut self) {
        self.parts_mut()
            .sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap())
    }

    /// Do the flux exchange in a non symmetrical manner
    /// (each particle calculates the fluxes it recieves independently of
    /// its neighbours using its own timestep).
    pub fn flux_exchange_non_symmetrical(&mut self, engine: &Engine) {
        for i in 1..self.num_parts + 1 {
            let part = &self.parts[i];
            let left = &self.parts[i - 1];
            let right = &self.parts[i + 1];

            if !part.is_active_flux(engine) {
                continue;
            }

            // Flux from the left neighbour
            let dt = part.dt;
            let dx = 0.5 * (part.x - left.x);
            let primitives = pairwise_limiter(
                part.primitives,
                left.primitives,
                part.primitives - dx * part.gradients + part.extrapolations,
            );
            let primitives_left = pairwise_limiter(
                left.primitives,
                part.primitives,
                left.primitives + dx * left.gradients + left.extrapolations,
            );
            let v_face = 0.5 * (left.v + part.v);
            let flux = engine.solver.solve_for_flux(
                &primitives_left.boost(-v_face),
                &primitives.boost(-v_face),
                v_face,
                &self.eos,
            );
            let mut fluxes = dt * flux;
            let mut m_flux = dx * flux.mass();

            // Flux from the right neighbour
            let dx = 0.5 * (right.x - part.x);
            let primitives = pairwise_limiter(
                part.primitives,
                right.primitives,
                part.primitives + dx * part.gradients + part.extrapolations,
            );
            let primitives_right = pairwise_limiter(
                right.primitives,
                part.primitives,
                right.primitives - dx * right.gradients + right.extrapolations,
            );
            let v_face = 0.5 * (right.v + part.v);
            let flux = engine.solver.solve_for_flux(
                &primitives.boost(-v_face),
                &primitives_right.boost(-v_face),
                v_face,
                &self.eos,
            );
            fluxes -= dt * flux;
            m_flux -= dx * flux.mass();

            self.parts[i].fluxes += fluxes;
            self.parts[i].gravity_mflux += m_flux;
        }
    }

    /// Do flux exchange between neighbouring particles
    pub fn flux_exchange(&mut self, engine: &Engine) {
        for i in 0..self.num_parts + 1 {
            let left = &self.parts[i];
            let right = &self.parts[i + 1];

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
            let dx_left = 0.5 * left.volume;
            let dx_right = 0.5 * right.volume;
            let dx = dx_left + dx_right;

            // Gradient extrapolation
            let primitives_left = pairwise_limiter(
                    left.primitives,
            right.primitives,
            left.primitives + dx_left * left.gradients + left.extrapolations,
            );
            let primitives_right = pairwise_limiter(
                    right.primitives,
            left.primitives,
            right.primitives - dx_right * right.gradients + right.extrapolations,
            );

            // Boost the primitives to the frame of reference of the interface:
            let v_face = 0.5 * (left.v + right.v);
            let fluxes = engine.solver.solve_for_flux(
                    &primitives_left.boost(-v_face),
            &primitives_right.boost(-v_face),
            v_face,
            &self.eos,
            );

            // update accumulated fluxes
            self.parts[i].fluxes -= dt * fluxes;
            self.parts[i].gravity_mflux -= dx * fluxes.mass();

            self.parts[i + 1].fluxes += dt * fluxes;
            self.parts[i + 1].gravity_mflux += dx * fluxes.mass();
        }
    }

    /// Apply the accumulated fluxes to all active particles
    pub fn apply_flux(&mut self, engine: &Engine) {
        for part in self.parts_mut() {
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
        let box_size = self.box_size;
        let boundary = self.boundary;
        let mut to_remove = vec![];
        for (idx, part) in self.parts_mut().iter_mut().enumerate() {
            if part.x < 0. || part.x >= box_size {
                match boundary {
                    Boundary::Periodic => {
                        while part.x < 0. {
                            part.x += box_size;
                        }
                        while part.x >= box_size {
                            part.x -= box_size;
                        }
                    }
                    Boundary::Spherical => {
                        if part.x < 0. {
                            part.x = -part.x;
                        }
                        if part.x >= box_size {
                            to_remove.push(idx + 1);
                        }
                    }
                    _ => todo!(),
                }
            }
        }
        // remove particles if needed
        for idx in to_remove {
            self.parts.remove(idx);
            self.num_parts -= 1;
        }
    }

    /// Estimate the gradients for all particles
    pub fn gradient_estimate(&mut self, engine: &Engine) {
        for i in 1..self.num_parts + 1 {
            let part = &self.parts[i];
            if !part.is_active_primitive_calculation(engine) {
                continue;
            }

            self.parts[i].gradients = {
                let left = &self.parts[i - 1];
                let right = &self.parts[i + 1];
                let dx = right.x - left.x;
                let dx_inv = 1. / dx;
                let dx_left = 0.5 * (left.x - part.x);
                let dx_right = 0.5 * (right.x - part.x);

                let gradients = dx_inv * (right.primitives - left.primitives);

                cell_wide_limiter(
                    gradients,
                    part.primitives,
                    left.primitives,
                    right.primitives,
                    dx_left,
                    dx_right,
                )
            };
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
                    part.timestep(engine.cfl_criterion(), &self.eos, &engine.particle_motion);
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
        // First collect info about the timesteps of the neighbouring particles
        for i in 1..self.num_parts + 1 {
            // Get a slice of neighbouring parts
            let parts = &mut self.parts[i - 1..=i + 1];

            // This particles timebin
            let mut wakeup = parts[1].timebin;

            if parts[0].is_ending(engine) {
                wakeup = wakeup.min(parts[0].timebin + TIME_BIN_NEIGHBOUR_MAX_DELTA_BIN);
            }
            if parts[2].is_ending(engine) {
                wakeup = wakeup.min(parts[2].timebin + TIME_BIN_NEIGHBOUR_MAX_DELTA_BIN);
            }

            parts[1].wakeup = wakeup;
        }

        // Now apply the limiter
        for part in self.parts_mut() {
            part.timestep_limit(engine);
        }
    }

    /// Collect the gravitational accellerations in the particles
    pub fn self_gravity(&mut self, engine: &Engine) {
        if !engine.with_gravity {
            return;
        }

        let mut m_tot = 0.;
        for part in self.parts_mut() {
            let m_cell = m_tot + part.primitives.density() * part.half_physical_volume();
            let r2 = part.centroid.powi(2);
            part.a_grav = 1. * m_cell / r2;
            m_tot += part.primitives.density() * part.physical_volume();
        }
    }

    /// Apply the first half kick (gravity) to the particles
    pub fn kick1(&mut self, engine: &Engine) {
        if !engine.with_gravity {
            return;
        }
        for part in self.parts_mut() {
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
        for part in self.parts_mut() {
            if part.is_active(engine) {
                part.grav_kick();
                part.reset_fluxes();
            }
        }
    }

    #[cfg(any(dimensionality = "2D", dimensionality = "3D"))]
    pub fn add_spherical_source_term(&mut self, engine: &Engine) {
        let eos = self.eos;
        for part in self.parts_mut() {
            if part.is_active(engine) {
                part.add_spherical_source_term(&eos);
            }
        }
    }

    /// Prepare the particles for the next timestep
    pub fn prepare(&mut self, engine: &Engine) {
        self.apply_boundary_condition();
        for part in self.parts_mut() {
            if part.is_active(engine) {
                part.reset_fluxes();
            }
        }
    }

    /// Dump snapshot of space at the current time
    pub fn dump(&mut self, f: &mut BufWriter<File>) -> Result<(), IoError> {
        writeln!(
            f,
            "# x (m)\trho (kg m^-3)\tv (m s^-1)\tP (kg m^-1 s^-2)\ta (m s^-2)\tu (J / kg)\tS\ttime (s)"
        )?;
        for part in self.parts() {
            let internal_energy = part.internal_energy();
            let density = part.primitives.density();
            let entropy = self
                .eos
                .gas_entropy_from_internal_energy(internal_energy, density);
            writeln!(
                f,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                part.x,
                density,
                part.primitives.velocity(),
                part.primitives.pressure(),
                part.a_grav,
                internal_energy,
                entropy,
                part.dt,
            )?;
        }

        Ok(())
    }

    pub fn self_check(&self) {
        for part in self.parts() {
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
    use assert_approx_eq::assert_approx_eq;
    use yaml_rust::YamlLoader;

    use crate::equation_of_state::EquationOfState;

    use super::Space;

    const IC: [(f64, f64, f64, f64); 4] = [
        (0.25, 1., 0.5, 1.),
        (0.65, 0.125, 0.5, 0.1),
        (0.75, 0.125, 0.5, 0.1),
        (0.85, 0.125, 0.5, 0.1),
    ];
    const GAMMA: f64 = 5. / 3.;
    const CFG_STR: &'_ str = "box_size: 2. \nperiodic: true";

    #[test]
    #[cfg(dimensionality="1D")]
    fn test_init() {
        let eos = EquationOfState::Ideal { gamma: GAMMA };
        let config = &YamlLoader::load_from_str(CFG_STR).unwrap()[0];
        let space = Space::from_ic(&IC, config, eos);
        assert_eq!(space.parts.len(), 6);
        assert_eq!(space.num_parts, 4);
        // Check volumes
        assert_approx_eq!(space.parts[0].volume, 0.5, space.parts[0].volume * 1e-8);
        assert_approx_eq!(space.parts[1].volume, 0.8, space.parts[1].volume * 1e-8);
        assert_approx_eq!(space.parts[2].volume, 0.5, space.parts[2].volume * 1e-8);
        assert_approx_eq!(space.parts[3].volume, 0.2, space.parts[3].volume * 1e-8);
        assert_approx_eq!(space.parts[4].volume, 0.5, space.parts[4].volume * 1e-8);
        assert_approx_eq!(space.parts[5].volume, 0.8, space.parts[5].volume * 1e-8);
    }
}
