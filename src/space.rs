use std::{
    fmt::Display,
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
    vec,
};

#[cfg(debug)]
use float_cmp::assert_approx_eq;
use glam::DVec3;
use meshless_voronoi::{Voronoi, VoronoiCell, VoronoiFace};
use rayon::prelude::*;

use crate::{
    cell::Cell,
    engine::{Engine, TimestepInfo},
    flux::FluxInfo,
    gas_law::GasLaw,
    gradients::GradientData,
    initial_conditions::InitialConditions,
    kernels::{Kernel, OneOver},
    macros::{create_attr, create_dataset},
    part::Particle,
    physical_quantities::State,
    riemann_solver::RiemannFluxSolver,
    time_integration::Iact,
    timeline::{
        get_integer_time_end, make_integer_timestep, make_timestep, IntegerTime, MAX_NR_TIMESTEPS,
        NUM_TIME_BINS, TIME_BIN_NEIGHBOUR_MAX_DELTA_BIN,
    },
    utils::contains,
    Dimensionality, GravitySolver, ParticleMotion, Runner,
};
use crate::{
    flux::{flux_exchange, flux_exchange_boundary},
    gradients::LimiterData,
};

#[derive(Clone, Copy, Debug)]
pub enum Boundary {
    Periodic,
    Reflective,
    Open,
    Vacuum,
}

impl Boundary {
    fn periodic(&self) -> bool {
        match self {
            Boundary::Periodic => true,
            _ => false,
        }
    }
}

impl Display for Boundary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Boundary::Periodic => f.write_str("periodic"),
            Boundary::Reflective => f.write_str("reclective"),
            Boundary::Open => f.write_str("open"),
            Boundary::Vacuum => f.write_str("vacuum"),
        }
    }
}

macro_rules! get_other {
    ($space:expr, $part:expr, $idx:expr, $face:expr, $_reflected:expr) => {
        if $idx == $face.left() {
            match $face.right() {
                Some(idx) => &$space.parts[idx],
                None => {
                    $_reflected = $space.get_boundary_part($part, $face);
                    &$_reflected
                }
            }
        } else {
            &$space.parts[$face.left()]
        }
    };
}

pub struct Space {
    parts: Vec<Particle>,
    cells: Vec<Cell>,
    boundary: Boundary,
    box_size: DVec3,
    cell_width: DVec3,
    cell_dim: [usize; 3],
    voronoi_faces: Vec<VoronoiFace>,
    voronoi_cell_face_connections: Vec<usize>,
    eos: GasLaw,
    dimensionality: Dimensionality,
}

impl Space {
    /// Constructs a space from given ic's (partially initialised particles) and
    /// boundary conditions.
    ///
    /// Particles typically will only have conserved quantities and coordinates set at this point.
    pub fn initialize(
        initial_conditions: InitialConditions,
        max_top_level_cells: usize,
        boundary: Boundary,
        eos: GasLaw,
    ) -> Self {
        let box_size = initial_conditions.box_size;
        // Calculate the number of cells along each dimension
        let max_width = box_size.max_element();
        let cwidth = max_width / max_top_level_cells as f64;
        let mut cdim = [
            (box_size.x / cwidth).round() as i32,
            (box_size.y / cwidth).round() as i32,
            (box_size.z / cwidth).round() as i32,
        ];
        match initial_conditions.dimensionality() {
            Dimensionality::OneD => {
                cdim[1] = 1;
                cdim[2] = 1;
            }
            Dimensionality::TwoD => cdim[2] = 1,
            _ => (),
        }
        let cwidth = DVec3::new(
            box_size.x / cdim[0] as f64,
            box_size.y / cdim[1] as f64,
            box_size.z / cdim[2] as f64,
        );

        // Initialize the cells
        let periodic = boundary.periodic();
        let mut cells = vec![];
        for i in 0..cdim[0] {
            for j in 0..cdim[1] {
                for k in 0..cdim[2] {
                    let loc = DVec3::new(
                        i as f64 + cwidth.x,
                        j as f64 * cwidth.y,
                        k as f64 * cwidth.z,
                    );
                    let mut ngb_cid = vec![];
                    let mut ngb_shift = vec![];
                    let di = 1;
                    let dj = match initial_conditions.dimensionality() {
                        Dimensionality::OneD => 0,
                        _ => 1,
                    };
                    let dk = match initial_conditions.dimensionality() {
                        Dimensionality::ThreeD => 1,
                        _ => 0,
                    };
                    for ii in (i - di)..=(i + di) {
                        if !periodic && (ii < 0 || ii >= cdim[0]) {
                            continue;
                        }
                        for jj in (j - dj)..=(j + dj) {
                            if !periodic && (jj < 0 || jj >= cdim[1]) {
                                continue;
                            }
                            for kk in (k - dk)..=(k + dk) {
                                if !periodic && (kk < 0 || kk >= cdim[2]) {
                                    continue;
                                }
                                let mut shift = DVec3::ZERO;
                                let mut iii = ii;
                                if ii < 0 {
                                    shift.x = -box_size.x;
                                    iii += cdim[0];
                                } else if ii >= cdim[0] {
                                    shift.x = box_size.x;
                                    iii -= cdim[0];
                                }
                                let mut jjj = jj;
                                if jj < 0 {
                                    shift.y = -box_size.y;
                                    jjj += cdim[1];
                                } else if jj >= cdim[1] {
                                    shift.y = box_size.y;
                                    jjj -= cdim[1];
                                }
                                let mut kkk = kk;
                                if kk < 0 {
                                    shift.z = -box_size.z;
                                    kkk += cdim[2];
                                } else if kk >= cdim[2] {
                                    shift.z = box_size.z;
                                    kkk -= cdim[2];
                                }

                                if iii == i && jjj == j && kkk == k {
                                    continue;
                                }

                                ngb_cid
                                    .push((iii * cdim[1] * cdim[2] + jjj * cdim[2] + kkk) as usize);
                                ngb_shift.push(shift);
                            }
                        }
                    }
                    cells.push(Cell::new(loc, ngb_cid, ngb_shift));
                }
            }
        }

        // create space

        let mut space = Space {
            cells,
            boundary,
            box_size,
            eos,
            cell_width: cwidth,
            cell_dim: [cdim[0] as usize, cdim[1] as usize, cdim[2] as usize],
            dimensionality: initial_conditions.dimensionality(),
            parts: initial_conditions.into_parts(),
            voronoi_faces: vec![],
            voronoi_cell_face_connections: vec![],
        };

        // Set up the conserved quantities
        space.first_init_parts();

        space
    }

    fn get_boundary_part(&self, part: &Particle, face: &VoronoiFace) -> Particle {
        match self.boundary {
            Boundary::Reflective => part
                .reflect(face.centroid(), face.normal())
                .reflect_quantities(face.normal()),
            Boundary::Open => part.reflect(face.centroid(), face.normal()),
            Boundary::Vacuum => {
                let mut reflected = part.reflect(face.centroid(), face.normal());
                reflected.primitives = State::vacuum();
                reflected
            }
            _ => panic!(
                "Trying to create boundary particle with {:?} boundary conditions",
                self.boundary
            ),
        }
    }

    /// Do the volume calculation for all the active parts in the space
    ///
    /// NOTE: the face counts and offsets of inactive particles are also updated by this method!
    /// This information is used e.g. in the timestep limiter, where we might want to loop over
    /// the active neighbours of an inactive particle
    pub fn volume_calculation(&mut self, runner: &Runner, timestep_info: &TimestepInfo) {
        let generators = self.parts.iter().map(|p| p.loc()).collect::<Vec<_>>();
        let mask = self
            .parts
            .iter()
            .map(|p| runner.part_is_active(p, Iact::Volume, timestep_info))
            .collect::<Vec<_>>();
        let voronoi = Voronoi::build_partial(
            &generators,
            &mask,
            DVec3::ZERO,
            self.box_size,
            self.dimensionality.into(),
            self.boundary.periodic(),
        );

        self.parts
            .par_iter_mut()
            .zip(voronoi.cells().par_iter())
            .zip(mask.par_iter())
            .for_each(|((part, voronoi_cell), is_active)| {
                if *is_active {
                    part.update_geometry(voronoi_cell);
                } else {
                    // just update the face offset and counts for inactive particles
                    part.face_connections_offset = voronoi_cell.face_connections_offset();
                    part.face_count = voronoi_cell.face_count();
                }
            });

        self.voronoi_cell_face_connections = voronoi.cell_face_connections().to_vec();
        self.voronoi_faces = voronoi.into_faces();
    }

    /// Compute voronoi cells of particles using back extrapolated neighbours
    ///
    /// NOTE: Just like the normal volume calculation, we also update the offset and count of
    /// all inactive particles, even if they have no faces
    ///
    /// WARNING: slow, doesnt work with periodic boundary conditions
    pub fn volume_calculation_back_extrapolate(
        &mut self,
        runner: &Runner,
        timestep_info: &TimestepInfo,
    ) {
        if self.boundary.periodic() {
            panic!("Periodic boundary conditions are not supported for back extrapolation volume calculation!")
        }
        // Loop over the parts and construct the back extrapolated positions of its neighbours
        // Construct the voronoi cell of the neigbours
        let cells_faces: Vec<Option<(VoronoiCell, Vec<VoronoiFace>)>> = self
            .parts
            .par_iter()
            .enumerate()
            .map(|(idx, part)| {
                if !runner.part_is_active(part, Iact::Volume, timestep_info) {
                    return None;
                }
                let cell = &self.cells[part.cell_id];
                let dt_ext = -0.5 * part.dt;
                let mut positions = vec![part.loc + dt_ext * part.v];
                let mut ids = vec![idx];

                // Collect back extrapolated positions
                for (ngb_pos, ngb_idx) in cell.nn_iter(part.loc) {
                    if (part.loc - ngb_pos).length_squared()
                        > 4. * part.search_radius * part.search_radius
                    {
                        break;
                    }
                    positions.push(ngb_pos + dt_ext * self.parts[ngb_idx].v);
                    ids.push(ngb_idx);
                }

                // Construct single voronoi cell around central particle
                let mut mask = vec![false; positions.len()];
                mask[0] = true;
                let anchor = DVec3::ZERO;
                let width = self.box_size;
                let vortess = Voronoi::build_partial(
                    &positions,
                    &mask,
                    anchor,
                    width,
                    self.dimensionality.into(),
                    false,
                );

                let cell = vortess.cells()[0].clone();
                // Collect the faces between neigbouring particles if this particle has the smallest timestep
                // OR if the timesteps are equal and this particle has the smallest id
                let faces = cell
                    .faces(&vortess)
                    .filter_map(|face| {
                        let mut localized_face = face.clone();
                        localized_face.set_left(ids[face.left()]);
                        if let Some(right) = face.right() {
                            let right_idx = ids[right];
                            // Only keep a single face between particles.
                            // Prefer the face of the particle with the smallest timestep.
                            // If the timesteps are equal, prefer the particle with the smallest id.
                            match part
                                .dt
                                .partial_cmp(&self.parts[right_idx].dt)
                                .expect("dt should not be NaN!")
                            {
                                std::cmp::Ordering::Greater => return None,
                                std::cmp::Ordering::Equal => {
                                    if idx > right_idx {
                                        return None;
                                    }
                                }
                                std::cmp::Ordering::Less => (),
                            }
                            // Update the right idx as well.
                            localized_face.set_right(right_idx);
                        }
                        Some(localized_face)
                    })
                    .collect();

                Some((cell, faces))
            })
            .collect();

        // Loop over the cells and faces and set the particles properties,
        // and flatten the voronoi faces
        self.voronoi_faces = self
            .parts
            .par_iter_mut()
            .zip(cells_faces.into_par_iter())
            .filter_map(|(part, maybe_cell_faces)| {
                if let Some((cell, faces)) = maybe_cell_faces {
                    // Update the geometry of the particle based on it's voronoi cell
                    // Note that the face connections offset and face count will be overwritten later, but we must set the
                    // centroid, volume and search radius here.
                    part.update_geometry(&cell);
                    Some(faces)
                } else {
                    None
                }
            })
            .flatten()
            .collect::<Vec<_>>();

        // Finally, compute the cell_face_connections
        let mut cell_face_connections: Vec<Vec<usize>> = vec![vec![]; self.parts.len()];
        for (i, face) in self.voronoi_faces.iter().enumerate() {
            let left = face.left();
            cell_face_connections[left].push(i);
            if let Some(right) = face.right() {
                cell_face_connections[right].push(i);
            }
        }
        let mut offset = 0;
        self.voronoi_cell_face_connections = cell_face_connections
            .into_iter()
            .zip(self.parts.iter_mut())
            .map(|(connections, part)| {
                // Overwrite the particles cell-face connections and face counts, based on the actual faces between all particles.
                // Note we do this for inactive particles as well (sets their face count to 0), so no mor faces are linked to them.
                part.face_count = connections.len();
                part.face_connections_offset = offset;
                offset += part.face_count;

                connections
            })
            .flatten()
            .collect();

        // DEBUGGING: Compute faces and cells the normal way and compare
        #[cfg(debug)]
        {
            let faces_back_extrapolate = self.voronoi_faces.clone();
            let cell_face_connections = self.voronoi_cell_face_connections.clone();
            let face_counts: Vec<_> = self.parts.iter().map(|part| part.face_count).collect();
            let face_offsets: Vec<_> = self
                .parts
                .iter()
                .map(|part| part.face_connections_offset)
                .collect();
            let volumes: Vec<_> = self.parts.iter().map(|part| part.volume).collect();
            let centroids: Vec<_> = self.parts.iter().map(|part| part.centroid).collect();

            self.volume_calculation(engine);
            assert_eq!(self.voronoi_faces.len(), faces_back_extrapolate.len());
            assert_eq!(
                self.voronoi_cell_face_connections.len(),
                cell_face_connections.len()
            );

            for (i, part) in self.parts.iter().enumerate() {
                assert_approx_eq!(f64, part.volume, volumes[i]);
                assert_approx_eq!(f64, part.centroid.x, centroids[i].x, ulps = 16);
                assert_approx_eq!(f64, part.centroid.y, centroids[i].y, ulps = 16);
                assert_approx_eq!(
                    f64,
                    part.centroid.z,
                    centroids[i].z,
                    ulps = 8,
                    epsilon = 1e-13
                );
                assert_eq!(part.face_count, face_counts[i]);

                let new_face_idx = &self.voronoi_cell_face_connections
                    [part.face_connections_offset..part.face_connections_offset + part.face_count];
                let new_faces: Vec<_> = new_face_idx
                    .iter()
                    .map(|&idx| &self.voronoi_faces[idx])
                    .collect();
                let old_face_idx =
                    &cell_face_connections[face_offsets[i]..face_offsets[i] + face_counts[i]];
                let old_faces: Vec<_> = old_face_idx
                    .iter()
                    .map(|&idx| &faces_back_extrapolate[idx])
                    .collect();
                for new_face in new_faces {
                    let new_left = new_face.left();
                    let new_right = new_face.right();
                    // loop over the old faces and check whether we find the equivalent face
                    let n_equivalent = old_faces
                        .iter()
                        .filter(|&old_face| {
                            let old_left = old_face.left();
                            let old_right = old_face.right();
                            let matching = if let Some(new_right) = new_right {
                                if let Some(old_right) = old_right {
                                    old_left == new_left && old_right == new_right
                                        || old_left == new_right && old_right == new_left
                                } else {
                                    false
                                }
                            } else {
                                // new_right is also None, check that the left particles and the normals match
                                if old_right.is_none() {
                                    old_left == new_left && old_face.normal().eq(&new_face.normal())
                                } else {
                                    false
                                }
                            };
                            if matching {
                                // Do some extra sanity checks
                                assert_approx_eq!(
                                    f64,
                                    new_face.area(),
                                    old_face.area(),
                                    ulps = 16,
                                    epsilon = 1e-13
                                );
                                assert_approx_eq!(
                                    f64,
                                    new_face.centroid().x,
                                    old_face.centroid().x,
                                    ulps = 16,
                                    epsilon = 1e-13
                                );
                                assert_approx_eq!(
                                    f64,
                                    new_face.centroid().y,
                                    old_face.centroid().y,
                                    ulps = 16,
                                    epsilon = 1e-13
                                );
                                assert_approx_eq!(
                                    f64,
                                    new_face.centroid().z,
                                    old_face.centroid().z,
                                    epsilon = 1e-13
                                );
                            }
                            matching
                        })
                        .count();
                    assert_eq!(n_equivalent, 1)
                }
            }
        }
    }

    /// Drift the particles centroid back to the end of the timestep
    /// This needs to happen after flux exchange, before gradient calculation
    /// in schemes using the back extrapolated volume calculation.
    pub fn drift_centroids_to_current_time(
        &mut self,
        runner: &Runner,
        timestep_info: &TimestepInfo,
    ) {
        self.parts.par_iter_mut().for_each(|part| {
            if runner.part_is_active(part, Iact::Flux, timestep_info) {
                part.centroid += 0.5 * part.v * part.dt;
            }
        })
    }

    pub fn regrid(&mut self) {
        // Sort the parts along their cell index
        self.parts.par_iter_mut().for_each(|part| {
            let i = (part.loc.x / self.cell_width.x).floor() as usize;
            let j = (part.loc.y / self.cell_width.y).floor() as usize;
            let k = (part.loc.z / self.cell_width.z).floor() as usize;
            part.cell_id = (i * self.cell_dim[1] + j) * self.cell_dim[2] + k;
        });
        self.parts.sort_by_key(|part| part.cell_id);

        // Now count the number of parts for each cell
        let mut offset = 0;
        let mut pid = 0;
        for (cid, cell) in self.cells.iter_mut().enumerate() {
            while pid < self.parts.len() && self.parts[pid].cell_id == cid {
                pid += 1;
            }
            cell.part_offset = offset;
            cell.part_count = pid - offset;
            offset = pid;
        }

        // Now update the cells search structures
        let trees = self
            .cells
            .par_iter()
            .map(|cell| cell.build_search_tree(&self.cells, &self.parts))
            .collect::<Vec<_>>();
        self.cells
            .par_iter_mut()
            .zip(trees.into_par_iter())
            .for_each(|(cell, tree)| cell.assign_search_tree(tree));
    }

    pub fn convert_conserved_to_primitive(
        &mut self,
        runner: &Runner,
        timestep_info: &TimestepInfo,
    ) {
        let eos = self.eos;
        self.parts.par_iter_mut().for_each(|part| {
            if runner.part_is_active(part, Iact::Primitive, timestep_info) {
                part.convert_conserved_to_primitive(&eos);

                // This also invalidates the extrapolations
                part.reset_gradients();
            }
        });
    }

    /// Convert the primitive quantities to conserved quantities. This is only done when creating the space from ic's.
    fn first_init_parts(&mut self) {
        // Calculate the volume of *all* particles
        let generators: Vec<_> = self.parts.iter().map(|p| p.loc()).collect();
        let voronoi = Voronoi::build(
            &generators,
            DVec3::ZERO,
            self.box_size,
            self.dimensionality.into(),
            self.boundary.periodic(),
        );

        for (part, voronoi_cell) in self.parts.iter_mut().zip(voronoi.cells().iter()) {
            part.update_geometry(voronoi_cell);
        }

        self.voronoi_cell_face_connections = voronoi.cell_face_connections().to_vec();
        self.voronoi_faces = voronoi.into_faces();

        // Calculate the primitive quantities
        let eos = self.eos;
        for part in self.parts.iter_mut() {
            part.first_init(&eos);
        }
    }

    /// Compute the fluxes for each face (if any).
    fn compute_fluxes<RiemannSolver: RiemannFluxSolver>(
        &self,
        time_extrapolate_fac: f64,
        runner: &Runner,
        timestep_info: &TimestepInfo,
        riemann_solver: &RiemannSolver,
    ) -> Vec<Option<FluxInfo>> {
        self.voronoi_faces
            .par_iter()
            .map(|face| {
                let left = &self.parts[face.left()];
                let left_active = runner.part_is_active(left, Iact::Flux, timestep_info);
                match face.right() {
                    Some(right_idx) => {
                        let right = &self.parts[right_idx];
                        // anything to do here?
                        // Since we do the flux exchange symmetrically, we only want to do it when the particle with the
                        // smallest timestep is active for the flux exchange. This is important for the half drift case,
                        // since then the half of the longer timestep might coincide with the full smaller timestep and
                        // we do not want to do the flux exchange in that case.
                        let dt = if left_active && left.dt <= right.dt {
                            left.dt
                        } else if runner.part_is_active(right, Iact::Flux, timestep_info)
                            && right.dt <= left.dt
                        {
                            right.dt
                        } else {
                            return None;
                        };
                        Some(flux_exchange(
                            left,
                            right,
                            dt,
                            face,
                            time_extrapolate_fac,
                            &self.eos,
                            riemann_solver,
                        ))
                    }
                    None => {
                        if !left_active {
                            return None;
                        }
                        Some(flux_exchange_boundary(
                            left,
                            face,
                            self.boundary,
                            time_extrapolate_fac,
                            &self.eos,
                            riemann_solver,
                        ))
                    }
                }
            })
            .collect()
    }

    /// Add fluxes to the parts
    fn add_fluxes(
        &mut self,
        fluxes: Vec<Option<FluxInfo>>,
        runner: &Runner,
        timestep_info: &TimestepInfo,
    ) {
        self.parts
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, part)| {
                // Loop over faces and add fluxes if any
                let part_is_active = runner.part_is_active(part, Iact::ApplyFlux, timestep_info);
                let offset = part.face_connections_offset;
                let faces = &self.voronoi_cell_face_connections[offset..offset + part.face_count];
                for &face_idx in faces {
                    if let Some(flux_info) = &fluxes[face_idx] {
                        // Left or right?
                        if idx == self.voronoi_faces[face_idx].left() {
                            part.update_fluxes_left(flux_info, part_is_active);
                        } else {
                            part.update_fluxes_right(flux_info, part_is_active);
                        }
                    }
                }
            });
    }

    /// Do flux exchange between neighbouring particles
    /// This method will extrapolate back in time for half the duration of the timestep over which
    /// fluxes are exchanges (depends on the neigbour).
    pub fn flux_exchange<RiemannSolver: RiemannFluxSolver>(
        &mut self,
        runner: &Runner,
        timestep_info: &TimestepInfo,
        riemann_solver: &RiemannSolver,
    ) {
        self.add_fluxes(
            self.compute_fluxes(0.5, runner, timestep_info, riemann_solver),
            runner,
            timestep_info,
        );
    }

    /// Do flux exchange between neighbouring particles, without extrapolating back in time
    pub fn flux_exchange_no_back_extrapolation<RiemannSolver: RiemannFluxSolver>(
        &mut self,
        runner: &Runner,
        timestep_info: &TimestepInfo,
        riemann_solver: &RiemannSolver,
    ) {
        self.add_fluxes(
            self.compute_fluxes(0., runner, timestep_info, riemann_solver),
            runner,
            timestep_info,
        );
    }

    /// Do flux exchange between neighbouring particles according to pakmors hybrid Heun/MUSCL-Hankock scheme
    ///
    /// We first calculate the fluxes without extrapolating and then with extrapolating and take the average of the two.
    pub fn flux_exchange_pakmor_single<RiemannSolver: RiemannFluxSolver>(
        &mut self,
        runner: &Runner,
        timestep_info: &TimestepInfo,
        riemann_solver: &RiemannSolver,
    ) {
        let mut fluxes = self.compute_fluxes(1., runner, timestep_info, riemann_solver);
        let fluxes_end = self.compute_fluxes(0., runner, timestep_info, riemann_solver);
        // Average the fluxes
        fluxes
            .par_iter_mut()
            .zip(fluxes_end.par_iter())
            .for_each(|(flux_start, flux_end)| {
                if let Some(flux_start) = flux_start {
                    let flux_end = flux_end
                        .as_ref()
                        .expect("Cannot be None, since flux start was not None.");
                    *flux_start = FluxInfo {
                        fluxes: 0.5 * (flux_end.fluxes + flux_start.fluxes),
                        mflux: 0.5 * (flux_end.mflux + flux_start.mflux),
                        ..*flux_end
                    };
                }
            });
        // Add fluxes to parts
        self.add_fluxes(fluxes, runner, timestep_info);
    }

    /// Apply the (first or second) half flux (for Heun's method).
    pub fn half_flux_exchange<RiemannSolver: RiemannFluxSolver>(
        &mut self,
        runner: &Runner,
        timestep_info: &TimestepInfo,
        riemann_solver: &RiemannSolver,
    ) {
        // Calculate the fluxes accross each face
        let mut fluxes = self.compute_fluxes(0., runner, timestep_info, riemann_solver);

        // Half the fluxes
        fluxes.par_iter_mut().for_each(|flux_info| {
            if let Some(flux_info) = flux_info {
                flux_info.fluxes = 0.5 * flux_info.fluxes;
                flux_info.mflux = 0.5 * flux_info.mflux;
            }
        });

        self.add_fluxes(fluxes, runner, timestep_info);
    }

    pub fn extrapolate_flux<RiemannSolver: RiemannFluxSolver>(
        &mut self,
        runner: &Runner,
        timestep_info: &TimestepInfo,
        riemann_solver: &RiemannSolver,
    ) {
        let fluxes = self.compute_fluxes(0., runner, timestep_info, riemann_solver);

        let mut d_conserved = vec![State::vacuum(); self.parts.len()];
        for (i, part) in self.parts.iter().enumerate() {
            if !runner.part_is_active(part, Iact::Flux, timestep_info) {
                continue;
            }
            let offset = part.face_connections_offset;
            let faces = &self.voronoi_cell_face_connections[offset..offset + part.face_count];
            for &face_idx in faces {
                if let Some(flux_info) = &fluxes[face_idx] {
                    // Left or right?
                    if i == self.voronoi_faces[face_idx].left() {
                        d_conserved[i] -= 0.5 * flux_info.fluxes;
                    } else {
                        debug_assert_eq!(
                            i,
                            self.voronoi_faces[face_idx]
                                .right()
                                .expect("Should not be boundary face!")
                        );
                        d_conserved[i] += 0.5 * flux_info.fluxes;
                    }
                }
            }
        }

        self.parts
            .iter_mut()
            .zip(d_conserved.into_iter())
            .for_each(|(part, d_conserved)| {
                if part.conserved.mass() + d_conserved.mass() > 0. {
                    let mut new_primitives = State::from_conserved(
                        &(part.conserved + d_conserved),
                        part.volume(),
                        &self.eos,
                    );
                    new_primitives.check_physical();
                    part.primitives = new_primitives;
                }
            });
    }

    /// Apply the accumulated fluxes to all active particles before calculating the primitives
    pub fn apply_flux(&mut self, runner: &Runner, timestep_info: &TimestepInfo) {
        self.parts.par_iter_mut().for_each(|part| {
            if runner.part_is_active(part, Iact::ApplyFlux, timestep_info) {
                part.apply_flux();
            }
        });
    }

    /// Apply the time extrapolations
    pub fn apply_time_extrapolations(&mut self, runner: &Runner, timestep_info: &TimestepInfo) {
        self.parts.par_iter_mut().for_each(|part| {
            if runner.part_is_active(part, Iact::Volume, timestep_info) {
                part.primitives += part.extrapolations;
                part.extrapolations = State::vacuum();
            }
        });
    }

    /// drift all particles forward over the given timestep
    pub fn drift(&mut self, dt_drift: f64, dt_extrapolate: f64) {
        let eos = self.eos;
        // drift all particles, including inactive particles
        self.parts.par_iter_mut().for_each(|part| {
            part.drift(dt_drift, dt_extrapolate, &eos, self.dimensionality);
        });

        // Handle particles that left the box.
        let mut to_remove = vec![];
        for (idx, part) in self.parts.iter_mut().enumerate() {
            if !contains(self.box_size, part.loc, self.dimensionality.into()) {
                match self.boundary {
                    Boundary::Reflective => part.box_reflect(self.box_size, self.dimensionality),
                    Boundary::Open | Boundary::Vacuum => to_remove.push(idx),
                    Boundary::Periodic => part.box_wrap(self.box_size, self.dimensionality),
                }
            }
        }
        // remove particles if needed
        for idx in to_remove {
            self.parts.remove(idx);
        }
    }

    /// Estimate the gradients for all particles
    pub fn gradient_estimate(&mut self, runner: &Runner, timestep_info: &TimestepInfo) {
        // First calculate the gradients in parallel
        let gradients = self
            .parts
            .par_iter()
            .enumerate()
            .map(|(idx, part)| {
                if !runner.part_is_active(part, Iact::Gradient, timestep_info) {
                    return None;
                }

                // Get this particles faces
                let offset = part.face_connections_offset;
                let faces = &self.voronoi_cell_face_connections[offset..offset + part.face_count];

                // Loop over the faces to do the initial gradient calculation
                let mut gradient_data = GradientData::init(self.dimensionality);
                for &face_idx in faces {
                    let face = &self.voronoi_faces[face_idx];

                    let _reflected;
                    let other = get_other!(self, part, idx, face, _reflected);

                    let mut shift = face.shift().unwrap_or(DVec3::ZERO);
                    if let Some(right_idx) = face.right() {
                        if idx == right_idx {
                            shift = -shift;
                        }
                    }
                    let ds = other.centroid + shift - part.centroid;
                    let w = face.area() / ds.length_squared();
                    gradient_data.collect(&part.primitives, &other.primitives, w, ds);
                }
                let mut gradients = gradient_data.finalize();

                // Now loop again over the faces of this cell to collect the limiter info and limit the gradient estimate
                let mut limiter = LimiterData::init(&part.primitives);
                for &face_idx in faces {
                    let face = &self.voronoi_faces[face_idx];

                    let _reflected;
                    let other = get_other!(self, part, idx, face, _reflected);

                    let shift = match face.right() {
                        Some(right_idx) if idx == right_idx => face.shift().unwrap_or(DVec3::ZERO),
                        _ => DVec3::ZERO,
                    };
                    let extrapolated = gradients
                        .dot(face.centroid() - part.centroid - shift)
                        .into();
                    limiter.collect(&other.primitives, &extrapolated)
                }
                limiter.limit(&mut gradients, &part.primitives);

                debug_assert!(gradients.is_finite());
                Some(gradients)
            })
            .collect::<Vec<_>>();

        // Now apply the gradients to the particles
        self.parts
            .par_iter_mut()
            .zip(gradients.par_iter())
            .for_each(|(part, gradient)| {
                if let Some(gradient) = gradient {
                    part.gradients = *gradient;
                    part.gradients_centroid = part.centroid;
                    debug_assert!(runner.part_is_active(part, Iact::Gradient, timestep_info));
                }
            });
    }

    pub fn meshless_gradient_estimate(&mut self, runner: &Runner, timestep_info: &TimestepInfo) {
        // Compute the gradients for all the active parts
        let gradients = self
            .parts
            .par_iter()
            .map(|part| {
                if !runner.part_is_active(part, Iact::Gradient, timestep_info) {
                    return None;
                }

                let cell = &self.cells[part.cell_id];

                // Loop over the nearest neighbours of this particle until we reach the safety radius
                // to compute the gradients
                let mut gradient_data = GradientData::init(self.dimensionality);
                for (loc, id) in cell.nn_iter(part.loc) {
                    let ds = loc - part.loc;
                    let distance_squared = ds.length_squared();
                    debug_assert!(distance_squared > 0.);
                    if distance_squared > part.search_radius * part.search_radius {
                        break;
                    }
                    let ngb_part = &self.parts[id];
                    if distance_squared > ngb_part.search_radius * ngb_part.search_radius {
                        continue;
                    }
                    let dx_centroid = ngb_part.centroid - part.centroid;
                    gradient_data.collect(
                        &part.primitives,
                        &ngb_part.primitives,
                        OneOver(2).kernel(dx_centroid.length(), part.search_radius),
                        dx_centroid,
                    );
                }
                let mut gradients = gradient_data.finalize();

                // Loop over the nearest neighbours of this particle until we reach the safety radius
                // to limit the gradients
                let mut limiter = LimiterData::init(&part.primitives);
                for (loc, id) in cell.nn_iter(part.loc) {
                    let ds = loc - part.loc;
                    let distance_squared = ds.length_squared();
                    if distance_squared > part.search_radius * part.search_radius {
                        break;
                    }
                    let ngb_part = &self.parts[id];
                    if distance_squared > ngb_part.search_radius * ngb_part.search_radius {
                        continue;
                    }
                    let midpoint = part.loc + 0.5 * ds;
                    let extrapolated = gradients.dot(midpoint - part.centroid).into();
                    limiter.collect(&ngb_part.primitives, &extrapolated)
                }
                limiter.limit(&mut gradients, &part.primitives);

                debug_assert!(gradients.is_finite());
                Some(gradients)
            })
            .collect::<Vec<_>>();

        // Now apply the gradients to the particles
        self.parts
            .par_iter_mut()
            .zip(gradients.par_iter())
            .for_each(|(part, gradient)| {
                if let Some(gradient) = gradient {
                    part.gradients = *gradient;
                    part.gradients_centroid = part.centroid;
                    debug_assert!(runner.part_is_active(part, Iact::Gradient, timestep_info));
                }
            });
    }

    pub fn volume_derivative_estimate(&mut self, runner: &Runner, timestep_info: &TimestepInfo) {
        let mut dvol_dt: Vec<Option<f64>> = vec![None; self.parts.len()];
        dvol_dt
            .par_iter_mut()
            .zip(self.parts.par_iter().enumerate())
            .for_each(|(maybe_dvol, (idx, part))| {
                if !runner.part_is_active(part, Iact::Volume, timestep_info) {
                    return;
                }
                // Loop over faces of particle
                let offset = part.face_connections_offset;
                let faces = &self.voronoi_cell_face_connections[offset..offset + part.face_count];
                let mut dvol = 0.;
                for &face_idx in faces {
                    let face = &self.voronoi_faces[face_idx];
                    let left_idx = face.left();
                    if let Some(right_idx) = face.right() {
                        let ngb = if left_idx == idx {
                            &self.parts[right_idx]
                        } else {
                            &self.parts[left_idx]
                        };
                        // Get the relative velocity of the face
                        let v_face = 0.5 * (ngb.v - part.v);
                        let sign = if left_idx == idx { 1. } else { -1. };
                        dvol += sign * face.area() * v_face.dot(face.normal());
                    }
                }
                *maybe_dvol = Some(dvol);
            });

        self.parts
            .par_iter_mut()
            .zip(dvol_dt)
            .for_each(|(part, dvol)| {
                if let Some(dvol) = dvol {
                    part.dvdt = dvol;
                }
            });
    }

    /// Calculate the next timestep for all active particles
    pub fn timestep<R: RiemannFluxSolver>(&mut self, engine: &Engine<R>) -> IntegerTime {
        // Some useful variables
        let ti_current = engine.ti_current();
        let dti_min = AtomicU64::new(MAX_NR_TIMESTEPS);

        self.parts.par_iter_mut().for_each(|part| {
            if engine.timestep_info.bin_is_ending(part.timebin) {
                // Compute new timestep
                let mut dt = part.timestep(
                    engine.cfl_criterion(),
                    &engine.particle_motion,
                    &self.eos,
                    self.dimensionality,
                );
                if let Some(solver) = &engine.gravity_solver {
                    dt = dt.min(solver.get_timestep(part));
                }
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
                part.set_timestep(make_timestep(dti, engine.time_base()), dti);

                dti_min.fetch_min(dti, Ordering::Relaxed);
            } else {
                debug_assert!(
                    !engine.sync_all(),
                    "Found particle not ending it's timestep while syncing timesteps!"
                );
                dti_min.fetch_min(
                    get_integer_time_end(ti_current, part.timebin) - ti_current,
                    Ordering::Relaxed,
                );
            }
        });

        // Update the particles timesteps to the minimal timestep if syncing timesteps
        let dti_min = dti_min.into_inner();
        debug_assert!(dti_min > 0, "Next sync point before current time!");
        let dt = make_timestep(dti_min, engine.time_base());
        if engine.sync_all() {
            for part in self.parts.iter_mut() {
                debug_assert!(
                    engine.timestep_info.bin_is_ending(part.timebin),
                    "Found particle not ending it's timestep while syncing timesteps!"
                );
                part.set_timestep(dt, dti_min);
            }
        }

        ti_current + dti_min
    }

    /// Apply the timestep limiter to the particles
    pub fn timestep_limiter(&mut self, timestep_info: &TimestepInfo) {
        // Loop over all the particles to compute their wakeup times
        let wakeups = self
            .parts
            .par_iter()
            .enumerate()
            .map(|(idx, part)| {
                // Get this particles faces
                let offset = part.face_connections_offset;
                let faces = &self.voronoi_cell_face_connections[offset..offset + part.face_count];

                // Loop over the faces to do the initial gradient calculation
                let mut wakeup = NUM_TIME_BINS;
                for &face_idx in faces {
                    let face = &self.voronoi_faces[face_idx];

                    let _reflected;
                    let other = get_other!(self, part, idx, face, _reflected);
                    if !timestep_info.bin_is_starting(other.timebin) {
                        continue;
                    }

                    wakeup = wakeup.min(other.timebin + TIME_BIN_NEIGHBOUR_MAX_DELTA_BIN);
                }

                wakeup
            })
            .collect::<Vec<_>>();

        // Now apply the timestep limiter
        self.parts
            .par_iter_mut()
            .zip(wakeups.par_iter())
            .for_each(|(part, &wakeup)| {
                part.timestep_limit(wakeup, timestep_info);
            });
    }

    /// Apply the timestep limiter to the particles, in a meshless fashion
    pub fn meshless_timestep_limiter(&mut self, timestep_info: &TimestepInfo) {
        // Loop over all the particles to compute their wakeup times
        let wakeups = self
            .parts
            .par_iter()
            .map(|part| {
                // Loop over the nearest neighbours of this particle until we reach the safety radius
                let mut wakeup = NUM_TIME_BINS;
                for (loc, id) in self.cells[part.cell_id].nn_iter(part.loc) {
                    let ds = loc - part.loc;
                    let distance_squared = ds.length_squared();
                    debug_assert!(distance_squared > 0.);
                    if distance_squared > part.search_radius * part.search_radius {
                        break;
                    }
                    let ngb_part = &self.parts[id];
                    if !timestep_info.bin_is_starting(ngb_part.timebin) {
                        continue;
                    }
                    if distance_squared > ngb_part.search_radius * ngb_part.search_radius {
                        continue;
                    }
                    wakeup = wakeup.min(ngb_part.timebin + TIME_BIN_NEIGHBOUR_MAX_DELTA_BIN);
                }

                wakeup
            })
            .collect::<Vec<_>>();

        // Now apply the timestep limiter
        self.parts
            .par_iter_mut()
            .zip(wakeups.par_iter())
            .for_each(|(part, &wakeup)| {
                part.timestep_limit(wakeup, timestep_info);
            });
    }

    /// Collect the gravitational accelerations in the particles
    pub fn gravity(&mut self, solver: &Option<GravitySolver>) {
        let Some(solver) = solver else {
            return;
        };

        let accelerations = solver.accelerations(&self.parts);
        self.parts
            .par_iter_mut()
            .zip(accelerations.par_iter())
            .for_each(|(p, a)| {
                p.a_grav = *a;
            });
    }

    /// Apply the first half kick (gravity) to the particles
    /// + hydro half kick to particle velocities
    pub fn kick1(&mut self, timestep_info: &TimestepInfo, particle_motion: &ParticleMotion) {
        self.parts.par_iter_mut().for_each(|part| {
            if timestep_info.bin_is_starting(part.timebin) {
                part.hydro_kick1(particle_motion, self.dimensionality);
                part.grav_kick();
            }
        })
    }

    /// Apply the second half kick (gravity) to the particles
    pub fn kick2(&mut self, timestep_info: &TimestepInfo) {
        self.parts.par_iter_mut().for_each(|part| {
            if timestep_info.bin_is_ending(part.timebin) {
                part.grav_kick();
            }
        })
    }

    /// Prepare the particles for the next timestep
    pub fn prepare(&mut self) {
        // Nothing to do here
    }

    pub fn status(&self, timestep_info: &TimestepInfo) -> String {
        let total_mass: f64 = self
            .parts
            .iter()
            .map(|part| part.conserved.mass() + part.fluxes.mass())
            .sum();
        let total_energy: f64 = self
            .parts
            .iter()
            .map(|part| part.conserved.energy() + part.fluxes.energy())
            .sum();
        let min_timestep = self
            .parts
            .iter()
            .map(|part| part.dt)
            .min_by(|a, b| a.partial_cmp(b).expect("Uncomparable timestep found!"))
            .expect("particles cannot be empty!");
        let max_timestep = self
            .parts
            .iter()
            .map(|part| part.dt)
            .max_by(|a, b| a.partial_cmp(b).expect("Uncomparable timestep found!"))
            .expect("particles cannot be empty!");
        let n_active = self
            .parts
            .iter()
            .filter(|part| timestep_info.bin_is_starting(part.timebin))
            .count();
        format!(
            "{}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}",
            n_active, min_timestep, max_timestep, total_mass, total_energy,
        )
    }

    /// Dump snapshot of space at the current time
    pub fn dump<P: AsRef<Path>>(
        &self,
        timestep_info: &TimestepInfo,
        save_faces: bool,
        filename: P,
    ) -> Result<(), hdf5::Error> {
        let file = hdf5::File::create(filename)?;

        // Write header
        let header = file.create_group("Header")?;
        let time = timestep_info.dt_from_dti(timestep_info.ti_current);
        create_attr!(header, [time], "Time")?;
        create_attr!(header, self.box_size.to_array(), "BoxSize")?;
        create_attr!(header, [self.parts.len(), 0, 0, 0, 0], "Numpart_Total")?;
        create_attr!(header, [usize::from(self.dimensionality)], "Dimension")?;

        // Write particle data
        let part_data = file.create_group("PartType0")?;
        create_dataset!(
            part_data,
            self.parts.iter().map(|part| part.loc.to_array()),
            "Coordinates"
        )?;
        create_dataset!(
            part_data,
            self.parts.iter().map(|part| part.conserved.mass()),
            "Masses"
        )?;
        create_dataset!(
            part_data,
            self.parts.iter().map(|part| part.primitives.density()),
            "Densities"
        )?;
        create_dataset!(
            part_data,
            self.parts
                .iter()
                .map(|part| part.conserved.momentum().to_array()),
            "Momenta"
        )?;
        create_dataset!(
            part_data,
            self.parts
                .iter()
                .map(|part| part.primitives.velocity().to_array()),
            "Velocities"
        )?;
        create_dataset!(
            part_data,
            self.parts.iter().map(|part| part.primitives.pressure()),
            "Pressures"
        )?;
        create_dataset!(
            part_data,
            self.parts.iter().map(|part| part.conserved.energy()),
            "TotalEnergies"
        )?;
        create_dataset!(
            part_data,
            self.parts.iter().map(|part| part.internal_energy()),
            "InternalEnergies"
        )?;
        create_dataset!(
            part_data,
            self.parts
                .iter()
                .map(|part| self.eos.gas_entropy_from_internal_energy(
                    part.internal_energy(),
                    part.primitives.density()
                )),
            "Entropies"
        )?;
        create_dataset!(
            part_data,
            self.parts.iter().map(|part| part.volume),
            "Volumes"
        )?;
        create_dataset!(
            part_data,
            self.parts.iter().map(|part| part.centroid.to_array()),
            "Centroids"
        )?;
        create_dataset!(
            part_data,
            self.parts.iter().map(|part| part.v.to_array()),
            "ParticleVelocities"
        )?;
        create_dataset!(
            part_data,
            self.parts.iter().map(|part| part.a_grav.to_array()),
            "GravitationalAccelerations"
        )?;
        create_dataset!(part_data, self.parts.iter().map(|part| part.dt), "Timestep")?;

        if save_faces {
            // Write Voronoi faces
            let face_data = file.create_group("VoronoiFaces")?;
            create_dataset!(
                face_data,
                self.voronoi_faces.iter().map(|face| face.area()),
                "Area"
            )?;
            create_dataset!(
                face_data,
                self.voronoi_faces
                    .iter()
                    .map(|face| face.centroid().to_array()),
                "Centroid"
            )?;
            create_dataset!(
                face_data,
                self.voronoi_faces
                    .iter()
                    .map(|face| face.normal().to_array()),
                "Normal"
            )?;
            create_dataset!(
                face_data,
                self.voronoi_faces.iter().map(|face| face.left()),
                "Left"
            )?;
            create_dataset!(
                face_data,
                self.voronoi_faces.iter().map(|face| match face.right() {
                    Some(right) => right as i64,
                    None => -1,
                }),
                "Right"
            )?;
        }

        file.close()?;

        Ok(())
    }

    pub fn self_check(&self) {
        for part in self.parts.iter() {
            debug_assert!(part.loc.is_finite());
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

    pub fn parts(&self) -> &[Particle] {
        self.parts.as_ref()
    }
}

#[cfg(test)]
mod test {
    use std::f64;

    use crate::gas_law::{EquationOfState, GasLaw};
    use crate::initial_conditions::InitialConditions;
    use crate::riemann_solver::HLLCRiemannSolver;
    use crate::Engine;
    use crate::{Dimensionality, ParticleMotion, Runner};
    use float_cmp::assert_approx_eq;
    use glam::DVec3;

    use super::{Boundary, Space};

    const GAMMA: f64 = 5. / 3.;
    const BOUNDARY: Boundary = Boundary::Reflective;
    const MAX_TOP_LEVEL_CELLS: usize = 3;

    #[test]
    fn test_init_1d() {
        let eos = GasLaw::new(GAMMA, EquationOfState::Ideal);

        // generate some simple 1D ics
        const X: [f64; 4] = [0.25, 0.65, 0.75, 0.85];
        const MASS: [f64; 4] = [1., 1.125, 1.125, 1.125];
        const V_X: [f64; 4] = [0.5, 0.5, 0.5, 0.5];
        const E_INT: [f64; 4] = [1., 0.1, 0.1, 0.1];
        let ics = InitialConditions::empty(
            4,
            (1usize).try_into().expect("dimension is smaller than 3"),
            DVec3::ONE,
            false,
        )
        .set_coordinates(X.into_iter().map(|x| x * DVec3::X).collect())
        .set_masses(Vec::from(&MASS))
        .set_velocities(V_X.into_iter().map(|v| v * DVec3::X).collect())
        .set_internal_energies(Vec::from(&E_INT));
        let space = Space::initialize(ics, MAX_TOP_LEVEL_CELLS, BOUNDARY, eos);
        assert_eq!(space.parts.len(), 4);
        // Check volumes
        assert_approx_eq!(f64, space.parts[0].volume, 0.45, epsilon = 1e-10);
        assert_approx_eq!(f64, space.parts[1].volume, 0.25, epsilon = 1e-10);
        assert_approx_eq!(f64, space.parts[2].volume, 0.1, epsilon = 1e-10);
        assert_approx_eq!(f64, space.parts[3].volume, 0.2, epsilon = 1e-10);
        // Check faces
        assert_eq!(space.voronoi_faces.len(), 5);
        assert_approx_eq!(f64, space.voronoi_faces[0].area(), 1.);
        assert_approx_eq!(f64, space.voronoi_faces[1].area(), 1.);
        assert_approx_eq!(f64, space.voronoi_faces[2].area(), 1.);
        assert_approx_eq!(f64, space.voronoi_faces[3].area(), 1.);
        assert_approx_eq!(f64, space.voronoi_faces[4].area(), 1.);
        assert_approx_eq!(f64, space.voronoi_faces[0].centroid().x, 0.0);
        assert_approx_eq!(f64, space.voronoi_faces[1].centroid().x, 0.45);
        assert_approx_eq!(f64, space.voronoi_faces[2].centroid().x, 0.7);
        assert_approx_eq!(f64, space.voronoi_faces[3].centroid().x, 0.8);
        assert_approx_eq!(f64, space.voronoi_faces[4].centroid().x, 1.);
    }

    #[test]
    fn test_init_2d() {
        let box_size = DVec3::new(2., 2., 1.);
        let num_part = 256;
        let dimensionality = Dimensionality::TwoD;
        let eos = GasLaw::new(GAMMA, EquationOfState::Ideal);

        let mapper = |_| (1., DVec3::ZERO, 1.);
        let ics = InitialConditions::from_fn(
            box_size,
            num_part,
            dimensionality,
            false,
            &eos,
            Some(0.1),
            mapper,
        );

        let space = Space::initialize(ics, MAX_TOP_LEVEL_CELLS, BOUNDARY, eos);
        assert_eq!(space.parts.len(), num_part);
        let mut volume_total = 0.;
        let mut areas = vec![0.; num_part];
        for face in space.voronoi_faces {
            areas[face.left()] += face.area();
            if let Some(right_idx) = face.right() {
                areas[right_idx] += face.area();
            }
        }
        for (part, &area) in space.parts.iter().zip(areas.iter()) {
            let dx = part.loc() - part.centroid;
            assert!(dx.length() < 0.5 * part.search_radius);
            volume_total += part.volume();
            let radius = (f64::consts::FRAC_1_PI * part.volume()).powf(1. / 2.);
            assert!(area > 2. * f64::consts::PI * radius);
        }
        assert_approx_eq!(f64, volume_total, 4., ulps = 8);
    }

    #[test]
    fn test_init_3d() {
        let box_size = DVec3::new(2., 2., 2.);
        let num_part = 512;
        let dimensionality = Dimensionality::ThreeD;
        let eos = GasLaw::new(GAMMA, EquationOfState::Ideal);

        let mapper = |_| (1., DVec3::ZERO, 1.);
        let ics = InitialConditions::from_fn(
            box_size,
            num_part,
            dimensionality,
            false,
            &eos,
            Some(0.1),
            mapper,
        );

        let space = Space::initialize(ics, MAX_TOP_LEVEL_CELLS, BOUNDARY, eos);
        assert_eq!(space.parts.len(), num_part);
        let mut volume_total = 0.;
        let mut areas = vec![0.; num_part];
        for face in space.voronoi_faces {
            areas[face.left()] += face.area();
            if let Some(right_idx) = face.right() {
                areas[right_idx] += face.area();
            }
        }
        for (part, &area) in space.parts.iter().zip(areas.iter()) {
            let dx = part.loc() - part.centroid;
            assert!(dx.length() < 0.5 * part.search_radius);
            volume_total += part.volume();
            let radius = (3. / 4. * f64::consts::FRAC_1_PI * part.volume()).powf(1. / 3.);
            assert!(area > 4. * f64::consts::PI * radius * radius);
        }
        assert_approx_eq!(f64, volume_total, 8., ulps = 8);
    }

    #[test]
    fn test_back_extrapolate_volume() {
        let box_size = DVec3::splat(1.);
        let num_part = 16;
        let dimensionality = Dimensionality::TwoD;

        let eos = GasLaw::new(GAMMA, EquationOfState::Ideal);

        let xvel = 0.5;
        let mapper = |_| (1., xvel * DVec3::X, 1.);
        let ics = InitialConditions::from_fn(
            box_size,
            num_part,
            dimensionality,
            false,
            &eos,
            Some(0.5),
            mapper,
        );

        let mut space = Space::initialize(ics, 1, Boundary::Reflective, eos);
        // Save ICs for later reference
        let parts_old = space.parts.clone();
        let faces_old = space.voronoi_faces.clone();
        let cell_face_connections_old = space.voronoi_cell_face_connections.clone();
        assert_eq!(parts_old.len(), num_part);

        // Compute timesteps...
        let engine = Engine::new(
            Runner::OptimalOrder,
            HLLCRiemannSolver,
            None,
            0.5,
            1e-10,
            1e-2,
            true,
            0.3,
            0.5,
            "test",
            0.5,
            false,
            ParticleMotion::Fluid,
        );
        let ti_end = space.timestep(&engine);
        let dt = engine
            .timestep_info
            .dt_from_dti(ti_end - engine.ti_current());

        // ... and drift
        // (only for half the timestep, so that the back extrapolation in
        // the volume calculation moves the particles back to their original positions).
        space.drift(0.5 * dt, 0.5 * dt);

        // Do back extrapolation volume calculation
        space.regrid();
        let runner = Runner::VolumeBackExtrapolate;
        space.volume_calculation_back_extrapolate(&runner, &engine.timestep_info);
        for (part_old, part) in parts_old.iter().zip(space.parts.iter()) {
            assert_approx_eq!(f64, part_old.volume, part.volume);
            assert_eq!(
                part_old.face_count, part.face_count,
                "Number of faces does not match!"
            );
            assert_eq!(
                part_old.face_connections_offset, part.face_connections_offset,
                "Face connections offset does not match!"
            );
            let faces_part_old = &cell_face_connections_old[part_old.face_connections_offset
                ..part_old.face_connections_offset + part_old.face_count];
            let faces_part = &space.voronoi_cell_face_connections
                [part.face_connections_offset..part.face_connections_offset + part.face_count];
            'outer: for &idx in faces_part_old {
                let face_old = &faces_old[idx];
                let left = face_old.left();
                if let Some(right) = face_old.right() {
                    // loop over new faces to find the same face:
                    for &idx in faces_part {
                        let face = &space.voronoi_faces[idx];
                        if face.left() == left
                            && face.right().is_some()
                            && face.right().unwrap() == right
                        {
                            assert_approx_eq!(f64, face_old.area(), face.area());
                            assert_approx_eq!(f64, face_old.centroid().x, face.centroid().x);
                            assert_approx_eq!(f64, face_old.centroid().y, face.centroid().y);
                            assert_approx_eq!(f64, face_old.centroid().z, face.centroid().z);
                            // found face, continue with next old face
                            continue 'outer;
                        }
                    }
                    panic!("No matching face found!");
                }
            }
        }
    }
}
