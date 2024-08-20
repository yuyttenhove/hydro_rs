use common::{
    get_engine, get_eos, get_space, ENGINE_CONFIG, EOS_CONFIG, GRAVITY_CONFIG, HYDRO_CONFIG,
    SNAPSHOTS_CONFIG, TIME_INTEGRATION_CONFIG,
};
use float_cmp::{approx_eq, assert_approx_eq};
use glam::DVec3;
use hydro_rs::{Engine, EquationOfState, InitialConditions, Space};

mod common;

const SPACE_CONFIG: &'static str = r##"
boundary: "periodic"
"##;

fn get_ic_2d(v: DVec3, eos: &EquationOfState) -> InitialConditions {
    InitialConditions::from_fn(DVec3::ONE, 9, 2, eos, None, |position| {
        let pressure = if approx_eq!(f64, position.x, 0.5) && approx_eq!(f64, position.y, 0.5) {
            100.
        } else {
            1.
        };
        (1., v, pressure)
    })
}

macro_rules! assert_approx_eq_dvec3 {
    ($a:expr, $b:expr) => {
        assert_approx_eq!(f64, $a.x, $b.x);
        assert_approx_eq!(f64, $a.y, $b.y);
        assert_approx_eq!(f64, $a.z, $b.z);
    };
    ($a:expr, $b:expr, epsilon=$eps:expr) => {
        assert_approx_eq!(f64, $a.x, $b.x, epsilon = $eps);
        assert_approx_eq!(f64, $a.y, $b.y, epsilon = $eps);
        assert_approx_eq!(f64, $a.z, $b.z, epsilon = $eps);
    };
}

fn drift(space: &mut Space, space_boosted: &mut Space, engine: &Engine) {
    let dt = space.parts()[0].dt;
    space.drift(dt, 0.5 * dt, engine);
    space_boosted.drift(dt, 0.5 * dt, engine);

    space
        .parts()
        .iter()
        .zip(space_boosted.parts().iter())
        .for_each(|(part, part_boosted)| {
            assert_approx_eq!(
                f64,
                part.extrapolations.density(),
                part_boosted.extrapolations.density(),
                epsilon = 1e-10
            );
            assert_approx_eq_dvec3!(
                part.extrapolations.velocity(),
                part_boosted.extrapolations.velocity(),
                epsilon = 1e-10
            );
            assert_approx_eq!(
                f64,
                part.extrapolations.pressure(),
                part_boosted.extrapolations.pressure(),
                epsilon = 1e-10
            );
        });
}

fn timestep(space: &mut Space, space_boosted: &mut Space, engine: &Engine) -> u64 {
    let dti = space.timestep(&engine);
    let dti_boosted = space_boosted.timestep(&engine);
    assert_eq!(dti, dti_boosted);

    space.timestep_limiter(&engine);
    space_boosted.timestep_limiter(&engine);

    space
        .parts()
        .iter()
        .zip(space_boosted.parts().iter())
        .for_each(|(part, part_boosted)| {
            assert_approx_eq!(f64, part.dt, part_boosted.dt);
        });

    dti
}

fn convert_conserved_to_primitive(
    space: &mut Space,
    space_boosted: &mut Space,
    engine: &Engine,
    boost_velocity: DVec3,
) {
    space.convert_conserved_to_primitive(engine);
    space_boosted.convert_conserved_to_primitive(engine);
    space
        .parts()
        .iter()
        .zip(space_boosted.parts().iter())
        .for_each(|(part, part_boosted)| {
            assert_approx_eq!(
                f64,
                part.primitives.density(),
                part_boosted.primitives.density(),
                epsilon = 1e-10
            );
            assert_approx_eq_dvec3!(
                part.primitives.velocity(),
                part_boosted.primitives.velocity() - boost_velocity,
                epsilon = 1e-10
            );
            assert_approx_eq!(
                f64,
                part.primitives.pressure(),
                part_boosted.primitives.pressure(),
                epsilon = 1e-10
            );
        });
}

fn gradient_estimate(space: &mut Space, space_boosted: &mut Space, engine: &Engine) {
    space.gradient_estimate(engine);
    space_boosted.gradient_estimate(engine);
    space
        .parts()
        .iter()
        .zip(space_boosted.parts().iter())
        .for_each(|(part, part_boosted)| {
            assert_approx_eq_dvec3!(
                part.gradients[0],
                part_boosted.gradients[0],
                epsilon = 1e-10
            );
            assert_approx_eq_dvec3!(
                part.gradients[1],
                part_boosted.gradients[1],
                epsilon = 1e-10
            );
            assert_approx_eq_dvec3!(
                part.gradients[2],
                part_boosted.gradients[2],
                epsilon = 1e-10
            );
            assert_approx_eq_dvec3!(
                part.gradients[3],
                part_boosted.gradients[3],
                epsilon = 1e-10
            );
            assert_approx_eq_dvec3!(
                part.gradients[4],
                part_boosted.gradients[4],
                epsilon = 1e-10
            );
        });
}

fn volume_calculation(space: &mut Space, engine: &Engine, space_boosted: &mut Space) {
    space.volume_calculation(engine);
    space_boosted.volume_calculation(engine);
    space
        .parts()
        .iter()
        .zip(space_boosted.parts().iter())
        .for_each(|(part, part_boosted)| {
            assert_approx_eq!(f64, part.volume, part_boosted.volume, epsilon = 1e-10);
        });
}

fn flux_exchange(
    space: &mut Space,
    space_boosted: &mut Space,
    engine: &Engine,
    boost_velocity: DVec3,
) {
    space.flux_exchange(&engine);
    space_boosted.flux_exchange(&engine);

    space.apply_flux(&engine);
    space_boosted.apply_flux(&engine);
    space
        .parts()
        .iter()
        .zip(space_boosted.parts().iter())
        .for_each(|(part, part_boosted)| {
            assert_approx_eq!(
                f64,
                part.conserved.mass(),
                part_boosted.conserved.mass(),
                epsilon = 1e-10
            );
            let deboosted_momentum =
                part_boosted.conserved.momentum() - part_boosted.conserved.mass() * boost_velocity;
            assert_approx_eq_dvec3!(
                part.conserved.momentum(),
                deboosted_momentum,
                epsilon = 1e-10
            );
            let deboosted_energy = part_boosted.conserved.energy()
                - part_boosted.conserved.momentum().dot(boost_velocity)
                + 0.5 * part_boosted.conserved.mass() * boost_velocity.length_squared();
            assert_approx_eq!(
                f64,
                part.conserved.energy(),
                deboosted_energy,
                epsilon = 1e-10
            );
        });
}

#[test]
fn test_invariance() {
    let engine = get_engine(
        ENGINE_CONFIG,
        TIME_INTEGRATION_CONFIG,
        SNAPSHOTS_CONFIG,
        HYDRO_CONFIG,
        GRAVITY_CONFIG,
    );
    let eos = get_eos(EOS_CONFIG);
    let ic = get_ic_2d(DVec3::ZERO, &eos);
    let boost_velocity = 10. * DVec3::X;
    let ic_boosted = get_ic_2d(boost_velocity, &eos);
    let mut space = get_space(SPACE_CONFIG, ic, eos);
    let mut space_boosted = get_space(SPACE_CONFIG, ic_boosted, eos);

    gradient_estimate(&mut space, &mut space_boosted, &engine);
    let _dti = timestep(&mut space, &mut space_boosted, &engine);

    drift(&mut space, &mut space_boosted, &engine);
    volume_calculation(&mut space, &engine, &mut space_boosted);
    flux_exchange(&mut space, &mut space_boosted, &engine, boost_velocity);
    convert_conserved_to_primitive(&mut space, &mut space_boosted, &engine, boost_velocity);
    gradient_estimate(&mut space, &mut space_boosted, &engine);
    let _dti = timestep(&mut space, &mut space_boosted, &engine);

    drift(&mut space, &mut space_boosted, &engine);
    volume_calculation(&mut space, &engine, &mut space_boosted);
    flux_exchange(&mut space, &mut space_boosted, &engine, boost_velocity);
    convert_conserved_to_primitive(&mut space, &mut space_boosted, &engine, boost_velocity);
}
