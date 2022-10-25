use std::mem::size_of;

pub const NUM_TIME_BINS: i8 = 56;
pub const MAX_NR_TIMESTEPS: u64 = 1 << (NUM_TIME_BINS + 1);
pub const TIME_BIN_INHIBITED: i8 = NUM_TIME_BINS + 2;
pub const TIME_BIN_NOT_CREATED: i8 = NUM_TIME_BINS + 3;
pub const TIME_BIN_NOT_AWAKE: i8 = -NUM_TIME_BINS;
pub const TIME_BIN_NEIGHBOUR_MAX_DELTA_BIN: i8 = 2;

pub type timebin_t = i8;
pub type intergertime_t = u64;

pub const fn get_integer_timestep(bin: timebin_t) -> intergertime_t {
    if bin <= 0 {
        0
    } else {
        1 << (bin + 1)
    }
}

/// Returns the time bin corresponding to a given timestep size.
///
/// Given our definitions, this is log_2 of the time_step rounded down minus one.
///
/// log_2(x) = (number of bits in the type) - (number of leading 0-bits in x) - 1
pub const fn get_time_bin(time_step: intergertime_t) -> timebin_t {
    ((8 * size_of::<intergertime_t>()) as u32 - time_step.leading_ones() - 1) as i8
}

/// Returns the integer time corresponding to the start of the time-step
/// given by a time-bin.
/// If the current time is a possible beginning for the given time-bin, return
/// the current time minus the time-step size.
pub const fn get_integer_time_begin(ti_current: intergertime_t, bin: timebin_t) -> intergertime_t {
    let dti = get_integer_timestep(bin);

    if dti == 0 { return 0; }

    dti * ((ti_current - 1) / dti)
}

/// Returns the integer time corresponding to the end of the time-step
/// given by a time-bin.
/// If the current time is a possible end for the given time-bin, return the
/// current time.
pub const fn get_integer_time_end(ti_current: intergertime_t, bin: timebin_t) -> intergertime_t {
    let dti = get_integer_timestep(bin);

    if dti == 0 { return 0; }

    let modulus = ti_current % dti;

    if modulus == 0 {
        ti_current
    } else {
        ti_current - modulus + dti
    }
}

/// Returns the highest active time bin at a given point on the time line.
///
/// I.e. The highest bin whose corresponding timestep divides the current time.
pub const fn get_max_active_bin(time: intergertime_t) -> timebin_t {
    if time == 0 {
        return NUM_TIME_BINS;
    }

    let mut bin = 1;
    while (1 << (bin + 1)) & time == 0 { bin += 1; }

    bin
}

/// Returns the lowest active time bin at a given point on the time line.
///
/// # Arguments
/// * `ti_current` - The current point on the timeline.
/// * `ti_old` - The last synchronisation point on the timeline.
pub const fn get_min_active_bin(ti_current: intergertime_t, ti_old: intergertime_t) -> timebin_t {
    debug_assert!(ti_old < ti_current);

    get_max_active_bin(ti_current - ti_old)
}
