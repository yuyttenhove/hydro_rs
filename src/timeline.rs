use std::mem::size_of;

/// Timebins are stored as `i8`
pub type Timebin = i8;
/// Integertime is stored as `u64'
pub type IntegerTime = u64;

pub const NUM_TIME_BINS: Timebin = 56;
pub const MAX_NR_TIMESTEPS: IntegerTime = get_integer_timestep(NUM_TIME_BINS);
pub const TIME_BIN_INHIBITED: Timebin = NUM_TIME_BINS + 2;
pub const TIME_BIN_NOT_CREATED: Timebin = NUM_TIME_BINS + 3;
pub const TIME_BIN_NOT_AWAKE: Timebin = -NUM_TIME_BINS;
pub const TIME_BIN_NEIGHBOUR_MAX_DELTA_BIN: Timebin = 2;


/// Returns the integer timestep corresponding to a given timebin.
///
/// This is defined as 2^(timebin + 1). This way, timesteps are always
/// power of two multiples of each other.
pub const fn get_integer_timestep(bin: Timebin) -> IntegerTime {
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
pub const fn get_time_bin(time_step: IntegerTime) -> Timebin {
    ((8 * size_of::<IntegerTime>()) as u32 - time_step.leading_zeros() - 2) as i8
}

/// Returns the integer time corresponding to the start of the time-step
/// given by a time-bin.
/// If the current time is a possible beginning for the given time-bin, return
/// the current time minus the time-step size.
pub const fn get_integer_time_begin(ti_current: IntegerTime, bin: Timebin) -> IntegerTime {
    let dti = get_integer_timestep(bin);

    if dti == 0 { return 0; }

    dti * ((ti_current - 1) / dti)
}

/// Returns the integer time corresponding to the end of the time-step
/// given by a time-bin.
/// If the current time is a possible end for the given time-bin, return the
/// current time.
pub const fn get_integer_time_end(ti_current: IntegerTime, bin: Timebin) -> IntegerTime {
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
pub const fn get_max_active_bin(time: IntegerTime) -> Timebin {
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
pub const fn get_min_active_bin(ti_current: IntegerTime, ti_old: IntegerTime) -> Timebin {
    debug_assert!(ti_old < ti_current);

    get_max_active_bin(ti_current - ti_old)
}

/// Compute a valid integer time-step form a given time-step
///
/// We consider the minimal time-bin of any neighbours and prevent particles
/// to differ from it by a fixed constant `time_bin_neighbour_max_delta_bin`.
///
/// If min_ngb_bin is set to `NUM_TIME_BINS`, then no limit from the neighbours
/// is imposed.
pub fn make_integer_timestep(dt: f64, old_bin: Timebin, min_ngb_bin: Timebin, ti_current: IntegerTime, time_base_inv: f64) -> IntegerTime {

    /* Limit timestep given neighbours */
    let mut new_dti = (dt * time_base_inv) as IntegerTime;
    let new_bin = get_time_bin(new_dti).min(min_ngb_bin + TIME_BIN_NEIGHBOUR_MAX_DELTA_BIN);
    new_dti = get_integer_timestep(new_bin);

    /* Limit timestep increase. */
    let current_dti = get_integer_timestep(old_bin);
    if old_bin > 0 { new_dti = new_dti.min(2 * current_dti); }

    /* Put this timestep on the timeline */
    let mut timeline_dti = MAX_NR_TIMESTEPS;
    while new_dti < timeline_dti { timeline_dti /= 2; }
    new_dti = timeline_dti;

    /* Make sure we are allowed a timestep increase. */
    if new_dti > current_dti {
        let ti_end = get_integer_time_end(ti_current, old_bin);
        if (MAX_NR_TIMESTEPS - ti_end) % new_dti != 0 { new_dti = current_dti; }
    }

    debug_assert_ne!(new_dti, 0, "Computed new integer timestep of 0!");

    new_dti
}


pub fn make_timestep(dti: IntegerTime, time_base: f64) -> f64 {
    dti as f64 * time_base
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_conversions() {
        let bin = 8;
        let ti = get_integer_timestep(bin);
        let bin2 = get_time_bin(ti);
        assert_eq!(bin, bin2);
    }
}