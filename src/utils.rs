pub trait Round {
    fn round_to(&self, decimal_places: u8) -> Self;
}

impl Round for f64 {
    fn round_to(&self, decimal_places: u8) -> Self {
        let factor = 10.0f64.powi(decimal_places as i32);
        (self * factor).round() as f64 / factor
    }
}