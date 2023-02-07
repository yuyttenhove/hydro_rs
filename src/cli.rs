extern crate clap;

use clap::Parser;
use std::path;

#[derive(Parser)]
pub struct Cli {
    /// The path to the config file to read
    #[clap(parse(from_os_str))]
    pub config: path::PathBuf,
}
