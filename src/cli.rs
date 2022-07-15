extern crate clap;

use std::path;
use clap::Parser;

#[derive(Parser)]
pub struct Cli {
    /// The path to the config file to read
    #[clap(parse(from_os_str))]
    pub config: path::PathBuf,
}