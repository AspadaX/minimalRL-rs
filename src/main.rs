use std::io::{Stdin, stdin};

use anyhow::Result;

mod ac;
mod dqn;
mod ppo;
mod sac;
mod shared;

fn main() -> Result<()> {
    ac::run_session();
    return Ok(());

    println!("Please choose an algorithm to run (ppo, dqn, ac): ");
    let mut input: String = String::new();
    let buffer: Stdin = stdin();
    buffer.read_line(&mut input)?;

    println!("You had selected {}.", input);

    if input.trim() == "ppo".to_string() {
        ppo::run_session()?;
        return Ok(());
    }

    if input.trim() == "dqn".to_string() {
        dqn::run_session()?;
        return Ok(());
    }

    if input.trim() == "ac".to_string() {
        ac::run_session()?;
        return Ok(());
    }

    println!(
        "{} is not implemented! Pleaes input an implemented algorithm to begin with.",
        input
    );
    Ok(())
}
