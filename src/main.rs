use std::io::{stdin, Stdin};

use anyhow::Result;

mod ppo;
mod dqn;
mod a2c;

fn main() -> Result<()> {
    println!("Please choose an algorithm to run (ppo, dqn): ");
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
    
    println!("{} is not implemented! Pleaes input an implemented algorithm to begin with.", input);
    Ok(())
}
