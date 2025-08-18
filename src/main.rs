use anyhow::Result;
use ppo::run_session;

mod ppo;
mod dqn;

fn main() -> Result<()> {
    println!("Hello, world!");

    run_session()?;

    Ok(())
}
