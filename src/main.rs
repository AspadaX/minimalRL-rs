use anyhow::Result;
use ppo::run_session;

mod ppo;

fn main() -> Result<()> {
    println!("Hello, world!");

    run_session()?;

    Ok(())
}
