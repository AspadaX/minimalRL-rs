# minimalRL-rs

A Rust implementation of [minimalRL](https://github.com/seungeunrho/minimalRL), basic reinforcement learning with minimal codes. The Rust implementation is based on [Burn](https://github.com/tracel-ai/burn) and CPU. 

Hopefully, this repo can inspire more people to use Rust in the AI field. 

## Progress

- [X] Merged DQN into the main branch on Aug 25th, 2025

Other implementations are on the way. 

I am still exploring reinforcement learning. If you would like to add other algorithms, please feel free to make a PR! 

## How to use

You need to have Rust installed on your system first. Refer [Rust official website](https://www.rust-lang.org/tools/install) for installation details. 

Clone this repository:
```bash
git clone https://github.com/AspadaX/minimalRL-rs
```

Switch to the repository folder, use `cargo run --release` to run it in release mode. 

Follow the guide there to run an algorithm you want. 

### Notice

The dependency `gym` used by this project works well on Linux systems. It does not work well on macOS, and I didn't test it on a Windows device. For your convenience, you may run the project in a Linux environment. 

In case if you find a way to make `gym` work on macOS, please make a PR or issue to let us know. 

## Implemented Algorithm

- PPO
- DQN

## How to get the most of this repo? 

> "What I cannot create, I do not understand"
> 
> -- Richard Feynman, Nobel prize winner in 1965 for his work developing quantum electrodynamics

You can mimick this repo in a different programming langauge, or make better engineering over the original code, or make a PR to improve this repo. No matter what work you do with this repo, it will become your knowledge. 