#[derive(Debug, Clone)]
pub struct Data {
    pub state: [f32; 4], // Current state, correspond to `s` in the original Python code. Below are the same.
    pub action: u8,      // Action taken, correspond to `a`
    pub reward: f32,     // Reward received, correspond to `r`
    pub next_state: [f32; 4], // Next state, correspond to `s_prime`
    pub action_prob: f64, // Action probability, correspond to `prob_a`
    pub done: bool,      // Episode done flag, correspond to `done`
}