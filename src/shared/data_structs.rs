#[derive(Debug, Clone)]
pub struct Data {
    pub state: [f32; 4], // Current state, correspond to `s` in the original Python code. Below are the same.
    pub action: u8,      // Action taken, correspond to `a`
    pub reward: f32,     // Reward received, correspond to `r`
    pub next_state: [f32; 4], // Next state, correspond to `s_prime`
    pub action_probability: f64, // Action probability, correspond to `prob_a`
    pub done: bool,      // Episode done flag, correspond to `done`
}

impl Data {
    pub fn from_step_result(
        raw_state: CartPoleObservation,
        step: ActionReward<CartPoleObservation, ()>,
        action: u8,
        action_probability: f64,
    ) -> Self {
        let mut original_state: [f32; 4] = [0.0; 4];
        let mut next_state: [f32; 4] = [0.0; 4];

        for (index, element) in Vec::from(step.observation).iter().enumerate() {
            next_state[index] = element.to_f32();
        }

        for (index, element) in Vec::from(raw_state).iter().enumerate() {
            original_state[index] = element.to_f32();
        }

        Self {
            state: original_state,
            action,
            reward: step.reward.to_f32() / 100.0,
            next_state,
            action_probability,
            done: step.done,
        }
    }
}