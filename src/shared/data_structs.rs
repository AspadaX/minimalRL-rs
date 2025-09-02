use burn::{prelude::Backend, tensor::{cast::ToElement, Int, Tensor}};
use gym_rs::{core::ActionReward, envs::classical_control::cartpole::CartPoleObservation};

/// What's included in the data
///
/// The difference is that the Python version does not need to know the data size at "compile time",
/// but Rust does.
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

/// A batch of data
///
/// Tensor<B, 2> reads: a tensor of two dimensions
#[derive(Debug, Clone)]
pub struct DataBatch<B>
where
    B: Backend,
{
    pub states: Tensor<B, 2>,
    pub actions: Tensor<B, 2, Int>,
    pub rewards: Tensor<B, 2>,
    pub next_states: Tensor<B, 2>,
    pub action_probabilities: Tensor<B, 2>,
    pub dones: Tensor<B, 2>,
}
