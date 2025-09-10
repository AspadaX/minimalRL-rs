use anyhow::Result;
use burn::{
    backend::{ndarray::NdArrayDevice, Autodiff, NdArray},
    module::{AutodiffModule, Module},
    nn::{
        loss::HuberLossConfig, Linear, LinearConfig, Relu
    },
    optim::{
        adaptor::OptimizerAdaptor, decay::WeightDecayConfig, Adam, AdamConfig, GradientsParams, Optimizer, SimpleOptimizer
    },
    prelude::Backend,
    tensor::{activation::softmax, backend::AutodiffBackend, cast::ToElement, Int, Tensor, TensorData},
};
use gym_rs::{
    core::{ActionReward, Env},
    envs::classical_control::cartpole::{CartPoleEnv, CartPoleObservation},
    utils::renderer::RenderMode,
};
use rand::{Rng, rng};

use crate::shared::{data_structs::Data, replay_buffer::{self, ReplayBuffer}, utilities::sample_action};

// Hyperparameters
const LEARNING_RATE: f32 = 0.0002;
const UPDATE_INTERVAL: usize = 10;
const GAMMA: f32 = 0.98;
const MAX_TRAIN_EPISODES: usize = 10000;
const PRINT_INTERVAL: usize = UPDATE_INTERVAL * 100;

#[derive(Debug, Module)]
pub struct ActorCritic<B: Backend> {
    fully_connected_layer: Linear<B>,
    policy_fully_connected_layer: Linear<B>,
    value_function_fully_connected_layer: Linear<B>,
    relu: Relu,
}

impl<B> ActorCritic<B>
where
    B: Backend + AutodiffBackend,
{
    pub fn new(device: &B::Device) -> Self {
        Self {
            fully_connected_layer: LinearConfig::new(4, 256).init(device),
            policy_fully_connected_layer: LinearConfig::new(256, 2).init(device),
            value_function_fully_connected_layer: LinearConfig::new(256, 1).init(device),
            relu: Relu::new(),
        }
    }

    /// softmax dimension is default to 1
    pub fn use_policy_function<const D: usize>(
        &self,
        x: Tensor<B, D>,
        softmax_dimension: Option<usize>,
    ) -> Tensor<B, D> {
        let x: Tensor<B, D> = self.relu.forward(self.fully_connected_layer.forward(x));
        let x: Tensor<B, D> = self.policy_fully_connected_layer.forward(x);

        softmax(x, softmax_dimension.unwrap_or(1))
    }

    pub fn use_value_function<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let x: Tensor<B, D> = self.relu.forward(self.fully_connected_layer.forward(x));

        self.value_function_fully_connected_layer.forward(x)
    }
    
    pub fn train<O>(&mut self, optimizer: &mut OptimizerAdaptor<O, Self, B>, replay_buffer: &mut ReplayBuffer, device: &B::Device)
    where
        O: SimpleOptimizer<B::InnerBackend>,
    {
        // Sample a batch from replay buffer
        let batch = replay_buffer.sample_batch::<B, UPDATE_INTERVAL>(device);

        // Compute TD target and advantage (delta)
        let temporal_difference_target = batch.rewards
            + GAMMA * self.use_value_function(batch.next_states) * batch.dones;
        let delta = temporal_difference_target.clone() - self.use_value_function(batch.states.clone());

        // Policy loss component
        let policy = self.use_policy_function(batch.states.clone(), Some(1));
        let policy_action = policy.gather(1, batch.actions);

        // Value loss component (Huber)
        let huber_loss: burn::nn::loss::HuberLoss = HuberLossConfig::new(1.0).init();
        let loss = -policy_action.log() * delta.detach()
            + huber_loss.forward_no_reduction(
                self.use_value_function(batch.states),
                temporal_difference_target.detach(),
            );

        // Back-propagate
        let gradients = loss.backward();

        // Optimizer step — returns updated model
        let updated = optimizer.step(
            LEARNING_RATE as f64,
            self.clone(),
            GradientsParams::from_grads(gradients, self),
        );

        // Update self in-place
        *self = updated;
    }
}

pub fn convert_to_array(state: CartPoleObservation) -> [f32; 4] {
    let mut array_state: [f32; 4] = [0.0; 4];
    for (index, element) in Vec::from(state).iter().enumerate() {
        array_state[index] = element.to_owned().to_f32();
    }

    array_state
}

pub fn run_session() -> Result<()> {
    let mut env: CartPoleEnv = CartPoleEnv::new(RenderMode::None);

    let device: NdArrayDevice = NdArrayDevice::default();
    // Initialize the neural networks
    let mut model: ActorCritic<Autodiff<NdArray>> = ActorCritic::new(&device);
    let mut optimizer: OptimizerAdaptor<Adam, ActorCritic<Autodiff<NdArray>>, Autodiff<NdArray>> =
        AdamConfig::new()
            .with_beta_1(0.9)
            .with_beta_2(0.999)
            .with_epsilon(1e-08)
            .with_weight_decay(Some(WeightDecayConfig::new(0.0)))
            .init();
    let mut replay_buffer = ReplayBuffer::new::<MAX_TRAIN_EPISODES>();
    let mut score: f32 = 0.0;

    for n_epi in 0..MAX_TRAIN_EPISODES {
        // Reset environment at the beginning of each episode
        let (mut state, _) = env.reset(Some(rng().random()), false, None);
        let mut step_in_episode: usize = 0;
        let mut done: bool = false;

        // Roll out until the episode terminates
        while !done {
            // Choose an action from the current policy
            let probability: Tensor<Autodiff<NdArray>, 1> =
                model.use_policy_function::<1>(Tensor::from(convert_to_array(state)), Some(0));
            let action: usize = sample_action(
                &probability
                    .to_data()
                    .convert::<f32>()
                    .to_vec::<f32>()
                    .unwrap(),
            )?;

            // Step the environment
            let result: ActionReward<CartPoleObservation, ()> = env.step(action);

            // Accumulate reward for logging
            score += result.reward.to_f32();

            // Push transition into replay buffer (scale reward just like Py impl: r/100)
            replay_buffer.put(
                Data::from_step_result(
                    state,                              // s
                    result.observation,                 // s'
                    action as u8,                       // a
                    0.0,                                // prob_a (unused)
                    result.done,                        // done
                    result.reward.to_f32() / 100.0,     // r (scaled)
                ),
            );

            // Update current state & done flag
            state = result.observation;
            done = result.done;
            step_in_episode += 1;

            // 6. Perform an update every `UPDATE_INTERVAL` steps OR when episode terminates
            if (step_in_episode % UPDATE_INTERVAL == 0) || done {
                // Only train when we have enough samples in the buffer
                if replay_buffer.size() >= UPDATE_INTERVAL {
                    model.train(&mut optimizer, &mut replay_buffer, &device);
                }
            }
        }

        if (n_epi % PRINT_INTERVAL == 0) && (n_epi != 0) {
            println!(
                "# of episode :{}, avg score : {}",
                n_epi,
                score / PRINT_INTERVAL as f32
            );
            score = 0.0;
        }
    }

    env.close();

    Ok(())
}
