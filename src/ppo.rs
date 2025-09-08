use anyhow::Result;
use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::module::Module;
use burn::optim::{Adam, AdamConfig, adaptor::OptimizerAdaptor, decay::WeightDecayConfig};
use burn::tensor::Shape;
use burn::{
    backend::Autodiff,
    nn::{Linear, LinearConfig, Relu, loss::HuberLossConfig},
    optim::{GradientsParams, Optimizer},
    prelude::Backend,
    tensor::{
        Int, Tensor, TensorData, activation::softmax, backend::AutodiffBackend, cast::ToElement,
    },
};
use gym_rs::{
    core::{ActionReward, Env},
    envs::classical_control::cartpole::{CartPoleEnv, CartPoleObservation},
    utils::renderer::RenderMode,
};

use crate::shared::data_structs::{Data, DataBatch};
use crate::shared::utilities::sample_action;

// Hyperparameters
const LEARNING_RATE: f64 = 0.0005;
const GAMMA: f32 = 0.98;
const LAMBDA: f32 = 0.95;
const EPS_CLIP: f32 = 0.1;
const K_EPOCH: usize = 3;
// Time Horizon, the rounds passed before starting a training
const T_HORIZON: usize = 20;

#[derive(Debug, Module)]
pub struct PPOModule<B: Backend> {
    fc1: Linear<B>, // fc -> Fullly Connected
    fc_pi: Linear<B>,
    fc_v: Linear<B>,
}

/// Rust version of the Python PPO implementation
///
/// Reference(s):
/// - CartPole V1: https://gymnasium.farama.org/environments/classic_control/cart_pole/
pub struct PPO<T>
where
    T: AutodiffBackend,
{
    data: Vec<Data>,
    module: PPOModule<T>,
    optimizer: OptimizerAdaptor<Adam, PPOModule<T>, T>,
    device: T::Device,
    relu: Relu,
}

impl<T> PPO<T>
where
    T: AutodiffBackend,
{
    pub fn new(device: T::Device) -> Self {
        // nn module in Burn: https://burn.dev/docs/burn/nn/index.html

        // `LinearConfig` is an equavalance of Linear in torch.nn
        // The difference is that the Rust Burn library requires initializing the actual `Linear` object from a Config builder.
        // In the below references section, I posted the codebase that I learned when handling the `device` variable required
        // by `LinearConfig`'s `init` method.
        //
        // Reference(s):
        // - `LinearConfig` usage: https://github.com/nanoporetech/modkit/blob/bc8a0f118e29aef0dcefdb7aa31044e343a163c9/ochm/src/models.rs#L6
        let fc1: Linear<T> = LinearConfig::new(4, 256).init(&device);
        let fc_pi: Linear<T> = LinearConfig::new(256, 2).init(&device);
        let fc_v: Linear<T> = LinearConfig::new(256, 1).init(&device);

        // The original Python code did not manipulate nor add new flavors to the beta_1 and beta_2 values.
        // Therefore, we use the default value provided by `torch.optim.Adam`
        //
        // Reference(s):
        // - `AdamConfig` usage: https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html#adam
        let optimizer: OptimizerAdaptor<Adam, PPOModule<T>, T> = AdamConfig::new()
            .with_beta_1(0.9)
            .with_beta_2(0.999)
            .with_epsilon(1e-08)
            .with_weight_decay(Some(WeightDecayConfig::new(0.0)))
            .init();

        Self {
            module: PPOModule { fc1, fc_pi, fc_v },
            optimizer,
            data: vec![],
            device,
            relu: Relu::new(),
        }
    }

    /// Add data to the buffer
    ///
    /// The `data` field in `PPO` will be used for training at the end of each time horizon
    pub fn put_data(&mut self, data: Data) {
        self.data.push(data);
    }

    /// Prepare the training data.
    /// According to PPO's design, we use the latest experiences, aka data, to train the model
    pub fn make_batch(&mut self) -> DataBatch<T> {
        let mut states: [[f32; 4]; T_HORIZON] = [[0.0; 4]; T_HORIZON];
        let mut actions: [[u8; 1]; T_HORIZON] = [[0; 1]; T_HORIZON];
        let mut rewards: [[f32; 1]; T_HORIZON] = [[0.0; 1]; T_HORIZON];
        let mut next_states: [[f32; 4]; T_HORIZON] = [[0.0; 4]; T_HORIZON];
        let mut action_probs: [[f64; 1]; T_HORIZON] = [[0.0; 1]; T_HORIZON];
        let mut dones: [[u8; 1]; T_HORIZON] = [[0; 1]; T_HORIZON];

        for (index, data) in self.data.iter().enumerate() {
            states[index] = data.state;
            actions[index] = [data.action];
            rewards[index] = [data.reward];
            next_states[index] = data.next_state;
            action_probs[index] = [data.action_probability];

            if data.done {
                dones[index] = [0];
                continue;
            }

            dones[index] = [1];
        }

        let states_data: Tensor<T, 2> = Tensor::from_data(states, &self.device);
        let actions_data: Tensor<T, 2, Int> = Tensor::from_data(actions, &self.device);
        let rewards_data: Tensor<T, 2> = Tensor::from_data(rewards, &self.device);
        let next_states_data: Tensor<T, 2> = Tensor::from_data(next_states, &self.device);
        let action_probs_data: Tensor<T, 2> = Tensor::from_data(action_probs, &self.device);
        let dones_data: Tensor<T, 2> = Tensor::from_data(dones, &self.device);

        self.data.clear();

        DataBatch {
            states: states_data,
            actions: actions_data,
            rewards: rewards_data,
            next_states: next_states_data,
            action_probabilities: action_probs_data,
            dones: dones_data,
        }
    }

    /// The policy function
    ///
    /// We will use the policy function in two places,
    /// one is when training the model,
    /// another is when inferencing the model.
    ///
    /// Since the two will use different tensor dimensions,
    /// using a constant generic parameter, const X: usize,
    /// will allowing the differences.
    ///
    /// The `x` here will be determined when you compile the code.
    pub fn pi<const X: usize>(
        &mut self,
        x: Tensor<T, X>,
        softmax_dim: Option<usize>,
    ) -> Tensor<T, X> {
        // Softmax dimension is default to 0
        let mut softmax_dimension: usize = 0;
        if let Some(dim) = softmax_dim {
            softmax_dimension = dim;
        }

        let x: Tensor<T, X> = self.module.fc1.forward(x);
        let x: Tensor<T, X> = self.relu.forward(x);
        let x: Tensor<T, X> = self.module.fc_pi.forward(x);

        // Return the probability
        softmax(x, softmax_dimension)
    }

    pub fn v(&mut self, x: Tensor<T, 2>) -> Tensor<T, 2> {
        let x: Tensor<T, 2> = self.module.fc1.forward(x);
        let x: Tensor<T, 2> = self.relu.forward(x);

        self.module.fc_v.forward(x)
    }

    pub fn train_net(&mut self) {
        let data_tensor: DataBatch<T> = self.make_batch();

        for _ in 0..K_EPOCH {
            // Calculate the advantage
            let td_target: Tensor<T, 2> = data_tensor.rewards.clone()
                + GAMMA * self.v(data_tensor.next_states.clone()) * data_tensor.dones.clone();
            let delta: Tensor<T, 2> = td_target.clone() - self.v(data_tensor.states.clone());
            let delta: Tensor<T, 2> = delta.detach();

            let mut advantage_list: [[f32; 1]; T_HORIZON] = [[0.0; 1]; T_HORIZON];
            let mut advantage: f32 = 0.0;
            for (index_delta_t, delta_t) in delta.flip([0]).iter_dim(0).enumerate() {
                advantage =
                    GAMMA * LAMBDA * advantage + delta_t.to_data().to_vec::<f32>().unwrap()[0];
                advantage_list[index_delta_t] = [advantage];
            }

            advantage_list.reverse();
            let advantage_tensor: Tensor<T, 2> = Tensor::from_data(advantage_list, &self.device);

            // Calculate the ratio for clipping
            let pi: Tensor<T, 2> = self.pi(data_tensor.states.clone(), Some(1));
            let pi_action: Tensor<T, 2> = pi.gather(1, data_tensor.actions.clone());
            // We use clamp_min here to prevent NaN values
            // This is equal to a / b
            let ratio: Tensor<T, 2> = (pi_action.clamp_min(1e-8).log()
                - data_tensor
                    .action_probabilities
                    .clone()
                    .clamp_min(1e-8)
                    .log())
            .exp();

            // We have both unclipped and clipped surrogate objective here,
            // then we need to choose the minimal one from them.
            // This is to ensure the knowledge to learn, aka gradient update, won't be too aggressive.
            let unclipped_surrogate_advantage: Tensor<T, 2> =
                ratio.clone() * advantage_tensor.clone();
            let clipped_surrogate_advantage: Tensor<T, 2> =
                Tensor::clamp(ratio, 1.0 - EPS_CLIP, 1.0 + EPS_CLIP) * advantage_tensor.clone();

            // This is also known as `smooth L1 loss` in PyTorch.
            // The 1.0 delta value originates from PyTorch default.
            let huber_loss: burn::nn::loss::HuberLoss = HuberLossConfig::new(1.0).init();

            // We choose the minimal clipped objective by using `min_pair`.
            // Then we calculate the loss with it.
            let loss: Tensor<T, 2> = -unclipped_surrogate_advantage
                .min_pair(clipped_surrogate_advantage)
                + huber_loss
                    .forward_no_reduction(self.v(data_tensor.states.clone()), td_target.detach());

            let gradients: <T as AutodiffBackend>::Gradients = loss.mean().backward();
            let gradients_params: GradientsParams =
                GradientsParams::from_grads(gradients, &self.module);

            self.module = self
                .optimizer
                .step(LEARNING_RATE, self.module.clone(), gradients_params);
        }
    }
}

/// The main training loop
pub fn run_session() -> Result<()> {
    let mut env = CartPoleEnv::new(RenderMode::None);

    let device = NdArrayDevice::default();
    let mut model: PPO<Autodiff<NdArray>> = PPO::new(device);

    let mut score: f32 = 0.0;
    let print_interval: usize = 20;

    for n_epi in 0..10000 {
        let (raw_state, _) = env.reset(None, false, None);
        let mut done: bool = false;

        let mut previous_state: CartPoleObservation = raw_state;

        while !done {
            // Reflect the shape of the state, which is 1-dimensional array with 4 elements
            let state_data: TensorData =
                TensorData::new(Vec::from(previous_state), Shape::new([4]));
            let state: Tensor<Autodiff<NdArray>, 1> = Tensor::from_data(state_data, &device);
            // Feed the state to the policy network
            let probability: Tensor<Autodiff<NdArray>, 1> = model.pi(state, None);
            let probability_vector: Vec<f32> = probability
                .to_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .unwrap();

            let action: usize = sample_action(&probability_vector)?;

            let result: ActionReward<CartPoleObservation, ()> = env.step(action);
            // Now, this turn's observation has become the next turn's previous observation
            done = result.done;

            score += result.reward.to_f32();

            let data: Data = Data::from_step_result(
                previous_state,
                result.observation,
                action as u8,
                probability_vector[action] as f64,
                done,
                result.reward.to_f32() / 100.0,
            );
            model.put_data(data);
            // Update the previous state with the new state we got in this turn
            previous_state = result.observation;

            // We update the model every T_HORIZON steps
            // This is different from the Python implementation,
            // because the Rust implementation uses fixed length for the data buffer.
            if model.data.len() == T_HORIZON {
                model.train_net();
            }
        }

        if (n_epi % print_interval == 0) && (n_epi != 0) {
            println!(
                "# of episode :{}, avg score : {}",
                n_epi,
                score / print_interval as f32
            );
            score = 0.0;
        }
    }

    env.close();

    Ok(())
}
