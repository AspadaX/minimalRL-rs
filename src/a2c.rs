use anyhow::Result;
use burn::{backend::{ndarray::NdArrayDevice, Autodiff, NdArray}, module::Module, nn::{loss::HuberLossConfig, Linear, LinearConfig, Relu}, optim::{adaptor::OptimizerAdaptor, decay::WeightDecayConfig, Adam, AdamConfig, GradientsParams, Optimizer}, prelude::Backend, serde::de::value, tensor::{activation::softmax, cast::ToElement, Tensor, TensorData}};
use gym_rs::{
    core::{ActionReward, Env},
    envs::classical_control::cartpole::{CartPoleEnv, CartPoleObservation},
    utils::renderer::RenderMode,
};
use rand::{rng, Rng};

use crate::utilities::sample_action;


// Hyperparameters
const N_TRAIN_PROCESS: usize = 3;
const LEARNING_RATE: f32 = 0.0002;
const UPDATE_INTERVAL: usize = 5;
const GAMMA: f32 = 0.98;
const MAX_TRAIN_STEPS: usize = 60000;
const PRINT_INTERVAL: usize = UPDATE_INTERVAL * 100;

#[derive(Debug, Module)]
pub struct ActorCritic<B: Backend> {
    fully_connected_layer: Linear<B>,
    policy_fully_connected_layer: Linear<B>,
    value_function_fully_connected_layer: Linear<B>,
}

impl<B> ActorCritic<B> 
where 
    B: Backend
{
    pub fn new(device: &B::Device) -> Self {
        // self.fc1 = nn.Linear(4, 256)
        // self.fc_pi = nn.Linear(256, 2)
        // self.fc_v = nn.Linear(256, 1)
        Self {
            fully_connected_layer: LinearConfig::new(4, 256).init(device), 
            policy_fully_connected_layer: LinearConfig::new(256, 2).init(device), 
            value_function_fully_connected_layer: LinearConfig::new(256, 1).init(device) 
        }
    }
    
    /// softmax dimension is default to 1
    pub fn use_policy_function(&mut self, x: Tensor<B, 1>, softmax_dimension: Option<usize>) -> Tensor<B, 1> {
        let relu = Relu::new();
        let x = relu.forward(self.fully_connected_layer.forward(x));
        let x = self.policy_fully_connected_layer.forward(x);
        
        softmax(x, softmax_dimension.unwrap_or(1))
    }
    
    pub fn use_value_function<const D: usize>(&mut self, x: Tensor<B, D>) -> Tensor<B, D> {
        let relu = Relu::new();
        let x = relu.forward(self.fully_connected_layer.forward(x));
        
        self.value_function_fully_connected_layer.forward(x)
    }
}

pub fn compute_temporarl_difference_target<B: Backend>(
    value_function_result: Tensor<B, 1>, 
    rewards: [[f32; 1]; UPDATE_INTERVAL], 
    dones: [[usize; 1]; UPDATE_INTERVAL],
    device: &B::Device
) -> Tensor<B, 1> {
    let mut temporal_difference_target = Vec::new();

    let value_function_result_scalar = value_function_result.to_data().convert::<f32>().to_vec::<f32>().unwrap()[0];
    let reversed_rewards = rewards.iter().rev();
    let reversed_dones = dones.iter().rev();

    for (reward, done) in reversed_rewards.zip(reversed_dones) {
        let discounted_cumulative_reward = reward[0] + GAMMA * value_function_result_scalar * done[0] as f32;
        temporal_difference_target.push(discounted_cumulative_reward);
    }

    Tensor::from_data(
        TensorData::new(temporal_difference_target, [UPDATE_INTERVAL]), 
        device
    )
}

pub fn run_session() -> Result<()> {
    let mut env: CartPoleEnv = CartPoleEnv::new(RenderMode::None);

    let device: NdArrayDevice = NdArrayDevice::default();
    // Initialize the neural networks
    let mut model: ActorCritic<Autodiff<NdArray>> = ActorCritic::new(&device);
    let mut optimizer: OptimizerAdaptor<Adam, ActorCritic<Autodiff<NdArray>>, Autodiff<NdArray>> = AdamConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_epsilon(1e-08)
        .with_weight_decay(Some(WeightDecayConfig::new(0.0)))
        .init();
    
    let mut score: f32 = 0.0;
    
    let (mut state, _) = env.reset(Some(rng().random()), false, None);
    let mut array_state: [f32; 4] = [0.0; 4];
    for (index, element) in Vec::from(state).iter().enumerate() {
        array_state[index] = element.to_owned().to_f32();
    }
            
    for n_epi in 0..MAX_TRAIN_STEPS {
        let mut states: [[f32; 4]; UPDATE_INTERVAL] = [[0.0; 4]; UPDATE_INTERVAL];
        let mut actions: [[u32; 1]; UPDATE_INTERVAL] = [[0; 1]; UPDATE_INTERVAL];
        let mut rewards: [[f32; 1]; UPDATE_INTERVAL] = [[0.0; 1]; UPDATE_INTERVAL];
        let mut dones: [[usize; 1]; UPDATE_INTERVAL] = [[0; 1]; UPDATE_INTERVAL];
        
        for index in 0..UPDATE_INTERVAL {
            let probability = model.use_policy_function(Tensor::from(array_state), None);
            let action = sample_action(
                &probability
                    .to_data()
                    .convert::<f32>()
                    .to_vec::<f32>()
                    .unwrap()
            )?;
            
            let result: ActionReward<CartPoleObservation, ()> = env.step(action);

            states[index] = array_state;
            actions[index] = [action as u32];
            rewards[index] = [result.reward.to_f32() / 100.0];

            if result.done {
                dones[index] = [1];
            } else {
                dones[index] = [0];
            }
            
            // Record the state we get from this turn
            state = result.observation;
        }
        
        let value_function_result = model.use_value_function(Tensor::from(array_state));
        let temporal_difference_target = compute_temporarl_difference_target(value_function_result, rewards, dones, &device);
        
        let states_tensor = Tensor::from(states);
        let actions_tensor = Tensor::from(actions);
        
        let value_function_results = model.use_value_function(states_tensor.clone());
        let advantage = temporal_difference_target.clone() - value_function_results.clone();
        
        let policy_function_result = model.use_policy_function(states_tensor, None);
        let policy_action = policy_function_result.gather(1, actions_tensor);
        
        // This is also known as `smooth L1 loss` in PyTorch. 
        // The 1.0 delta value originates from PyTorch default.
        let huber_loss: burn::nn::loss::HuberLoss = HuberLossConfig::new(1.0).init();
        let loss = -(policy_action.log() * advantage.clone()).mean() + huber_loss
            .forward_no_reduction(value_function_results.clone(), temporal_difference_target);
        
        let gradients = loss.backward();
        model = optimizer.step(
            LEARNING_RATE as f64, 
            model.clone(), 
            GradientsParams::from_grads(gradients, &model)
        );
        
        if (n_epi % PRINT_INTERVAL == 0) && (n_epi != 0) {
            println!("# of episode :{}, avg score : {}", n_epi, score / PRINT_INTERVAL as f32);
            score = 0.0;
        }
    }
    
    env.close();
    
    Ok(())
}
