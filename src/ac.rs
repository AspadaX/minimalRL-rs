use anyhow::Result;
use burn::{backend::{ndarray::NdArrayDevice, Autodiff, NdArray}, module::Module, nn::{loss::{HuberLossConfig, Reduction}, Linear, LinearConfig, Relu}, optim::{adaptor::OptimizerAdaptor, decay::WeightDecayConfig, Adam, AdamConfig, GradientsParams, Optimizer}, prelude::Backend, serde::de::value, tensor::{activation::softmax, cast::ToElement, Int, Shape, Tensor, TensorData}};
use gym_rs::{
    core::{ActionReward, Env},
    envs::classical_control::cartpole::{CartPoleEnv, CartPoleObservation},
    utils::renderer::RenderMode,
};
use rand::{rng, Rng};

use crate::utilities::sample_action;


// Hyperparameters
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
    relu: Relu
}

impl<B> ActorCritic<B> 
where 
    B: Backend
{
    pub fn new(device: &B::Device) -> Self {
        Self {
            fully_connected_layer: LinearConfig::new(4, 256).init(device), 
            policy_fully_connected_layer: LinearConfig::new(256, 2).init(device), 
            value_function_fully_connected_layer: LinearConfig::new(256, 1).init(device),
            relu: Relu::new()
        }
    }
    
    /// softmax dimension is default to 1
    pub fn use_policy_function<const D: usize>(&self, x: Tensor<B, D>, softmax_dimension: Option<usize>) -> Tensor<B, D> {
        let x: Tensor<B, D> = self.relu.forward(self.fully_connected_layer.forward(x));
        let x: Tensor<B, D> = self.policy_fully_connected_layer.forward(x);
        
        softmax(x, softmax_dimension.unwrap_or(1))
    }
    
    pub fn use_value_function<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let x: Tensor<B, D> = self.relu.forward(self.fully_connected_layer.forward(x));
        
        self.value_function_fully_connected_layer.forward(x)
    }
}

pub fn compute_temporarl_difference_target<B: Backend, const D: usize>(
    value_function_result: Tensor<B, D>, 
    rewards: [[f32; 1]; UPDATE_INTERVAL], 
    dones: [[usize; 1]; UPDATE_INTERVAL],
    device: &B::Device
) -> Tensor<B, 1> {
    let mut temporal_difference_target: Vec<f32> = Vec::new();

    let value_function_result_scalar: f32 = value_function_result.to_data().convert::<f32>().to_vec::<f32>().unwrap()[0];
    let reversed_rewards = rewards.iter().rev();
    let reversed_dones = dones.iter().rev();

    for (index, (reward, _)) in reversed_rewards.zip(reversed_dones).enumerate() {
        let done: f32 = 1.0 - dones[index][0] as f32;
        let discounted_cumulative_reward: f32 = reward[0] + GAMMA * value_function_result_scalar * done;
        temporal_difference_target.push(discounted_cumulative_reward);
    }

    temporal_difference_target.reverse();
    Tensor::from_data(
        TensorData::new(temporal_difference_target, [UPDATE_INTERVAL]), 
        device
    )
}

pub fn test<B: Backend>(step_idx: usize, model: &ActorCritic<B>) -> Result<()> {
    let mut env: CartPoleEnv = CartPoleEnv::new(RenderMode::None);
    let mut score: f32 = 0.0;
    let mut done: bool = false;
    let number_test: usize = 10;

    for _ in 0..number_test {
        let (mut state, _) = env.reset(Some(rng().random()), false, None);
        while !done {
            let array_state: [f32; 4] = convert_to_array(state);
            let probability: Tensor<B, 1> = model.use_policy_function::<1>(Tensor::from(array_state), Some(0));
            let action: usize = sample_action(
                &probability
                    .to_data()
                    .convert::<f32>()
                    .to_vec::<f32>()
                    .unwrap()
            )?;
            let result: ActionReward<CartPoleObservation, ()> = env.step(action);

            // Record the state we get from this turn
            state = result.observation;
            score += result.reward.to_f32();
            done = result.done;
        }

        done = false;
    }

    println!("Step # :{}, avg score : {}", step_idx, score / number_test as f32);

    Ok(())
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
    let mut optimizer: OptimizerAdaptor<Adam, ActorCritic<Autodiff<NdArray>>, Autodiff<NdArray>> = AdamConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_epsilon(1e-08)
        .with_weight_decay(Some(WeightDecayConfig::new(0.0)))
        .init();
    
    let (mut state, _) = env.reset(Some(rng().random()), false, None);
            
    for n_epi in 0..MAX_TRAIN_STEPS {
        let array_state: [f32; 4] = convert_to_array(state);

        let mut states: [[f32; 4]; UPDATE_INTERVAL] = [[0.0; 4]; UPDATE_INTERVAL];
        let mut actions: [[u32; 1]; UPDATE_INTERVAL] = [[0; 1]; UPDATE_INTERVAL];
        let mut rewards: [[f32; 1]; UPDATE_INTERVAL] = [[0.0; 1]; UPDATE_INTERVAL];
        let mut dones: [[usize; 1]; UPDATE_INTERVAL] = [[0; 1]; UPDATE_INTERVAL];
        
        for index in 0..UPDATE_INTERVAL {
            let probability: Tensor<Autodiff<NdArray>, 1> = model.use_policy_function::<1>(Tensor::from(array_state), Some(0));
            let action: usize = sample_action(
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
        
        let value_function_result: Tensor<Autodiff<NdArray>, 1> = model.use_value_function(Tensor::from(array_state));
        let temporal_difference_target: Tensor<Autodiff<NdArray>, 1> = compute_temporarl_difference_target(value_function_result, rewards, dones, &device);
        
        let states_tensor: Tensor<Autodiff<NdArray>, 2> = Tensor::from(states);
        let actions_tensor: Tensor<Autodiff<NdArray>, 2, Int> = Tensor::from(actions);
        
        let value_function_results: Tensor<Autodiff<NdArray>, 1> = model.use_value_function(states_tensor.clone()).reshape([-1]);
        let advantage: Tensor<Autodiff<NdArray>, 1> = temporal_difference_target.clone() - value_function_results.clone();
        
        let policy_function_result: Tensor<Autodiff<NdArray>, 2> = model.use_policy_function(states_tensor, None);
        let policy_action: Tensor<Autodiff<NdArray>, 1> = policy_function_result.gather(1, actions_tensor).reshape([-1]);
        
        // This is also known as `smooth L1 loss` in PyTorch. 
        // The 1.0 delta value originates from PyTorch default.
        let huber_loss: burn::nn::loss::HuberLoss = HuberLossConfig::new(1.0).init();
        let loss: Tensor<Autodiff<NdArray>, 1> = -(policy_action.log() * advantage.clone()).mean() + huber_loss
            .forward_no_reduction(value_function_results.clone(), temporal_difference_target);
        
        let gradients = loss.backward();
        model = optimizer.step(
            LEARNING_RATE as f64, 
            model.clone(), 
            GradientsParams::from_grads(gradients, &model)
        );
        
        if (n_epi % PRINT_INTERVAL == 0) && (n_epi != 0) {
            test(n_epi, &model)?;
        }
    }
    
    env.close();
    
    Ok(())
}
