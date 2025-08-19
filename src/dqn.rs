use std::collections::VecDeque;

use anyhow::Result;
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::NdArray;
use burn::module::Module;
use burn::optim::{Adam, AdamConfig, adaptor::OptimizerAdaptor, decay::WeightDecayConfig};
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
use rand::seq::index::sample;
use rand::seq::IteratorRandom;
use rand::{Rng, distr::weighted::WeightedIndex, rng};

// Hyperparameters
const LEARNING_RATE: f64 = 0.0005;
const GAMMA: f32 = 0.98;
const BUFFER_LIMIT: usize = 50000;
const BATCH_SIZE: usize = 32;

#[derive(Debug, Clone)]
struct Data {
    pub state: [f32; 4], // Current state, correspond to `s` in the original Python code. Below are the same.
    pub action: u8,      // Action taken, correspond to `a`
    pub reward: f32,     // Reward received, correspond to `r`
    pub next_state: [f32; 4], // Next state, correspond to `s_prime`
    pub done: bool,      // Episode done flag, correspond to `done`
}

#[derive(Debug, Clone)]
pub struct DataTensor<B>
where
    B: Backend,
{
    pub state: Tensor<B, 2>,
    pub action: Tensor<B, 2, Int>,
    pub reward: Tensor<B, 2>,
    pub next_state: Tensor<B, 2>,
    pub done: Tensor<B, 2>,
}

pub struct ReplayBuffer {
    buffer: VecDeque<Data>
}

impl ReplayBuffer {
    pub fn new() -> Self {
        Self { buffer: VecDeque::new() }
    }
    
    pub fn put(&mut self, transition: Data) {
        self.buffer.push_back(transition); // add to the right end of the deque
    }
    
    pub fn sample<B: Backend>(&mut self, device: &B::Device) -> DataTensor<B> {
        let mut rng: rand::prelude::ThreadRng = rng();
        let mini_batches: Vec<&Data> = self.buffer.iter().choose_multiple(&mut rng, BATCH_SIZE);
        
        let mut states: [[f32; 4]; BATCH_SIZE] = [[0.0; 4]; BATCH_SIZE];
        let mut actions: [[u8; 1]; BATCH_SIZE] = [[0; 1]; BATCH_SIZE];
        let mut rewards: [[f32; 1]; BATCH_SIZE] = [[0.0; 1]; BATCH_SIZE];
        let mut next_states: [[f32; 4]; BATCH_SIZE] = [[0.0; 4]; BATCH_SIZE];
        let mut dones: [[u8; 1]; BATCH_SIZE] = [[0; 1]; BATCH_SIZE];
        
        for (index, transition) in mini_batches.iter().enumerate() {
            states[index] = transition.state;
            actions[index] = [transition.action];
            rewards[index] = [transition.reward];
            next_states[index] = transition.next_state;

            if transition.done {
                dones[index] = [0];
                continue;
            }

            dones[index] = [1];
        }
        
        let state = Tensor::from_data(states, device);
        let action = Tensor::from_data(actions, device);
        let reward = Tensor::from_data(rewards, device);
        let next_state = Tensor::from_data(next_states, device);
        let done = Tensor::from_data(dones, device);
        
        DataTensor {
            state, action, reward, next_state, done
        }
    }
    
    pub fn size(&self) -> usize {
        self.buffer.len()
    }
}

#[derive(Debug, Module)]
pub struct QNet<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
}

impl<B> QNet<B> 
where 
    B: Backend
{
    pub fn new(device: &B::Device) -> Self {
        Self {
            fc1: LinearConfig::new(4, 128).init(device), 
            fc2: LinearConfig::new(128, 128).init(device), 
            fc3: LinearConfig::new(128, 2).init(device),
        }
    }
    
    pub fn forward(&mut self, x: Tensor<B, 2>) -> Tensor<B, 2> {}
    
    pub fn sample_action(&mut self, observation: Tensor<B, 2>, epsilon: f32) -> {}
}
