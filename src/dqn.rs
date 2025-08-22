use std::collections::VecDeque;

use anyhow::Result;
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::NdArray;
use burn::module::{AutodiffModule, Module};
use burn::optim::SimpleOptimizer;
use burn::optim::{Adam, AdamConfig, adaptor::OptimizerAdaptor, decay::WeightDecayConfig};
use burn::record::Record;
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
pub struct DataBatch<B>
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
    
    pub fn sample<B: Backend>(&mut self, device: &B::Device) -> DataBatch<B> {
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
        
        DataBatch {
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
    
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let relu = Relu::new();
        let x = relu.forward(self.fc1.forward(x));
        let x = relu.forward(self.fc2.forward(x));
        
        self.fc3.forward(x)
    }
    
    /// Either use the model output as the action, 
    /// or to use a random digit between 0 and 1
    pub fn sample_action(&mut self, observation: Tensor<B, 2>, epsilon: f32) -> usize {
        let output = self.forward(observation);
        let coin: f32 = rand::random();
        if coin < epsilon {
            return rand::random_range(0..=1);
        }
        
        let argmax_tensor = output.argmax(1);
        
        argmax_tensor.into_scalar().to_usize()
    }
}

pub fn train<O: SimpleOptimizer<B::InnerBackend>, B: AutodiffBackend>(
    q_net: QNet<B>, 
    target_network: QNet<B>, 
    replay_buffer: &mut ReplayBuffer, 
    optimizer: &mut OptimizerAdaptor<O, QNet<B>, B>,
    device: &B::Device
) -> QNet<B> {
    let mut q_net_optimized = q_net;
    for _ in 0..10 {
        let data_batch = replay_buffer.sample::<B>(device);
        
        let q_net_output = q_net_optimized.forward(data_batch.state);
        let q_net_output_to_action = q_net_output.gather(1, data_batch.action);
        let target_network_output: Tensor<B, 2> = target_network.forward(data_batch.next_state)
            .max_dim(1)
            .select(0, Tensor::from_data([0], device))
            .unsqueeze_dim(1); // DIVERGENCE: differ from the original python code 
        
        let target = data_batch.reward + GAMMA * target_network_output * data_batch.done;
        
        // This is also known as `smooth L1 loss` in PyTorch. 
        // The 1.0 delta value originates from PyTorch default.
        let huber_loss: burn::nn::loss::HuberLoss = HuberLossConfig::new(1.0).init();
        let loss = huber_loss.forward_no_reduction(q_net_output_to_action, target);

        // Calculate gradients and then convert them to parameters
        let gradients = loss.backward();
        let gradients_params = GradientsParams::from_grads(gradients, &q_net_optimized);
        
        q_net_optimized = optimizer.step(LEARNING_RATE, q_net_optimized, gradients_params);
    }
    
    q_net_optimized
}

/// The main training loop
pub fn run_session() -> Result<()> {
    let mut env = CartPoleEnv::new(RenderMode::Human);

    let device = NdArrayDevice::default();
    let mut q_net = QNet::new(&device);
    let mut q_target_net = QNet::new(&device);
    let mut memory = ReplayBuffer::new();
    let mut optimizer = AdamConfig::new().init();

    let mut score: f32 = 0.0;
    let print_interval: usize = 20;

    for n_epi in 0..10000 {
        let epsilon = f32::max(0.01, 0.08 - 0.01 * ( n_epi as f32 / 200.0));
        let (mut raw_state, _) = env.reset(None, false, None);
        let mut done: bool = false;
        
        while !done {
            let action = q_net.sample_action(raw_state, epsilon);
            let result: ActionReward<CartPoleObservation, ()> = env.step(action);
            
            done = result.done;
            memory.put(
                Data { state: result.state, action: result.action, reward: result.award / 100.0, next_state: result.observation, done: result.done }
            );
            raw_state = result.observation;
            
            score += result.reward;
            
            if done {
                break;
            }
        }
        
        if memory.size() > 2000 {
            q_net = train(q_net, q_target_net, &mut memory, &mut optimizer, &device);
        }
        
        if (n_epi % print_interval == 0) && (n_epi != 0) {
            println!("# of episode :{}, avg score : {}", n_epi, score / print_interval as f32);
            score = 0.0;
        }
    }

    env.close();

    Ok(())
}