use anyhow::Result;
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::NdArray;
use burn::module::Module;
use burn::optim::SimpleOptimizer;
use burn::optim::{Adam, AdamConfig, adaptor::OptimizerAdaptor};
use burn::tensor::Shape;
use burn::{
    backend::Autodiff,
    nn::{Linear, LinearConfig, Relu, loss::HuberLossConfig},
    optim::{GradientsParams, Optimizer},
    prelude::Backend,
    tensor::{
        Int, Tensor, TensorData, backend::AutodiffBackend, cast::ToElement,
    },
};
use gym_rs::{
    core::{ActionReward, Env},
    envs::classical_control::cartpole::{CartPoleEnv, CartPoleObservation},
    utils::renderer::RenderMode,
};

use crate::shared::data_structs::{Data, DataBatch};
use crate::shared::replay_buffer::ReplayBuffer;

// Hyperparameters
const LEARNING_RATE: f64 = 0.0005;
const GAMMA: f32 = 0.98;
const BATCH_SIZE: usize = 32;
const MAX_CAPACITY: usize = 50000;

#[derive(Debug, Module)]
pub struct QNet<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
    relu: Relu,
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
            relu: Relu::new(),
        }
    }
    
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let x: Tensor<B, D> = self.relu.forward(self.fc1.forward(x));
        let x: Tensor<B, D> = self.relu.forward(self.fc2.forward(x));
        
        self.fc3.forward(x)
    }
    
    /// Either use the model output as the action, 
    /// or to use a random digit between 0 and 1
    pub fn sample_action(&mut self, observation: Tensor<B, 1>, epsilon: f32) -> usize {
        let output: Tensor<B, 1> = self.forward(observation).detach();
        let coin: f32 = rand::random();
        if coin < epsilon {
            return rand::random_range(0..=1);
        }
        
        // the 0-dim is the correct input, 
        // which will result in the same argmax tensor as the Python one
        let argmax_tensor: Tensor<B, 1, Int> = output.argmax(0);

        let scalar = argmax_tensor.into_scalar().to_usize();
        
        scalar
    }
}

pub fn train<O: SimpleOptimizer<B::InnerBackend>, B: AutodiffBackend>(
    q_net: QNet<B>, 
    target_network: &QNet<B>, 
    replay_buffer: &mut ReplayBuffer, 
    optimizer: &mut OptimizerAdaptor<O, QNet<B>, B>,
    device: &B::Device
) -> QNet<B> {
    let mut q_net_optimized: QNet<B> = q_net;
    for _ in 0..10 {
        let data_batch: DataBatch<B> = replay_buffer.sample::<B, BATCH_SIZE>(device);
        
        let q_net_output: Tensor<B, 2> = q_net_optimized.forward(data_batch.states);
        let q_net_output_to_action: Tensor<B, 2> = q_net_output.gather(1, data_batch.actions);
        // This might differ from the original Python approach, but I am not sure.
        // I experimented with it, and I found the below code has the same shape as the one in Python. 
        // In Python, the below variable is named as `max_q_prime`.
        let target_network_output: Tensor<B, 2> = target_network.forward(data_batch.next_states)
            .max_dim(1);
            // .select(0, Tensor::from_data([0], device))
            // .unsqueeze_dim(1); // DIVERGENCE: differ from the original python code 
        
        let target: Tensor<B, 2> = data_batch.rewards + GAMMA * target_network_output * data_batch.dones;
        
        // This is also known as `smooth L1 loss` in PyTorch. 
        // The 1.0 delta value originates from PyTorch default.
        let huber_loss: burn::nn::loss::HuberLoss = HuberLossConfig::new(1.0).init();
        let loss: Tensor<B, 2> = huber_loss.forward_no_reduction(q_net_output_to_action, target);

        // Calculate gradients and then convert them to parameters
        let gradients: <B as AutodiffBackend>::Gradients = loss.backward();
        let gradients_params: GradientsParams = GradientsParams::from_grads(gradients, &q_net_optimized);
        
        q_net_optimized = optimizer.step(LEARNING_RATE, q_net_optimized, gradients_params);
    }
    
    q_net_optimized
}

/// The main training loop
pub fn run_session() -> Result<()> {
    let mut env: CartPoleEnv = CartPoleEnv::new(RenderMode::None);

    let device: NdArrayDevice = NdArrayDevice::default();
    let mut q_net: QNet<Autodiff<NdArray>> = QNet::new(&device);
    let mut q_target_net: QNet<Autodiff<NdArray>> = QNet::new(&device);
    // We specify the max capacity during the compile time
    let mut memory: ReplayBuffer = ReplayBuffer::new::<MAX_CAPACITY>();
    let mut optimizer: OptimizerAdaptor<Adam, QNet<Autodiff<NdArray>>, Autodiff<NdArray>> = AdamConfig::new().init();

    let mut score: f32 = 0.0;
    let print_interval: usize = 20;

    for n_epi in 0..10000 {
        let epsilon: f32 = f32::max(0.01, 0.08 - 0.01 * ( n_epi as f32 / 200.0));
        let (raw_state, _) = env.reset(None, false, None);

        let mut done: bool = false;
        let mut previous_state: CartPoleObservation = raw_state;
        
        while !done {
            // Reflect the shape of the state, which is 1-dimensional array with 4 elements
            let state_data: TensorData = TensorData::new(Vec::from(previous_state), Shape::new([4]));
            let state: Tensor<Autodiff<NdArray>, 1> = Tensor::from_data(state_data, &device).detach(); 
            
            let action: usize = q_net.sample_action(state, epsilon);
            let result: ActionReward<CartPoleObservation, ()> = env.step(action);
            
            done = result.done;
            memory.put(
                Data::from_step_result(
                    previous_state, 
                    result.observation, 
                    action as u8, 
                    0.0, // We don't use action probability here
                    done,
                    result.reward.to_f32() / 100.0,
                )
            );

            previous_state = result.observation;
            
            score += result.reward.to_f32();
            
            if done {
                break;
            }
        }
        
        if memory.size() > 2000 {
            q_net = train(q_net, &q_target_net, &mut memory, &mut optimizer, &device);
        }
        
        if (n_epi % print_interval == 0) && (n_epi != 0) {
            q_target_net = q_net.clone();    
            println!("# of episode :{}, avg score : {}", n_epi, score / print_interval as f32);
            score = 0.0;
        }
    }

    env.close();

    Ok(())
}
