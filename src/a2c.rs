use anyhow::Result;
use burn::{backend::ndarray::NdArrayDevice, module::Module, nn::{Linear, LinearConfig, Relu}, optim::{decay::WeightDecayConfig, AdamConfig}, prelude::Backend, tensor::{activation::softmax, Tensor}};
use gym_rs::{
    core::{ActionReward, Env},
    envs::classical_control::cartpole::{CartPoleEnv, CartPoleObservation},
    utils::renderer::RenderMode,
};

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
    
    pub fn update_policy(&mut self, x: Tensor<B, 1>, softmax_dimension: Option<usize>) -> Tensor<B, 1> {
        let relu = Relu::new();
        let x = relu.forward(self.fully_connected_layer.forward(x));
        let x = self.policy_fully_connected_layer.forward(x);
        
        softmax(x, softmax_dimension.unwrap_or(1))
    }
    
    pub fn update_value_function(&mut self, x: Tensor<B, 1>) -> Tensor<B, 1> {
        let relu = Relu::new();
        let x = relu.forward(self.fully_connected_layer.forward(x));
        
        self.value_function_fully_connected_layer.forward(x)
    }
}

pub fn run_session() -> Result<()> {
    let mut env: CartPoleEnv = CartPoleEnv::new(RenderMode::None);

    let device: NdArrayDevice = NdArrayDevice::default();
    // Initialize the neural networks
    let model = ActorCritic::new(&device);
    let optimizer = AdamConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_epsilon(1e-08)
        .with_weight_decay(Some(WeightDecayConfig::new(0.0)))
        .init();
    
    let score: f32 = 0.0;
    
    let state = env.reset();
    for n_epi in 0..MAX_TRAIN_STEPS {
        // s_lst, a_lst, r_lst, mask_lst = list(), list(), list(), list()
        // for _ in range(update_interval):
        //     prob = model.pi(torch.from_numpy(s).float())
        //     a = Categorical(prob).sample().numpy()
        //     s_prime, r, done, info = envs.step(a)
    
        //     s_lst.append(s)
        //     a_lst.append(a)
        //     r_lst.append(r/100.0)
        //     mask_lst.append(1 - done)
    
        //     s = s_prime
        //     step_idx += 1
        let mut states: [[f32; 4]; UPDATE_INTERVAL] = [[0.0; 4]; UPDATE_INTERVAL];
        let mut actions: [[u8; 1]; UPDATE_INTERVAL] = [[0; 1]; UPDATE_INTERVAL];
        let mut rewards: [[f32; 1]; UPDATE_INTERVAL] = [[0.0; 1]; UPDATE_INTERVAL];
        let mut next_states: [[f32; 4]; UPDATE_INTERVAL] = [[0.0; 4]; UPDATE_INTERVAL];
        let mut dones: [[u8; 1]; UPDATE_INTERVAL] = [[0; 1]; UPDATE_INTERVAL];
        
        for _ in 0..UPDATE_INTERVAL {
            let probability = model.update_policy(state, None);
            let action = sample_action(probability)?;
        }
        
        if (n_epi % PRINT_INTERVAL == 0) && (n_epi != 0) {
            println!("# of episode :{}, avg score : {}", n_epi, score / print_interval as f32);
            score = 0.0;
        }
    }
    
    env.close();
    
    Ok(())
}