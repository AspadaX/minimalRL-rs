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
use rand::{Rng, distr::weighted::WeightedIndex, rng};

// Hyperparameters
const LEARNING_RATE: f64 = 0.0005;
const GAMMA: f32 = 0.98;
const BUFFER_LIMIT: usize = 50000;
const BATCH_SIZE: usize = 32;

pub struct ReplayBuffer {
    buffer: VecDeque<_>
}

impl ReplayBuffer {
    pub fn new() -> Self {}
    
    pub fn put(&mut self, transition: _) {}
    
    pub fn sample(&mut self) {}
    
    pub fn size(&self) -> usize {
        self.buffer.len()
    }
}
