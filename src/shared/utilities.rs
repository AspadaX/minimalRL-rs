use burn::{module::AutodiffModule, nn::loss::{HuberLoss, HuberLossConfig}, optim::{adaptor::OptimizerAdaptor, decay::WeightDecayConfig, Adam, AdamConfig}, prelude::Backend, tensor::{backend::AutodiffBackend, Shape, Tensor, TensorData}};
use gym_rs::envs::classical_control::cartpole::CartPoleObservation;
use rand::{Rng, distr::weighted::WeightedIndex, rng};

/// Shared function for sampling actions from a proability
pub fn sample_action(probability: &Vec<f32>) -> Result<usize, anyhow::Error> {
    let distributions: WeightedIndex<f32> = WeightedIndex::new(probability)?;

    let mut thread_rng: rand::prelude::ThreadRng = rng();
    let action: usize = thread_rng.sample(distributions);

    Ok(action)
}

/// Initialize an Adam optimizer with default config which resembles that of PyTorch
pub fn initialize_adam_optimizer<B: AutodiffBackend, M: AutodiffModule<B>>() -> OptimizerAdaptor<Adam, M, B> 
{
    AdamConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_epsilon(1e-08)
        .with_weight_decay(Some(WeightDecayConfig::new(0.0)))
        .init()
}

pub fn compute_logprob<const D: usize, B: Backend>(a: Tensor<B, D>, mu: Tensor<B, D>, std: Tensor<B, D>) -> Tensor<B, D> 
{
    let std2 = std.powf_scalar(2.0);

    let l = (a - mu).powf_scalar(2.0).div(std2.clone() + 1e-6);
    let r = (std2.mul_scalar(2.0 * std::f32::consts::PI)).log();
    (l + r).mul_scalar(-0.5)
}

/// This is also known as `smooth L1 loss` in PyTorch.
/// The 1.0 delta value originates from PyTorch default.
pub fn create_huber_loss() -> HuberLoss {
    HuberLossConfig::new(1.0).init()
}

/// Convert the original observation to a Tensor. 
/// This is usually required by a reset. 
pub fn convert_carte_pole_observation_to_tensor<B: Backend, const D: usize>(observation: CartPoleObservation, device: &B::Device) -> Tensor<B, D> {
    // Reflect the shape of the state, which is 1-dimensional array with 4 elements
    let state_data: TensorData =
        TensorData::new(Vec::from(observation), Shape::new([4]));

    Tensor::from_data(state_data, device)
}