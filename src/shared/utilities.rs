use burn::{module::AutodiffModule, optim::{adaptor::OptimizerAdaptor, decay::WeightDecayConfig, Adam, AdamConfig}, prelude::Backend, tensor::Tensor};
use rand::{Rng, distr::weighted::WeightedIndex, rng};

/// Shared function for sampling actions from a proability
pub fn sample_action(probability: &Vec<f32>) -> Result<usize, anyhow::Error> {
    let distributions: WeightedIndex<f32> = WeightedIndex::new(probability)?;

    let mut thread_rng: rand::prelude::ThreadRng = rng();
    let action: usize = thread_rng.sample(distributions);

    Ok(action)
}

/// Initialize an Adam optimizer with default config which resembles that of PyTorch
pub fn initialize_adam_optimizer() -> OptimizerAdaptor<Adam, AutodiffModule<B>, B> 
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
