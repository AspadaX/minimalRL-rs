use burn::{module::AutodiffModule, optim::adaptor::OptimizerAdaptor};
use rand::{Rng, distr::weighted::WeightedIndex, rng};

/// Shared function for sampling actions from a proability
pub fn sample_action(probability: &Vec<f32>) -> Result<usize, anyhow::Error> {
    let distributions: WeightedIndex<f32> = WeightedIndex::new(probability)?;

    let mut thread_rng: rand::prelude::ThreadRng = rng();
    let action: usize = thread_rng.sample(distributions);

    Ok(action)
}

/// Initialize an Adam optimizer with default config which resembles that of PyTorch
pub fn initialize_adam_optimizer() -> OptimizerAdaptor<Adam, AutodiffModule<B>, B> {
    AdamConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_epsilon(1e-08)
        .with_weight_decay(Some(WeightDecayConfig::new(0.0)))
        .init()
}
