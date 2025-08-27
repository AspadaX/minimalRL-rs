use burn::{prelude::Backend, tensor::Tensor};
use rand::{distr::weighted::WeightedIndex, rng, Rng};

/// Shared function for sampling actions from a proability
pub fn sample_action<B: Backend, const D: usize>(probability: Tensor<B, D>) -> Result<usize, anyhow::Error> {
    let probability_vector: Vec<f32> = probability
        .to_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .unwrap();
    let distributions: WeightedIndex<f32> = WeightedIndex::new(&probability_vector)?;
    
    let mut thread_rng: rand::prelude::ThreadRng = rng();
    let action: usize = thread_rng.sample(distributions);
    
    Ok(action)
}
