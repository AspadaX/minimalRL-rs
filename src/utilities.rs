use rand::{distr::weighted::WeightedIndex, rng, Rng};

/// Shared function for sampling actions from a proability
pub fn sample_action(probability: Vec<f32>) -> Result<usize, anyhow::Error> {
    let distributions: WeightedIndex<f32> = WeightedIndex::new(&probability)?;
    
    let mut thread_rng: rand::prelude::ThreadRng = rng();
    let action: usize = thread_rng.sample(distributions);
    
    Ok(action)
}
