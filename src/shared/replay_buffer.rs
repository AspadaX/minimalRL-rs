use std::collections::VecDeque;

use burn::{prelude::Backend, tensor::{Int, Shape, Tensor}};
use rand::{rng, rngs::ThreadRng, seq::IteratorRandom};

use crate::shared::data_structs::{Data, DataBatch};

pub struct ReplayBuffer {
    // Why do we use VecDeque, but not array nor Vec?
    // VecDeque is a double-ended queue, which is more efficient than Vec when we want to pop the first element.
    buffer: VecDeque<Data>,
    rng: ThreadRng
}

impl ReplayBuffer {
    pub fn new<const C: usize>() -> Self {
        Self {
            buffer: VecDeque::with_capacity(C),
            rng: rng()
        }
    }
    
    pub fn put(&mut self, transition: Data) {
        if self.buffer.len() == self.buffer.capacity() {
            self.buffer.pop_front();
        }

        self.buffer.push_back(transition);
    }
    
    pub fn sample<B: Backend, const S: usize>(&mut self, device: &B::Device) -> DataBatch<B> {
        let mini_batches: Vec<&Data> = self.buffer.iter().choose_multiple(&mut self.rng, S);
        
        let mut states: [[f32; 4]; S] = [[0.0; 4]; S];
        let mut actions: [[u8; 1]; S] = [[0; 1]; S];
        let mut rewards: [[f32; 1]; S] = [[0.0; 1]; S];
        let mut next_states: [[f32; 4]; S] = [[0.0; 4]; S];
        let mut dones: [[u8; 1]; S] = [[0; 1]; S];
        
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
        
        let states_data: Tensor<B, 2> = Tensor::from_data(states, device);
        let actions_data: Tensor<B, 2, Int> = Tensor::from_data(actions, device);
        let rewards_data: Tensor<B, 2> = Tensor::from_data(rewards, device);
        let next_states_data: Tensor<B, 2> = Tensor::from_data(next_states, device);
        let dones_data: Tensor<B, 2> = Tensor::from_data(dones, device);
        
        DataBatch {
            states: states_data,
            actions: actions_data,
            rewards: rewards_data,
            next_states: next_states_data,
            dones: dones_data,
            action_probabilities: Tensor::zeros(Shape::new([S, 2]), device) // This is not used in the DQN. Just a filler.
        }
    }
    
    pub fn size(&self) -> usize {
        self.buffer.len()
    }
}