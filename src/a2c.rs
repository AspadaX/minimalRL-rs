use burn::{module::Module, nn::{Linear, Relu}, prelude::Backend, tensor::{activation::softmax, Tensor}};


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