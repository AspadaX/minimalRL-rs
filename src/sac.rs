use burn::{module::Module, nn::{loss::HuberLossConfig, Linear, LinearConfig, Relu}, optim::{adaptor::OptimizerAdaptor, AdamConfig, Optimizer}, prelude::Backend, tensor::{activation::{log_softmax, relu, softplus, tanh}, backend::AutodiffBackend, cast::ToElement, linalg::vector_normalize, Distribution, Tensor}};
use gym_rs::envs::classical_control::cartpole::CartPoleObservation;
use rand::rng;

use crate::{shared::{data_structs::{Data, DataBatch}, utilities::{compute_logprob, create_huber_loss, initialize_adam_optimizer}}};

const POLICY_LEARNING_RATE: f32 = 0.0005;
const Q_LEARNING_RATE: f32 = 0.001;
const INIT_ALPHA: f32 = 0.01;
const GAMMA: f32 = 0.98;
const BATCH_SIZE: usize = 32;
const BUFFER_LIMIT: usize = 50000;
const TAU: f32 = 0.01; // For target network soft update
const TARGET_ENTROPY: f32 = -1.0; // For automated alpha update
const ALPHA_LEARNING_RATE: f32 = 0.001; // Same as above

#[derive(Debug, Module)]
pub struct QNet<B> 
where 
    B: AutodiffBackend
{
    fc_state: Linear<B>,
    fc_action: Linear<B>,
    fc_cat: Linear<B>,
    fc_out: Linear<B>,
    relu: Relu,
    optimizer: OptimizerAdaptor<Adam, QNet<B>, B>,
}

impl<B> QNet<B>
where
    B: AutodiffBackend,
{
    pub fn new(device: &B::Device) -> Self {
        Self {
            fc_state: LinearConfig::new(3, 64).init(device),
            fc_action: LinearConfig::new(1, 64).init(device),
            fc_cat: LinearConfig::new(128, 32).init(device),
            fc_out: LinearConfig::new(32, 1).init(device),
            relu: Relu::new(),
            optimizer: initialize_adam_optimizer(),
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>, a: Tensor<B, D>) -> Tensor<B, D> {
        let hidden_layer_one = relu(self.fc_state.forward(x));
        let hidden_layer_two = relu(self.fc_action.forward(a));
        let cat = Tensor::cat(vec![hidden_layer_one, hidden_layer_two], 1);
        let q = relu(self.fc_cat.forward(cat));
        let q = self.fc_out.forward(q);

        q
    }

    pub fn train_net<const D: usize>(&mut self, target: Tensor<B, D>, transition: Data) {
        let huber_loss = create_huber_loss();
        let loss = huber_loss.forward_no_reduction(self.forward(transition.state.into(), [transition.action as usize].into()), target);
        
        let optimizer = initialize_adam_optimizer();
        optimizer.step(Q_LEARNING_RATE, self, loss.mean());
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

#[derive(Debug, Module)]
pub struct PolicyNet<B: Backend> 
{
    fully_connected_layer_one: Linear<B>,
    fully_connected_layer_mean_output: Linear<B>,
    fully_connected_layer_standard_deviation: Linear<B>,
    relu: Relu,
    log_alpha: Tensor<B, 1>,
}

impl<B> PolicyNet<B> 
where
    B: Backend
{
    pub fn new(device: &B::Device) -> Self {
        let log_alpha = Tensor::from_floats([INIT_ALPHA], device);
        let log_alpha = log_alpha.log();

        Self { 
            fully_connected_layer_one: LinearConfig::new(3, 128).init(device), 
            fully_connected_layer_mean_output: LinearConfig::new(128, 1).init(device), 
            fully_connected_layer_standard_deviation: LinearConfig::new(128, 1).init(device), 
            relu: Relu::new(),
            log_alpha: log_alpha.require_grad(),
        }
    }
    
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> (Tensor<B, D>, Tensor<B, D>) {
        let x = self.relu.forward(self.fully_connected_layer_one.forward(x));
        let mean = self.fully_connected_layer_mean_output.forward(x.clone());
        let standard_deviation = softplus(
            self.fully_connected_layer_standard_deviation.forward(x),
            1.0 // originates from /torch/nn/modules/activation.py
        );

        let action = mean.clone() + standard_deviation.clone() * mean.random_like(Distribution::Normal(0.0, 1.0));
        let action_log_probability = compute_logprob(action.clone(), mean, standard_deviation);

        let real_action = tanh(action.clone());

        let log_probability = action_log_probability - (real_action.clone().ones_like() - real_action.clone().powf_scalar(2.0) + real_action.clone().full_like(1e-7)).log();

        (real_action, log_probability)
    }

    // def train_net(self, q1, q2, mini_batch):
    //     s, _, _, _, _ = mini_batch
    //     a, log_prob = self.forward(s)
    //     entropy = -self.log_alpha.exp() * log_prob

    //     q1_val, q2_val = q1(s,a), q2(s,a)
    //     q1_q2 = torch.cat([q1_val, q2_val], dim=1)
    //     min_q = torch.min(q1_q2, 1, keepdim=True)[0]

    //     loss = -min_q - entropy # for gradient ascent
    //     self.optimizer.zero_grad()
    //     loss.mean().backward()
    //     self.optimizer.step()

    //     self.log_alpha_optimizer.zero_grad()
    //     alpha_loss = -(self.log_alpha.exp() * (log_prob + target_entropy).detach()).mean()
    //     alpha_loss.backward()
    //     self.log_alpha_optimizer.step()
    pub fn train_net(&self, q_net_one: QNet<B>, q_net_two: QNet<B>, transition: DataBatch<B>) {
        let (action, log_probability) = self.forward(transition.states);
        // In Rust, the 1-dimensional tensor cannot multiply with a 2-dimensional tensor, which is log_probability. 
        // Hence, we convert it to a f32 digit before performing a multiplication. 
        let entropy = -self.log_alpha.clone().exp().into_scalar().to_f32() * log_probability;

        let (q1_value, q2_value) = // need to define a new Q-Net that tailors to SAC
    }
}

// class PolicyNet(nn.Module):
    // def __init__(self, learning_rate):
    //     super(PolicyNet, self).__init__()
    //     self.fc1 = nn.Linear(3, 128)
    //     self.fc_mu = nn.Linear(128,1)
    //     self.fc_std  = nn.Linear(128,1)
    // 
    //     this is how optimizer is initialized
    //     self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    //     self.log_alpha = torch.tensor(np.log(init_alpha))
    //     self.log_alpha.requires_grad = True
    //     self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

    // def forward(self, x):
    //     x = F.relu(self.fc1(x))
    //     mu = self.fc_mu(x)
    //     std = F.softplus(self.fc_std(x))
    //     dist = Normal(mu, std)
    //     action = dist.rsample()
    //     log_prob = dist.log_prob(action)
    //     real_action = torch.tanh(action)
    //     real_log_prob = log_prob - torch.log(1-torch.tanh(action).pow(2) + 1e-7)
    //     return real_action, real_log_prob

    // def train_net(self, q1, q2, mini_batch):
    //     s, _, _, _, _ = mini_batch
    //     a, log_prob = self.forward(s)
    //     entropy = -self.log_alpha.exp() * log_prob

    //     q1_val, q2_val = q1(s,a), q2(s,a)
    //     q1_q2 = torch.cat([q1_val, q2_val], dim=1)
    //     min_q = torch.min(q1_q2, 1, keepdim=True)[0]

    //     loss = -min_q - entropy # for gradient ascent
    //     self.optimizer.zero_grad()
    //     loss.mean().backward()
    //     self.optimizer.step()

    //     self.log_alpha_optimizer.zero_grad()
    //     alpha_loss = -(self.log_alpha.exp() * (log_prob + target_entropy).detach()).mean()
    //     alpha_loss.backward()
    //     self.log_alpha_optimizer.step()
