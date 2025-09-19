use burn::{module::Module, nn::{Linear, LinearConfig, Relu}, optim::{adaptor::OptimizerAdaptor, AdamConfig, Optimizer}, prelude::Backend, tensor::{activation::{log_softmax, softplus, tanh}, cast::ToElement, linalg::vector_normalize, Distribution, Tensor}};
use gym_rs::envs::classical_control::cartpole::CartPoleObservation;
use rand::rng;

use crate::shared::{data_structs::DataBatch, utilities::compute_logprob};

#[derive(Debug, Module)]
pub struct PolicyNet<B: Backend> 
{
    fully_connected_layer_one: Linear<B>,
    fully_connected_layer_mean_output: Linear<B>,
    fully_connected_layer_standard_deviation: Linear<B>,
    relu: Relu
}

impl<B> PolicyNet<B> 
where
    B: Backend
{
    pub fn new(device: &B::Device) -> Self {
        Self { 
            fully_connected_layer_one: LinearConfig::new(3, 128).init(device), 
            fully_connected_layer_mean_output: LinearConfig::new(128, 1).init(device), 
            fully_connected_layer_standard_deviation: LinearConfig::new(128, 1).init(device), 
            relu: Relu::new()
        }
    }
    
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
    pub fn forward<const D: usize>(&mut self, x: Tensor<B, D>) -> (Tensor<B, D>, Tensor<B, D>) {
        let x = self.relu.forward(self.fully_connected_layer_one.forward(x));
        let mean = self.fully_connected_layer_mean_output.forward(x);
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
