use anyhow::Result;
use burn::{backend::{ndarray::NdArrayDevice, Autodiff, NdArray}, module::Module, nn::{loss::HuberLossConfig, Linear, LinearConfig, Relu}, optim::{adaptor::OptimizerAdaptor, AdamConfig, Optimizer}, prelude::Backend, tensor::{activation::{log_softmax, relu, softplus, tanh}, backend::AutodiffBackend, cast::ToElement, linalg::vector_normalize, Distribution, Tensor}};
use gym_rs::{core::Env, envs::classical_control::cartpole::{CartPoleEnv, CartPoleObservation}, utils::renderer::RenderMode};
use rand::rng;

use crate::shared::{data_structs::{Data, DataBatch}, replay_buffer::ReplayBuffer, utilities::{compute_logprob, convert_carte_pole_observation_to_tensor, create_huber_loss, initialize_adam_optimizer}};

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
pub struct QNet<B: Backend> 
{
    fc_state: Linear<B>,
    fc_action: Linear<B>,
    fc_cat: Linear<B>,
    fc_out: Linear<B>,
    relu: Relu,
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

    // Train the net. It returns the loss for optimizers. 
    pub fn train_net<const D: usize>(&mut self, target: Tensor<B, D>, transition: Data) -> Tensor<B, D> {
        let huber_loss = create_huber_loss();
        huber_loss.forward_no_reduction(self.forward(transition.state.into(), [transition.action as usize].into()), target)
    }

    pub fn soft_update<const D: usize>(&mut self, net_target: QNet<B>) -> Tensor<B, D> {
        net_target.para
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

pub fn run_session() -> Result<()> {
    let mut env = CartPoleEnv::new(RenderMode::None);

    let device = NdArrayDevice::default();
    let mut memory = ReplayBuffer::new::<BUFFER_LIMIT>();

    let policy_net: PolicyNet<NdArray> = PolicyNet::new(&device);
    let q_net_one: QNet<Autodiff<NdArray>> = QNet::new(&device);
    let q_net_two: QNet<Autodiff<NdArray>> = QNet::new(&device);
    let q_net_one_target: QNet<Autodiff<NdArray>> = QNet::new(&device);
    let q_net_two_target: QNet<Autodiff<NdArray>> = QNet::new(&device);

    q_net_one_target.load_record(q_net_one.into_record());
    q_net_two_target.load_record(q_net_two.into_record());

    let mut score: f32 = 0.0;
    let print_interval: usize = 20;

    for episode in 0..10000 {
        let (obeservation, _) = env.reset(None, false, None);
        let mut done = false;
        let mut count = 0;
        let mut previous_observation = obeservation;

        while count < 200 && !done {
            let state: Tensor<NdArray, 1> = convert_carte_pole_observation_to_tensor(obeservation, &device);
            let (action, log_probability) = policy_net.forward(state);

            let step_result = env.step(action.clone().into_scalar().to_usize());
            memory.put(
                // TODO: need to refine the `from_step_result` method
                Data::from_step_result(
                    previous_observation, 
                    step_result.observation, 
                    action.into_scalar().to_u8(), 
                    0.0, // this is ignored
                    step_result.done, 
                    step_result.reward.to_f32()
                )
            );

            score += step_result.reward.to_f32();
            previous_observation = step_result.observation;
            count += 1;
        }

        if memory.size() > 1000 {}
    }

    Ok(())
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
