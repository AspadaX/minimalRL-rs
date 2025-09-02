use gym_rs::envs::classical_control::cartpole::CartPoleObservation;

pub struct DataBatch {}

#[derive(Debug, Clone)]
pub struct ReplayBuffer {
    experiences: Vec<DataBatch>
}