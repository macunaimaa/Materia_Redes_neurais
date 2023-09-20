import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from grid import CustomGridEnv as GridEnv

# Set up the environment
env = GridEnv(grid_size=5)
env.reset()

# Set up the hyperparameters
gamma = 0.90
learning_rate = 0.001
num_episodes = 10000
max_steps = 100
num_steps_total = []
num_episodes_total = []
reward_total = []
reward_list = []
loss_list = []
loss_total = []
loss_total_list = []
reward_total_list = []
num_steps_mean = []

# Set up the neural network
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space = 11
        self.action_space = 4
        self.fc1 = nn.Linear(self.state_space, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, self.action_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc3(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

# Initialize policy network and optimizer
policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

def select_action(state):
    state = np.array(state)
    state = torch.from_numpy(state).type(torch.FloatTensor)
    action_probs = policy(torch.autograd.Variable(state))
    c = torch.distributions.Categorical(action_probs)
    action = c.sample()
    log_prob = c.log_prob(action)  # Calculate log probability of the chosen action
    return action, log_prob

def update_policy(reward_episode, log_probs):
    discounts = [gamma**i for i in range(len(reward_episode)+1)]
    R = sum([a*b for a,b in zip(discounts, reward_episode)])
    policy_loss = []
    for log_prob in log_probs:
        policy_loss.append(-log_prob * R)
    policy_loss = torch.cat(policy_loss).sum()
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

def save_model(episode):
    save_dir = 'modelos'
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f'policy{episode}.pth')
    torch.save(policy, model_path)
    print("model saved")

def get_observation_data(state):
    observations = env.observe_surroundings(state)
    observations = np.array(observations)
    state = np.array(state)
    state = np.concatenate((state, observations))
    state = state.reshape(1, 11)
    return state

def main():
    for episode in range(num_episodes):
        state = env.reset()
        state = get_observation_data(state)
        reward_episode = []
        log_probs = []  # Collect log probabilities of actions
        for steps in range(max_steps):
            if steps != 0:
                state = get_observation_data(state)
            action, log_prob = select_action(state)
            log_probs.append(log_prob)
            state, reward, done, _ = env.step(action.item())
            reward_episode.append(reward)
            if done:
                break
        update_policy(reward_episode, log_probs)
        num_steps_total.append(steps)
        num_steps_mean.append(np.mean(num_steps_total[-100:]))
        num_episodes_total.append(episode)
        reward_total.append(reward)
        reward_total_list.append(np.mean(reward_total[-100:]))
        print("episode: {}, total reward: {}, average_reward: {}, total_steps: {}, average_steps: {}".format(
            episode, np.round(reward_total[-1], decimals=3),
            np.round(np.mean(reward_total[-100:]), decimals=3),
            np.sum(num_steps_total), np.mean(num_steps_mean))
        )
        if episode % 100 == 0:
            save_model(episode)

    # Plotting and saving
    save_path = 'C:\\Users\\jpedr\\OneDrive\\Área de Trabalho\\projetos\\Trabalho_3_RNP\\imagens'
    plot_and_save(save_path)

def plot_and_save(save_path):
    plt.plot(num_episodes_total, num_steps_mean)
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.savefig(f'{save_path}/steps.png')
    plt.close()

    plt.plot(num_episodes_total, reward_total)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.savefig(f'{save_path}/reward.png')
    plt.close()

    plt.plot(num_episodes_total, reward_total_list)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.savefig(f'{save_path}/average_reward.png')
    plt.close()

if __name__ == '__main__':
    main()
