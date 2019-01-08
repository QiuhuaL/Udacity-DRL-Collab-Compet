import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128       # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 2        # Frequency to update the networks (how many time steps to update once)
NOISE_WEIGHT_START = 1.0  # Initial noise weighting factor
NOISE_WEIGHT_DECAY = 0.9999  # Rate of noise weighting factor decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed, num_agents=2):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)
        self.time_step = 0
        self.noise_weight = NOISE_WEIGHT_START

        # Actor Network (w/ Target Network), 2 DDPG agents
        self.actors_local = [Actor(state_size, action_size, random_seed).to(device), Actor(state_size, action_size, random_seed).to(device)]
        self.actors_target = [Actor(state_size, action_size, random_seed).to(device), Actor(state_size, action_size, random_seed).to(device)]
        self.actors_optimizer = [optim.Adam(self.actors_local[0].parameters(), lr=LR_ACTOR), optim.Adam(self.actors_local[1].parameters(), lr=LR_ACTOR)]

        # Critic Network (w/ Target Network, 1 shared critic
        critic_input_size = (state_size + action_size) * num_agents
        self.critic_local = Critic(critic_input_size, action_size, random_seed).to(device)
        self.critic_target = Critic(critic_input_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(size=(self.num_agents, self.action_size), seed=random_seed)

        # Initialize the weights of the target network the same as those of the local network
        #self.hard_copy(self.actors_target[0], self.actors_local[0])
        #self.hard_copy(self.actors_target[1], self.actors_local[1])
        #self.hard_copy(self.critic_target, self.critic_local)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def return_actors(self):
        return self.actors_local

    def return_critic(self):
        return self.critic_local

    def hard_copy(self, target, source):
            """ Copy weights from one network to another network"""
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(param.data)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        #print(state.shape)
        #print(state)
        self.memory.add(state, action, reward, next_state, done)
        # Learn, if enough samples are available in memory
        self.time_step += 1
        if len(self.memory) > BATCH_SIZE and self.time_step % UPDATE_EVERY == 0:
           self.learn(GAMMA)
        self.noise_weight *= NOISE_WEIGHT_DECAY

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        # for multiple agents
        states = torch.from_numpy(np.array(states)).float().to(device)
        actions = np.zeros((self.num_agents, self.action_size))
        for i, actor in enumerate(self.actors_local):
            state = states[i, :]
            actor.eval()
            with torch.no_grad():
                action = actor(state)
            actor.train()
            actions[i, :] = action.cpu().numpy()
        if add_noise:
            actions += np.array(self.noise.sample())*self.noise_weight
            #actions += np.array(self.noise.sample())
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """

        # Learn for multiple agents
        for i in range(self.num_agents):
            states, actions, rewards, next_states, dones = self.memory.sample()
            # separate experiences into agents
            #print(states.shape)
            #print(states)
            #print(actions)
            #print(next_states)
            #print(rewards)
            #print(dones)
            
            states_i = states[:, i, :]
            #actions_i = actions[:, i, :]
            rewards_i = rewards[:, i]
            #next_states_i = next_states[:, i, :]
            dones_i = dones[:, i]

            # ---------------------------- update critic ---------------------------- #
            # Get predicted next-state actions and Q values from target models

            next_states_actions = torch.cat((next_states[:, 0, :], next_states[:, 1, :],
                                            self.actors_target[0](next_states[:, 0, :]),
                                            self.actors_target[1](next_states[:, 1, :])), dim=1)
            q_targets_next = self.critic_target(next_states_actions).detach()
            # Compute Q targets for current states (y_i)
            q_targets = rewards_i + (gamma * q_targets_next[:, i] * (1 - dones_i))

            # Compute critic loss
            states_actions = torch.cat((states[:, 0, :], states[:, 1, :],
                                        actions[:, 0, :], actions[:, 1, :]), dim=1)
            q_expected = self.critic_local(states_actions)
            critic_loss = F.mse_loss(q_expected[:, i], q_targets)

            # Minimize the loss
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
            self.critic_optimizer.step()

            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            actions_pred = copy.copy(actions)
            actions_pred[:, i, :] = self.actors_local[i](states_i)
            actor_loss = -self.critic_local(torch.cat((states[:, 0, :], states[:, 1, :], actions_pred[:, 0, :],
                                                       actions_pred[:, 1, :]), dim=1))[:, i].mean()

            # Minimize the loss
            self.actors_optimizer[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors_local[i].parameters(), 1)
            self.actors_optimizer[i].step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic_local, self.critic_target, TAU)
            self.soft_update(self.actors_local[i], self.actors_target[i], TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.1):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)   ### np.random or random module direcrly???????????????
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample_numpy(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        # dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.random_sample(self.size)])
        state = x + dx
        #print(x)
        #print(dx)
        self.state = np.squeeze(state, axis=0)
        return self.state

    def sample(self):    # the original sample
         """Update internal state and return it as a noise sample."""
         x = self.state
         # dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
         #nn = np.ones((2, 2))
         #nn[0, 0] = random.random()
         #nn[0, 1] = random.random()
         #nn[1, 0] = random.random()
         #nn[1, 1] = random.random()
         nn = np.ones(self.size)
         for i, j in zip(range(self.size[0]),range(self.size[1])):
            nn[i, j] = random.random()
         dx = self.theta * (self.mu - x) + self.sigma * nn
         self.state = x + dx
         return self.state    


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        #states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        #actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        #rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        #next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        #dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        states = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.array([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.array([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
