# src/utils/replay_buffer.py
import numpy as np
import random
from collections import deque, namedtuple

# Define Experience tuple for cleaner code
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples
    """
    def __init__(self, buffer_size, batch_size):
        """
        Initialize a ReplayBuffer object
        
        Args:
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = Experience
    
    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to memory
        
        Args:
            state: current state
            action: action taken
            reward: reward received
            next_state: resulting state
            done: whether the episode is complete
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """
        Randomly sample a batch of experiences from memory
        
        Returns:
            Dictionary containing states, actions, rewards, next_states, and dones
        """
        experiences = random.sample(self.memory, k=min(self.batch_size, len(self.memory)))
        
        states = np.array([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None])
        rewards = np.array([e.reward for e in experiences if e is not None])
        next_states = np.array([e.next_state for e in experiences if e is not None])
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.float32)
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }
    
    def __len__(self):
        """Return the current size of internal memory"""
        return len(self.memory)


class PrioritizedReplayBuffer:
    """
    Fixed-size buffer with prioritized experience replay
    """
    def __init__(self, buffer_size, batch_size, alpha=0.6, beta=0.4, beta_increment=0.001):
        """
        Initialize a PrioritizedReplayBuffer object
        
        Args:
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
            alpha: determines how much prioritization is used (0 = uniform, 1 = fully prioritized)
            beta: importance-sampling weight (0 = no correction, 1 = full correction)
            beta_increment: increment of beta per sampling
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = Experience
        self.priorities = deque(maxlen=buffer_size)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 1e-6  # small constant to ensure non-zero priority
    
    def add(self, state, action, reward, next_state, done, error=None):
        """
        Add a new experience to memory with priority
        
        Args:
            state: current state
            action: action taken
            reward: reward received
            next_state: resulting state
            done: whether the episode is complete
            error: TD error for prioritization (if None, max priority is used)
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
        # New experiences get max priority
        if error is None:
            max_priority = max(self.priorities) if self.priorities else 1.0
            self.priorities.append(max_priority)
        else:
            # Use provided error for priority
            self.priorities.append((abs(error) + self.epsilon) ** self.alpha)
    
    def sample(self):
        """
        Sample a batch of experiences based on priority
        
        Returns:
            Dictionary containing states, actions, rewards, next_states, dones, indices, and weights
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probabilities = priorities / sum(priorities)
        
        # Sample batch and get indices
        indices = np.random.choice(len(self.memory), self.batch_size, p=probabilities)
        experiences = [self.memory[idx] for idx in indices]
        
        # Calculate importance-sampling weights
        weights = (len(self.memory) * probabilities[indices]) ** (-self.beta)
        weights /= max(weights)  # Normalize weights
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Extract values
        states = np.array([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None])
        rewards = np.array([e.reward for e in experiences if e is not None])
        next_states = np.array([e.next_state for e in experiences if e is not None])
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.float32)
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'indices': indices,
            'weights': weights
        }
    
    def update_priorities(self, indices, errors):
        """
        Update priorities for sampled experiences
        
        Args:
            indices: indices of sampled experiences
            errors: TD errors for each experience
        """
        for idx, error in zip(indices, errors):
            if idx < len(self.priorities):  # Safety check
                self.priorities[idx] = (abs(error) + self.epsilon) ** self.alpha
    
    def __len__(self):
        """Return the current size of internal memory"""
        return len(self.memory)