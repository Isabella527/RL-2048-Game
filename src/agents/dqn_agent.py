# src/agents/dqn_agent.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
import random
import os
from collections import deque

from agents.base_agent import BaseAgent

class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent for playing 2048
    """
    def __init__(
        self, 
        env, 
        state_size=(4, 4), 
        action_size=4, 
        memory_size=10000,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        learning_rate=0.001,
        batch_size=64,
        update_target_freq=100
    ):
        """
        Initialize the DQN agent
        
        Args:
            env: Game environment
            state_size: Dimensions of the state
            action_size: Number of possible actions
            memory_size: Size of replay buffer
            gamma: Discount factor
            epsilon: Exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
            learning_rate: Learning rate for the optimizer
            batch_size: Batch size for training
            update_target_freq: Frequency of target network updates
        """
        super().__init__(env, state_size, action_size)
        self.name = "DQNAgent"
        
        # DQN parameters
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.update_target_freq = update_target_freq
        
        # Network update counter
        self.update_counter = 0
        
        # Build networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self._update_target_network()
    
    def _build_model(self):
        """
        Build a neural network model for DQN
        
        Returns:
            model: Compiled Keras model
        """
        # Convert to log scale to handle large numbers better
        model = Sequential([
            # Reshape and convert to log scale
            tf.keras.layers.Lambda(
                lambda x: tf.math.log(tf.cast(x, tf.float32) + 1.0),
                input_shape=self.state_size + (1,)
            ),
            
            # Convolutional layers
            Conv2D(128, (2, 2), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (2, 2), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (2, 2), activation='relu', padding='same'),
            BatchNormalization(),
            
            # Flatten and Dense layers
            Flatten(),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dense(128, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def _update_target_network(self):
        """Update the target network with weights from the main model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def _preprocess_state(self, state):
        """
        Preprocess the state for the neural network
        
        Args:
            state: Raw state from the environment
            
        Returns:
            processed_state: Preprocessed state for the network
        """
        # Reshape to add channel dimension
        return np.expand_dims(state, axis=-1)
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        state = self._preprocess_state(state)
        next_state = self._preprocess_state(next_state)
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=False):
        """
        Choose an action based on the current state
        
        Args:
            state: Current state
            training: Whether the agent is training (use epsilon-greedy) or not
            
        Returns:
            action: Selected action
        """
        processed_state = self._preprocess_state(state)
        
        # Epsilon-greedy exploration during training
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # For training or evaluation, use the best action from the model
        act_values = self.model.predict(np.expand_dims(processed_state, axis=0), verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self):
        """
        Train the model using experience replay
        
        Returns:
            loss: Training loss
        """
        if len(self.memory) < self.batch_size:
            return 0
        
        # Sample a batch from memory
        minibatch = random.sample(self.memory, self.batch_size)
        states, targets = [], []
        
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0]
            
            if done:
                target[action] = reward
            else:
                # Double DQN
                # Select best action using the online network
                best_action = np.argmax(self.model.predict(np.expand_dims(next_state, axis=0), verbose=0)[0])
                # Evaluate the action using the target network
                target[action] = reward + self.gamma * \
                    self.target_model.predict(np.expand_dims(next_state, axis=0), verbose=0)[0][best_action]
            
            states.append(state)
            targets.append(target)
        
        # Train the model
        history = self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0, batch_size=self.batch_size)
        
        # Update target network if needed
        self.update_counter += 1
        if self.update_counter % self.update_target_freq == 0:
            self._update_target_network()
            self.update_counter = 0
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return history.history['loss'][0]
    
    def train(self, num_episodes, max_steps=None, render_every=None):
        """
        Train the DQN agent for a given number of episodes
        
        Args:
            num_episodes: Number of episodes to train
            max_steps: Maximum number of steps per episode (None for no limit)
            render_every: Render every N episodes (None for no rendering)
            
        Returns:
            history: Training history (scores, losses, etc.)
        """
        scores = []
        losses = []
        max_tiles = []
        
        print("Starting training...")  # Added progress indicator
        
        for episode in range(num_episodes):
            print(f"\nStarting episode {episode+1}/{num_episodes}...")  # Added progress indicator
            state = self.env.reset()
            done = False
            episode_score = 0
            episode_losses = []
            step = 0
            
            while not done:
                if max_steps and step >= max_steps:
                    break
                
                # Print progress every 10 steps
                if step % 10 == 0:  # Added progress indicator
                    print(f"  Step {step}, current score: {episode_score}", end="\r")
                
                # Render if needed
                if render_every is not None and episode % render_every == 0:
                    self.env.render()
                
                # Choose action
                action = self.act(state, training=True)
                
                # Take action
                next_state, reward, done, info = self.env.step(action)
                
                # Store experience
                self.remember(state, action, reward, next_state, done)
                
                # Train the network
                loss = self.replay()
                if loss > 0:
                    episode_losses.append(loss)
                
                # Move to next state
                state = next_state
                episode_score += reward
                step += 1
            
            # Get max tile from the board
            max_tile = state.max()
            
            scores.append(info["score"])
            max_tiles.append(max_tile)
            losses.append(np.mean(episode_losses) if episode_losses else 0)
            
            print(f"\nEpisode {episode+1}/{num_episodes} completed:")
            print(f"  Score = {info['score']}, Max Tile = {max_tile}, Epsilon = {self.epsilon:.4f}")
            print(f"  Memory size: {len(self.memory)}/{self.memory.maxlen}")
            
            # Calculate and display progress percentage
            progress = (episode + 1) / num_episodes * 100
            print(f"Overall progress: {progress:.1f}% complete")
            
            # Save model periodically
            if (episode + 1) % 100 == 0:
                self.save(f"models/dqn_2048_episode_{episode+1}.h5")
                print(f"Model saved at episode {episode+1}")
        
        # Save final model
        self.save("models/dqn_2048_final.h5")
        
        # Return training history
        history = {
            "scores": scores,
            "losses": losses,
            "max_tiles": max_tiles
        }
        
        return history
    
    def save(self, filepath):
        """
        Save the model to a file
        
        Args:
            filepath: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model
        self.model.save(filepath)
    
    def load(self, filepath):
        """
        Load the model from a file
        
        Args:
            filepath: Path to load the model from
        """
        self.model = load_model(filepath)
        self._update_target_network()