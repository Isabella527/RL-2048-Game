# src/agents/ppo_agent.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from collections import deque

from agents.base_agent import BaseAgent
from models.neural_networks import create_combined_model

class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization agent for playing 2048
    """
    def __init__(
        self, 
        env, 
        state_size=(4, 4, 1), 
        action_size=4, 
        gamma=0.99,
        clip_ratio=0.2,
        policy_learning_rate=0.0003,
        value_learning_rate=0.001,
        batch_size=64,
        epochs=10,
        gae_lambda=0.95,
        entropy_beta=0.01
    ):
        """
        Initialize the PPO agent
        
        Args:
            env: Game environment
            state_size: Dimensions of the state
            action_size: Number of possible actions
            gamma: Discount factor
            clip_ratio: PPO clipping parameter
            policy_learning_rate: Learning rate for the policy network
            value_learning_rate: Learning rate for the value network
            batch_size: Batch size for training
            epochs: Number of epochs to train on each batch
            gae_lambda: Lambda parameter for Generalized Advantage Estimation
            entropy_beta: Coefficient for entropy bonus
        """
        super().__init__(env, state_size, action_size)
        self.name = "PPOAgent"
        
        # PPO parameters
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.policy_lr = policy_learning_rate
        self.value_lr = value_learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.gae_lambda = gae_lambda
        self.entropy_beta = entropy_beta
        
        # Build models
        self.combined_model, self.actor, self.critic = create_combined_model(
            input_shape=state_size,
            action_size=action_size,
            learning_rate=policy_learning_rate
        )
        
        # Create optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=policy_learning_rate)
        
        # Memory buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.action_probs = []
        self.values = []
    
    def _preprocess_state(self, state):
        """
        Preprocess the state for the neural network
        
        Args:
            state: Raw state from the environment
            
        Returns:
            processed_state: Preprocessed state for the network
        """
        # Reshape to include channel dimension if needed
        if len(state.shape) == 2:
            return np.expand_dims(state, axis=-1)
        return state
    
    def act(self, state, training=False):
        """
        Choose an action based on the current state
        
        Args:
            state: Current state
            training: Whether the agent is training or not
            
        Returns:
            action: Selected action
            action_probs: Probability distribution over actions (only if training)
            value: State value (only if training)
        """
        processed_state = self._preprocess_state(state)
        processed_state = np.expand_dims(processed_state, axis=0)  # Add batch dimension
        
        # Get action probabilities from actor network
        action_probs = self.actor.predict(processed_state, verbose=0)[0]
        
        if training:
            # Get state value from critic network
            value = self.critic.predict(processed_state, verbose=0)[0, 0]
            
            # Sample action from probability distribution
            action = np.random.choice(self.action_size, p=action_probs)
            
            return action, action_probs, value
        else:
            # For evaluation, use the action with highest probability
            action = np.argmax(action_probs)
            return action
    
    def remember(self, state, action, reward, next_state, done, action_probs, value):
        """
        Store experience in memory
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
            action_probs: Action probability distribution
            value: State value estimate
        """
        self.states.append(self._preprocess_state(state))
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(self._preprocess_state(next_state))
        self.dones.append(done)
        self.action_probs.append(action_probs)
        self.values.append(value)
    
    def _compute_gae(self, rewards, values, next_values, dones):
        """
        Compute Generalized Advantage Estimation
        
        Args:
            rewards: Array of rewards
            values: Array of state values
            next_values: Array of next state values
            dones: Array of done flags
            
        Returns:
            advantages: Computed advantages
            returns: Target values for critic (value function)
        """
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0
        
        # Start from the end and work backwards
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                # For the last step, use the next_value
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
            
            # Calculate TD error
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            
            # Calculate GAE
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        # Calculate returns (target values for critic)
        returns = advantages + np.array(values)
        
        return advantages, returns
    
    def _ppo_loss(self, old_probs, actions, advantages):
        """
        PPO loss function for the policy network
        
        Args:
            old_probs: Old action probabilities
            actions: Actions taken
            advantages: Advantage estimates
            
        Returns:
            loss: PPO loss
        """
        def loss_fn(y_true, y_pred):
            # y_pred contains the new action probabilities
            # Create a one-hot encoding for the actions
            actions_oh = tf.one_hot(actions, self.action_size)
            
            # Get new action probabilities
            new_probs = tf.reduce_sum(y_pred * actions_oh, axis=1)
            old_probs_tensor = tf.convert_to_tensor(old_probs, dtype=tf.float32)
            
            # Calculate ratio
            ratio = new_probs / old_probs_tensor
            
            # Calculate surrogate losses
            surrogate1 = ratio * advantages
            surrogate2 = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            
            # Calculate policy loss
            policy_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
            
            # Add entropy bonus for exploration
            entropy = -tf.reduce_mean(tf.reduce_sum(y_pred * tf.math.log(y_pred + 1e-10), axis=1))
            
            return policy_loss - self.entropy_beta * entropy
        
        return loss_fn
    
    def _value_loss(self, returns):
        """
        Loss function for the value network
        
        Args:
            returns: Target values
            
        Returns:
            loss: Value loss
        """
        def loss_fn(y_true, y_pred):
            # MSE loss
            return tf.reduce_mean(tf.square(y_pred - returns))
        
        return loss_fn
    
    def train(self, num_episodes, max_steps=None, render_every=None):
        """
        Train the PPO agent for a given number of episodes
        
        Args:
            num_episodes: Number of episodes to train
            max_steps: Maximum number of steps per episode (None for no limit)
            render_every: Render every N episodes (None for no rendering)
            
        Returns:
            history: Training history (scores, losses, etc.)
        """
        scores = []
        policy_losses = []
        value_losses = []
        max_tiles = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_score = 0
            episode_step = 0
            
            # Reset episode memory
            self.states = []
            self.actions = []
            self.rewards = []
            self.next_states = []
            self.dones = []
            self.action_probs = []
            self.values = []
            
            while not done:
                if max_steps and episode_step >= max_steps:
                    break
                
                # Render if needed
                if render_every is not None and episode % render_every == 0:
                    self.env.render()
                
                # Select action
                action, action_probs, value = self.act(state, training=True)
                
                # Take action
                next_state, reward, done, info = self.env.step(action)
                
                # Store experience
                self.remember(state, action, reward, next_state, done, action_probs[action], value)
                
                # Move to next state
                state = next_state
                episode_score += reward
                episode_step += 1
            
            # Get max tile from the board
            max_tile = state.max()
            
            # Process episode data
            if len(self.states) > 0:  # Make sure we have data to train on
                # Prepare data for training
                states = np.array(self.states)
                actions = np.array(self.actions)
                rewards = np.array(self.rewards)
                next_states = np.array(self.next_states)
                dones = np.array(self.dones)
                old_action_probs = np.array([probs for probs in self.action_probs])
                values = np.array(self.values)
                
                # Get next state values for GAE calculation
                next_values = np.zeros_like(values)
                if not done:
                    last_value = self.critic.predict(np.expand_dims(next_states[-1], axis=0), verbose=0)[0, 0]
                    next_values[-1] = last_value
                
                # Compute advantages and returns
                advantages, returns = self._compute_gae(rewards, values, next_values, dones)
                
                # Normalize advantages
                advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
                
                # Train for multiple epochs
                policy_loss = 0
                value_loss = 0
                
                # Train policy and value networks
                for _ in range(self.epochs):
                    # Create batches
                    indices = np.arange(len(states))
                    np.random.shuffle(indices)
                    
                    for start_idx in range(0, len(states), self.batch_size):
                        end_idx = min(start_idx + self.batch_size, len(states))
                        batch_indices = indices[start_idx:end_idx]
                        
                        batch_states = states[batch_indices]
                        batch_actions = actions[batch_indices]
                        batch_old_probs = old_action_probs[batch_indices]
                        batch_advantages = advantages[batch_indices]
                        batch_returns = returns[batch_indices]
                        
                        # Create dummy targets (not used in the custom loss)
                        dummy_targets = np.zeros((len(batch_indices), self.action_size))
                        dummy_values = np.zeros((len(batch_indices), 1))
                        
                        # Train actor (policy) network
                        with tf.GradientTape() as tape:
                            action_probs = self.actor(batch_states, training=True)
                            selected_action_probs = tf.reduce_sum(
                                action_probs * tf.one_hot(batch_actions, self.action_size),
                                axis=1
                            )
                            
                            # Calculate ratio
                            ratio = selected_action_probs / batch_old_probs
                            
                            # Calculate surrogate losses
                            surrogate1 = ratio * batch_advantages
                            surrogate2 = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                            
                            # Calculate policy loss
                            policy_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
                            
                            # Add entropy bonus for exploration
                            entropy = -tf.reduce_mean(tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-10), axis=1))
                            actor_loss = policy_loss - self.entropy_beta * entropy
                        
                        # Calculate gradients and update actor network
                        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
                        self.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
                        
                        # Train critic (value) network
                        with tf.GradientTape() as tape:
                            values_pred = self.critic(batch_states, training=True)
                            critic_loss = tf.reduce_mean(tf.square(values_pred - tf.expand_dims(batch_returns, axis=1)))
                        
                        # Calculate gradients and update critic network
                        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
                        self.optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
                        
                        policy_loss += actor_loss.numpy()
                        value_loss += critic_loss.numpy()
                
                # Average losses over epochs and batches
                policy_loss /= (self.epochs * (len(states) // self.batch_size + 1))
                value_loss /= (self.epochs * (len(states) // self.batch_size + 1))
                
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
            else:
                policy_losses.append(0)
                value_losses.append(0)
            
            scores.append(info["score"])
            max_tiles.append(max_tile)
            
            print(f"Episode {episode+1}/{num_episodes}: Score = {info['score']}, Max Tile = {max_tile}")
            print(f"Policy Loss: {policy_losses[-1]:.6f}, Value Loss: {value_losses[-1]:.6f}")
            
            # Save model periodically
            if (episode + 1) % 100 == 0:
                self.save(f"models/ppo_2048_episode_{episode+1}")
        
        # Save final model
        self.save("models/ppo_2048_final")
        
        # Return training history
        history = {
            "scores": scores,
            "policy_losses": policy_losses,
            "value_losses": value_losses,
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
        os.makedirs(filepath, exist_ok=True)
        
        # Save the models
        self.actor.save(f"{filepath}/actor.h5")
        self.critic.save(f"{filepath}/critic.h5")
    
    def load(self, filepath):
        """
        Load the model from a file
        
        Args:
            filepath: Path to load the model from
        """
        self.actor = load_model(f"{filepath}/actor.h5")
        self.critic = load_model(f"{filepath}/critic.h5")