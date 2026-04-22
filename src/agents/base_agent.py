# src/agents/base_agent.py
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    Base class for all reinforcement learning agents
    """
    def __init__(self, env, state_size, action_size):
        """
        Initialize the agent
        
        Args:
            env: Game environment
            state_size: Dimensions of the state space
            action_size: Dimensions of the action space
        """
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.name = "BaseAgent"
    
    @abstractmethod
    def act(self, state):
        """
        Select an action based on the current state
        
        Args:
            state: Current state of the environment
            
        Returns:
            action: Selected action
        """
        pass
    
    @abstractmethod
    def train(self, num_episodes):
        """
        Train the agent for a given number of episodes
        
        Args:
            num_episodes: Number of episodes to train
            
        Returns:
            training_history: Training metrics
        """
        pass
    
    @abstractmethod
    def save(self, filepath):
        """
        Save the agent's model to a file
        
        Args:
            filepath: Path to save the model
        """
        pass
    
    @abstractmethod
    def load(self, filepath):
        """
        Load the agent's model from a file
        
        Args:
            filepath: Path to load the model from
        """
        pass
    
    def evaluate(self, num_episodes=10):
        """
        Evaluate the agent's performance over multiple episodes
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            average_score: Average score achieved
            max_score: Maximum score achieved
            max_tile: Maximum tile achieved
        """
        scores = []
        max_tiles = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_score = 0
            
            while not done:
                action = self.act(state)
                next_state, reward, done, info = self.env.step(action)
                state = next_state
                episode_score += reward
            
            # Get max tile from the board
            max_tile = state.max()
            
            scores.append(info["score"])
            max_tiles.append(max_tile)
            
            print(f"Episode {episode+1}: Score = {info['score']}, Max Tile = {max_tile}")
        
        average_score = sum(scores) / len(scores)
        max_score = max(scores)
        max_tile_overall = max(max_tiles)
        
        print(f"\nEvaluation Results ({num_episodes} episodes):")
        print(f"Average Score: {average_score}")
        print(f"Max Score: {max_score}")
        print(f"Max Tile Achieved: {max_tile_overall}")
        
        return average_score, max_score, max_tile_overall