# src/agents/mcts_agent.py
import numpy as np
import copy
import math
import time
from collections import defaultdict

from agents.base_agent import BaseAgent

class Node:
    """
    Node class for MCTS tree structure
    """
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action  # Action that led to this state
        self.children = {}  # Map actions to child nodes
        self.visits = 0
        self.value = 0
        self.untried_actions = [0, 1, 2, 3]  # UP, RIGHT, DOWN, LEFT
    
    def is_fully_expanded(self):
        """Check if all possible actions have been tried from this state"""
        return len(self.untried_actions) == 0
    
    def best_child(self, exploration_weight=1.0):
        """
        Select the best child node according to UCB formula
        
        Args:
            exploration_weight: Weight for balancing exploration/exploitation
            
        Returns:
            best_child: The best child node
        """
        if not self.children:
            return None
            
        # UCB formula: (value / visits) + exploration_weight * sqrt(ln(parent_visits) / visits)
        def ucb(child):
            exploitation = child.value / child.visits if child.visits > 0 else 0
            exploration = exploration_weight * math.sqrt(2 * math.log(self.visits) / child.visits) if child.visits > 0 else float('inf')
            return exploitation + exploration
            
        return max(self.children.values(), key=ucb)


class MCTSAgent(BaseAgent):
    """
    Monte Carlo Tree Search agent for playing 2048
    """
    def __init__(
        self, 
        env, 
        state_size=(4, 4), 
        action_size=4, 
        simulation_count=100,
        search_depth=10,
        exploration_weight=1.0,
        time_limit=None
    ):
        """
        Initialize the MCTS agent
        
        Args:
            env: Game environment
            state_size: Dimensions of the state
            action_size: Number of possible actions
            simulation_count: Number of simulations per move
            search_depth: Maximum depth for rollout
            exploration_weight: Weight for UCB formula
            time_limit: Maximum time (in seconds) for selecting a move (None for no limit)
        """
        super().__init__(env, state_size, action_size)
        self.name = "MCTSAgent"
        
        # MCTS parameters
        self.simulation_count = simulation_count
        self.search_depth = search_depth
        self.exploration_weight = exploration_weight
        self.time_limit = time_limit
        
        # Statistics
        self.stats = {
            'simulations_per_move': [],
            'time_per_move': [],
            'depth_reached': []
        }
    
    def _copy_env(self, env):
        """
        Create a deep copy of the environment
        
        Args:
            env: Original environment
            
        Returns:
            env_copy: Copy of the environment
        """
        env_copy = copy.deepcopy(env)
        return env_copy
    
    def _is_valid_move(self, env, action):
        """
        Check if a move is valid in the given environment
        
        Args:
            env: Environment
            action: Action to check
            
        Returns:
            is_valid: Whether the move is valid
        """
        env_copy = self._copy_env(env)
        old_board = env_copy.board.copy()
        _, _, _, _ = env_copy.step(action)
        return not np.array_equal(old_board, env_copy.board)
    
    def _select(self, node):
        """
        Select a node to expand (using UCB)
        
        Args:
            node: Current node
            
        Returns:
            selected_node: Node selected for expansion
        """
        # Traverse tree until reaching a non-fully-expanded node or a terminal node
        current = node
        while not current.is_fully_expanded() and current.children:
            current = current.best_child(self.exploration_weight)
        
        return current
    
    def _expand(self, node, env):
        """
        Expand a node by adding a child node
        
        Args:
            node: Node to expand
            env: Environment
            
        Returns:
            child_node: Newly created child node
        """
        if not node.untried_actions:
            return node
            
        # Try actions until finding a valid one
        valid_action = None
        while node.untried_actions and valid_action is None:
            action = node.untried_actions.pop(0)
            if self._is_valid_move(env, action):
                valid_action = action
        
        if valid_action is None:
            return node
            
        # Create a new environment for the child
        child_env = self._copy_env(env)
        next_state, reward, done, _ = child_env.step(valid_action)
        
        # Create child node
        child_node = Node(state=next_state, parent=node, action=valid_action)
        node.children[valid_action] = child_node
        
        return child_node
    
    def _simulate(self, env, max_depth=10):
        """
        Perform a random rollout from the current state
        
        Args:
            env: Environment
            max_depth: Maximum depth for rollout
            
        Returns:
            total_reward: Cumulative reward from rollout
        """
        # Make a copy of the environment to avoid modifying the original
        sim_env = self._copy_env(env)
        
        # Keep track of cumulative reward
        cum_reward = 0
        depth = 0
        done = False
        
        while not done and depth < max_depth:
            # Try random valid actions
            valid_actions = [a for a in range(4) if self._is_valid_move(sim_env, a)]
            
            if not valid_actions:
                break
                
            # Select a random valid action
            action = np.random.choice(valid_actions)
            
            # Take the action
            _, reward, done, _ = sim_env.step(action)
            
            # Update reward
            cum_reward += reward
            depth += 1
        
        # Add a bonus for the final state
        if not done:
            # Heuristic: count empty cells and max tile value
            empty_cells = np.count_nonzero(sim_env.board == 0)
            max_tile = sim_env.board.max()
            
            # Bonus for empty cells and max tile
            cum_reward += empty_cells * 0.1
            cum_reward += math.log2(max_tile) * 0.5
        
        return cum_reward
    
    def _backpropagate(self, node, reward):
        """
        Backpropagate the reward through the tree
        
        Args:
            node: Leaf node
            reward: Reward to backpropagate
        """
        # Update statistics for all nodes in the path
        current = node
        while current:
            current.visits += 1
            current.value += reward
            current = current.parent
    
    def act(self, state, training=False):
        """
        Choose an action using MCTS
        
        Args:
            state: Current state
            training: Whether the agent is training (ignored in MCTS)
            
        Returns:
            best_action: Selected action
        """
        # Create a copy of the environment for MCTS
        env_copy = self._copy_env(self.env)
        env_copy.board = state.copy()
        
        # Start timing
        start_time = time.time()
        
        # Create root node
        root = Node(state=state)
        
        # Run MCTS for a fixed number of simulations or until time limit
        simulation_count = 0
        max_depth_reached = 0
        
        while simulation_count < self.simulation_count:
            if self.time_limit and (time.time() - start_time) > self.time_limit:
                break
                
            # Select
            selected_node = self._select(root)
            
            # Create a copy of the environment for this node
            node_env = self._copy_env(env_copy)
            
            # If the node is not the root, we need to simulate the actions to reach this state
            current = selected_node
            actions_to_root = []
            while current.parent:
                actions_to_root.append(current.action)
                current = current.parent
            
            # Execute actions in reverse order
            for action in reversed(actions_to_root):
                node_env.step(action)
            
            # Expand
            if not selected_node.is_fully_expanded():
                expanded_node = self._expand(selected_node, node_env)
                if expanded_node != selected_node:
                    # Update environment with the new action
                    action = expanded_node.action
                    node_env.step(action)
                    selected_node = expanded_node
            
            # Simulate
            reward = self._simulate(node_env, self.search_depth)
            
            # Backpropagate
            self._backpropagate(selected_node, reward)
            
            # Track depth
            depth = len(actions_to_root)
            max_depth_reached = max(max_depth_reached, depth)
            
            simulation_count += 1
        
        # Select the best action from the root
        best_action = 0
        best_value = float('-inf')
        
        for action, child in root.children.items():
            # Use exploitation only (no exploration) for the final decision
            value = child.value / child.visits if child.visits > 0 else 0
            if value > best_value:
                best_value = value
                best_action = action
        
        # Fallback to random valid action if no valid action found
        if best_value == float('-inf'):
            valid_actions = [a for a in range(4) if self._is_valid_move(env_copy, a)]
            if valid_actions:
                best_action = np.random.choice(valid_actions)
        
        # Track statistics
        end_time = time.time()
        self.stats['simulations_per_move'].append(simulation_count)
        self.stats['time_per_move'].append(end_time - start_time)
        self.stats['depth_reached'].append(max_depth_reached)
        
        return best_action
    
    def train(self, num_episodes, max_steps=None, render_every=None):
        """
        MCTS doesn't require explicit training, but this method plays games
        to collect statistics and refine parameters
        
        Args:
            num_episodes: Number of episodes to play
            max_steps: Maximum number of steps per episode (None for no limit)
            render_every: Render every N episodes (None for no rendering)
            
        Returns:
            history: Gaming history (scores, statistics, etc.)
        """
        scores = []
        max_tiles = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            step = 0
            
            while not done:
                if max_steps and step >= max_steps:
                    break
                
                # Render if needed
                if render_every is not None and episode % render_every == 0:
                    self.env.render()
                
                # Choose action
                action = self.act(state)
                
                # Take action
                next_state, reward, done, info = self.env.step(action)
                
                # Move to next state
                state = next_state
                step += 1
            
            # Get max tile from the board
            max_tile = state.max()
            
            scores.append(info["score"])
            max_tiles.append(max_tile)
            
            print(f"Episode {episode+1}/{num_episodes}: Score = {info['score']}, Max Tile = {max_tile}")
            print(f"Avg Simulations: {np.mean(self.stats['simulations_per_move'][-step:]):.1f}, " + 
                  f"Avg Time: {np.mean(self.stats['time_per_move'][-step:]):.3f}s")
        
        # Return history
        history = {
            "scores": scores,
            "max_tiles": max_tiles,
            "stats": self.stats
        }
        
        return history
    
    def save(self, filepath):
        """
        Save the agent's parameters and statistics
        
        Args:
            filepath: Path to save the data
        """
        # MCTS doesn't have parameters to save, but we can save statistics
        # Create directory if it doesn't exist
        import os
        import pickle
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save parameters and statistics
        data = {
            'simulation_count': self.simulation_count,
            'search_depth': self.search_depth,
            'exploration_weight': self.exploration_weight,
            'time_limit': self.time_limit,
            'stats': self.stats
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath):
        """
        Load the agent's parameters and statistics
        
        Args:
            filepath: Path to load the data from
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.simulation_count = data['simulation_count']
        self.search_depth = data['search_depth']
        self.exploration_weight = data['exploration_weight']
        self.time_limit = data['time_limit']
        self.stats = data['stats']