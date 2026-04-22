# src/utils/visualizer.py
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from matplotlib.colors import LogNorm
from IPython.display import clear_output

# Set style
sns.set_style("whitegrid")

class Visualizer:
    """
    Visualization tools for 2048 RL experiments
    """
    def __init__(self, save_dir='visualizations'):
        """
        Initialize the visualizer
        
        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_board(self, board, title=None, save_path=None):
        """
        Plot the 2048 game board
        
        Args:
            board: 2D numpy array representing the game board
            title: Title for the plot
            save_path: Path to save the visualization
        """
        # Define color map (logarithmic scale for better visualization)
        cmap = plt.cm.YlOrRd
        
        plt.figure(figsize=(6, 6))
        ax = sns.heatmap(
            board, 
            annot=True, 
            cmap=cmap,
            linewidths=5,
            linecolor='#BCBCBC',
            cbar=False,
            square=True,
            fmt="d",
            norm=LogNorm(vmin=2, vmax=board.max() * 2) if board.max() > 0 else None
        )
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add title if provided
        if title:
            plt.title(title, fontsize=16)
        
        plt.tight_layout()
        
        # Save if path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=100)
            
        plt.show()
    
    def plot_training_history(self, history, save_path=None):
        """
        Plot training history metrics
        
        Args:
            history: Dictionary containing training metrics
            save_path: Path to save the visualization
        """
        plt.figure(figsize=(15, 12))
        
        # Plot scores
        plt.subplot(2, 2, 1)
        plt.plot(history['scores'], label='Score')
        plt.plot(np.convolve(history['scores'], np.ones(10)/10, mode='valid'), 
                 label='Moving Avg (10 episodes)', linewidth=2)
        plt.title('Game Score per Episode', fontsize=14)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot max tiles
        plt.subplot(2, 2, 2)
        plt.plot(history['max_tiles'], label='Max Tile')
        plt.title('Maximum Tile per Episode', fontsize=14)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Max Tile Value', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Plot tile frequency (histogram)
        max_tiles = np.array(history['max_tiles'])
        plt.subplot(2, 2, 3)
        unique_tiles, counts = np.unique(max_tiles, return_counts=True)
        bars = plt.bar(unique_tiles, counts)
        
        # Add percentage labels
        for i, count in enumerate(counts):
            percentage = count / len(max_tiles) * 100
            plt.text(unique_tiles[i], count, f'{percentage:.1f}%', 
                    ha='center', va='bottom', fontsize=10)
        
        plt.title('Max Tile Distribution', fontsize=14)
        plt.xlabel('Tile Value', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Plot losses if available
        if 'losses' in history:
            plt.subplot(2, 2, 4)
            losses = np.array(history['losses'])
            valid_losses = losses[~np.isnan(losses)]  # Remove NaN values
            plt.plot(valid_losses, label='Loss')
            if len(valid_losses) > 10:  # Only calculate moving average if enough data
                plt.plot(np.convolve(valid_losses, np.ones(10)/10, mode='valid'), 
                         label='Moving Avg (10 episodes)', linewidth=2)
            plt.title('Training Loss', fontsize=14)
            plt.xlabel('Episode', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=100)
            
        plt.show()
    
    def plot_live_training(self, episode, score, max_tile, epsilon, avg_scores, max_tiles, losses=None):
        """
        Plot training progress in real-time
        
        Args:
            episode: Current episode number
            score: Latest episode score
            max_tile: Maximum tile achieved
            epsilon: Current exploration rate
            avg_scores: List of running average scores
            max_tiles: List of maximum tiles
            losses: List of training losses
        """
        clear_output(wait=True)
        plt.figure(figsize=(15, 8))
        
        # Plot scores
        plt.subplot(2, 2, 1)
        plt.plot(avg_scores)
        plt.title(f'Avg Score: {np.mean(avg_scores[-10:]):.2f}')
        plt.xlabel('Episode')
        plt.ylabel('Avg Score (100 episodes)')
        
        # Plot max tiles
        plt.subplot(2, 2, 2)
        plt.plot(max_tiles)
        plt.title(f'Latest Max Tile: {max_tile}')
        plt.xlabel('Episode')
        plt.ylabel('Max Tile')
        
        # Plot epsilon
        plt.subplot(2, 2, 3)
        plt.text(0.5, 0.5, f'ε: {epsilon:.4f}\nEpisode: {episode}\nLast Score: {score}', 
                 horizontalalignment='center', verticalalignment='center', fontsize=12)
        plt.axis('off')
        
        # Plot losses if available
        if losses is not None and len(losses) > 0:
            plt.subplot(2, 2, 4)
            valid_losses = [l for l in losses if l is not None]
            if valid_losses:
                plt.plot(valid_losses)
                plt.title(f'Latest Loss: {valid_losses[-1]:.4f}')
                plt.xlabel('Episode')
                plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.show()
    
    def compare_agents(self, agent_results, save_path=None):
        """
        Compare performance of multiple agents
        
        Args:
            agent_results: Dictionary with agent names as keys and performance metrics as values
            save_path: Path to save the visualization
        """
        plt.figure(figsize=(15, 12))
        
        # Plot average scores
        plt.subplot(2, 2, 1)
        for agent_name, results in agent_results.items():
            plt.plot(results['scores'], label=agent_name)
        plt.title('Score Comparison', fontsize=14)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot max tiles
        plt.subplot(2, 2, 2)
        for agent_name, results in agent_results.items():
            plt.plot(results['max_tiles'], label=agent_name)
        plt.title('Max Tile Comparison', fontsize=14)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Max Tile', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Bar chart for average scores
        plt.subplot(2, 2, 3)
        avg_scores = {agent: np.mean(results['scores']) for agent, results in agent_results.items()}
        bars = plt.bar(avg_scores.keys(), avg_scores.values())
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=12)
            
        plt.title('Average Score by Agent', fontsize=14)
        plt.ylabel('Average Score', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Bar chart for max tiles (max of max_tiles)
        plt.subplot(2, 2, 4)
        max_of_max_tiles = {agent: max(results['max_tiles']) for agent, results in agent_results.items()}
        bars = plt.bar(max_of_max_tiles.keys(), max_of_max_tiles.values())
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}',
                    ha='center', va='bottom', fontsize=12)
            
        plt.title('Maximum Tile Achieved by Agent', fontsize=14)
        plt.ylabel('Maximum Tile', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=100)
            
        plt.show()