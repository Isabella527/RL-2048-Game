# src/main.py
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import time

from environment.game_2048 import Game2048Env
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.mcts_agent import MCTSAgent
from utils.visualizer import Visualizer

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='2048 Reinforcement Learning')
    
    parser.add_argument('--agent', type=str, default='dqn', 
                        choices=['dqn', 'ppo', 'mcts', 'all'],
                        help='Agent type to use (default: dqn)')
    
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'evaluate', 'play', 'compare'],
                        help='Mode to run (default: train)')
    
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes (default: 1000)')
    
    parser.add_argument('--eval-episodes', type=int, default=100,
                        help='Number of evaluation episodes (default: 100)')
    
    parser.add_argument('--render', action='store_true',
                        help='Render the environment during evaluation')
    
    parser.add_argument('--render-delay', type=float, default=0.1,
                        help='Delay between renders in seconds (default: 0.1)')
    
    parser.add_argument('--load-model', type=str, default=None,
                        help='Path to load the model from')
    
    parser.add_argument('--save-model', type=str, default='models',
                        help='Directory to save the model to (default: models)')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize training progress and results')
    
    return parser.parse_args()

def create_agent(agent_type, env):
    """
    Create an agent based on the specified type
    
    Args:
        agent_type: Type of agent to create
        env: Game environment
        
    Returns:
        agent: Created agent
    """
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    
    if agent_type == 'dqn':
        return DQNAgent(env, state_size, action_size)
    elif agent_type == 'ppo':
        # Add channel dimension for convolutional layers
        conv_state_size = state_size + (1,)
        return PPOAgent(env, conv_state_size, action_size)
    elif agent_type == 'mcts':
        return MCTSAgent(env, state_size, action_size)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

def train_agent(agent, args):
    """
    Train the agent
    
    Args:
        agent: Agent to train
        args: Command line arguments
        
    Returns:
        history: Training history
    """
    print(f"Training {agent.name} for {args.episodes} episodes...")
    
    # Train the agent
    history = agent.train(
        num_episodes=args.episodes,
        render_every=100 if args.render else None
    )
    
    # Save the trained model
    os.makedirs(args.save_model, exist_ok=True)
    agent.save(f"{args.save_model}/{agent.name.lower()}_final")
    
    # Visualize if requested
    if args.visualize:
        visualizer = Visualizer()
        visualizer.plot_training_history(
            history,
            save_path=f"visualizations/{agent.name}_training_history.png"
        )
    
    return history

def evaluate_agent(agent, args):
    """
    Evaluate the agent
    
    Args:
        agent: Agent to evaluate
        args: Command line arguments
        
    Returns:
        results: Evaluation results
    """
    print(f"Evaluating {agent.name} for {args.eval_episodes} episodes...")
    
    # Load the model if specified
    if args.load_model:
        print(f"Loading model from {args.load_model}")
        agent.load(args.load_model)
    
    env = agent.env
    scores = []
    max_tiles = []
    
    for episode in range(args.eval_episodes):
        state = env.reset()
        done = False
        episode_score = 0
        
        while not done:
            # Render if requested
            if args.render:
                env.render()
                time.sleep(args.render_delay)
            
            # Choose action
            action = agent.act(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Move to next state
            state = next_state
            episode_score += reward
        
        # Get max tile
        max_tile = state.max()
        
        scores.append(info["score"])
        max_tiles.append(max_tile)
        
        print(f"Episode {episode+1}: Score = {info['score']}, Max Tile = {max_tile}")
    
    # Print summary
    print("\nEvaluation Results:")
    print(f"Average Score: {np.mean(scores)}")
    print(f"Max Score: {max(scores)}")
    print(f"Average Max Tile: {np.mean(max_tiles)}")
    print(f"Max Tile Achieved: {max(max_tiles)}")
    
    # Visualize if requested
    if args.visualize:
        visualizer = Visualizer()
        
        # Create a simple evaluation history
        eval_history = {
            'scores': scores,
            'max_tiles': max_tiles
        }
        
        visualizer.plot_training_history(
            eval_history,
            save_path=f"visualizations/{agent.name}_evaluation_results.png"
        )
    
    return {
        'scores': scores,
        'max_tiles': max_tiles
    }

def play_game(agent, args):
    """
    Play a single game with the agent and visualize it
    
    Args:
        agent: Agent to play with
        args: Command line arguments
    """
    print(f"Playing a game with {agent.name}...")
    
    # Load the model if specified
    if args.load_model:
        print(f"Loading model from {args.load_model}")
        agent.load(args.load_model)
    
    env = agent.env
    visualizer = Visualizer()
    
    state = env.reset()
    done = False
    episode_score = 0
    step = 0
    
    # Visualize initial state
    visualizer.plot_board(state, title=f"Step {step} - Score: 0")
    
    while not done:
        # Choose action
        action = agent.act(state)
        action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        
        # Take action
        next_state, reward, done, info = env.step(action)
        
        # Update step and score
        step += 1
        episode_score += reward
        
        # Visualize
        visualizer.plot_board(
            next_state, 
            title=f"Step {step} - Action: {action_names[action]} - Score: {info['score']}"
        )
        
        # Add delay for better visualization
        time.sleep(args.render_delay)
        
        # Move to next state
        state = next_state
    
    print(f"Game Over! Final Score: {info['score']}, Max Tile: {state.max()}")

def compare_agents(args):
    """
    Compare multiple agents
    
    Args:
        args: Command line arguments
    """
    print("Comparing agents...")
    
    # Create environment
    env = Game2048Env()
    
    # Create agents
    agents = {
        'DQN': DQNAgent(env, env.observation_space.shape, env.action_space.n),
        'PPO': PPOAgent(env, env.observation_space.shape + (1,), env.action_space.n),
        'MCTS': MCTSAgent(env, env.observation_space.shape, env.action_space.n)
    }
    
    # Load models if specified
    if args.load_model:
        for name, agent in agents.items():
            model_path = f"{args.load_model}/{name.lower()}_final"
            if os.path.exists(model_path):
                print(f"Loading {name} from {model_path}")
                agent.load(model_path)
    
    # Evaluate agents
    results = {}
    
    for name, agent in agents.items():
        print(f"\nEvaluating {name} agent...")
        result = evaluate_agent(agent, args)
        results[name] = result
    
    # Compare results
    print("\nComparison Results:")
    for name, result in results.items():
        avg_score = np.mean(result['scores'])
        max_score = np.max(result['scores'])
        avg_tile = np.mean(result['max_tiles'])
        max_tile = np.max(result['max_tiles'])
        
        print(f"{name} - Avg Score: {avg_score:.2f}, Max Score: {max_score}, " +
              f"Avg Max Tile: {avg_tile:.2f}, Highest Tile: {max_tile}")
    
    # Visualize if requested
    if args.visualize:
        visualizer = Visualizer()
        visualizer.compare_agents(results, save_path="visualizations/agent_comparison.png")

def main():
    """Main function"""
    args = parse_args()
    
    # Create environment
    env = Game2048Env()
    
    if args.agent == 'all' and args.mode == 'compare':
        compare_agents(args)
    else:
        # Create agent
        agent = create_agent(args.agent, env)
        
        # Run the selected mode
        if args.mode == 'train':
            train_agent(agent, args)
        elif args.mode == 'evaluate':
            evaluate_agent(agent, args)
        elif args.mode == 'play':
            play_game(agent, args)
        elif args.mode == 'compare':
            # Just compare with itself at different stages or configurations
            compare_agents(args)

if __name__ == '__main__':
    main()