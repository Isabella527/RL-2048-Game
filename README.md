## 🧩 RL 2048: Deep Q-Learning Agent

Reinforcement Learning agent trained to master the game of 2048 using Deep Q-Networks (DQN).

## 🚀 Overview

This project explores the use of Deep Reinforcement Learning to solve the game 2048, a stochastic, high-dimensional puzzle environment.

We model the game as a Markov Decision Process (MDP) and train an agent using Deep Q-Learning (DQN) to learn optimal strategies through self-play.

### 🎮 Gameplay GIF


### 🎯 Objectives
Build an RL agent capable of playing 2048 effectively
Design meaningful state representations
Develop a reward function that encourages strategy
Evaluate performance against baseline methods
### 🕹️ Game Environment
4×4 grid
Actions: Left, Right, Up, Down
Random tile spawning (2 or 4)
Terminal state when no moves remain
### 🧠 Model Architecture
Fully connected neural network
Input: 256-dimensional vector (one-hot encoded board)
Hidden layers: 256 → 128
Output: 4 Q-values (one per action)
### 🏆 Reward Design

The agent is trained using a composite reward function:

✅ Tile merge rewards
✅ Empty space bonus
✅ High-value tile bonus
✅ Monotonicity & smoothness
❌ Invalid move penalty
❌ Game over penalty
### ⚙️ Training Details
| Parameter |	Value |
| --- | --- |
| Learning Rate |	1e-4 |
| Gamma	 |   0.99 |
| Batch Size | 	64 |
| Replay Buffer |	100,000 |
| Episodes	|  500 |
| Epsilon	|  1.0 → 0.1 |

### 📊 Results

After training for 500 episodes:

Average Score: ~10,000
Maximum Score: >25,000
Average Max Tile: 256–512

Performance Breakdown
| Max Tile |	Percentage |
| --- | --- |
| 64 |	10% |
| 128 |	38% |
| 256 |	46% |
| 512 | 5% |
Comparison
| Method |	Avg Score |
| --- | --- |
| Random |	~900 |
| Greedy |	~3000 |
| DQN	| ~10,000 |

# put picture here

### 🧩 Learned Strategies

The agent independently learns:

Corner anchoring
Tile chain building
Space management
Directional consistency
### 📁 Project Structure
src/        → core RL implementation  
models/     → trained models  
docs/       → report + slides  
▶️ How to Run
pip install -r requirements.txt

### Train
python src/main.py --agent dqn --mode train --episodes 500 --visualize

### Evaluate
python src/main.py --agent dqn --mode evaluate --load-model models/dqn_final --visualize

### Watch gameplay
python src/main.py --agent dqn --mode play --load-model models/dqn_final --render
### 🔮 Future Work
Double DQN
Dueling Networks
Prioritized Experience Replay
CNN-based state representation
### 👥 Authors
Isabella Opoku-Ware
Ke-Huan Yeh
Peter Dordunu