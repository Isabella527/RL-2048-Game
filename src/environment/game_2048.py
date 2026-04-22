# src/environment/game_2048.py
import numpy as np
import random
import gym
from gym import spaces


class Game2048Env(gym.Env):
    """
    2048 Game Environment that follows gym interface.
    This environment simulates the game of 2048.
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, size=4):
        super(Game2048Env, self).__init__()
        
        # Define action and observation space
        # Actions: 0 = UP, 1 = RIGHT, 2 = DOWN, 3 = LEFT
        self.action_space = spaces.Discrete(4)
        
        # Observation space: 4x4 grid with tile values
        # Maximum tile value is 2^15 = 32768 (though very rare)
        self.observation_space = spaces.Box(
            low=0, high=2**15, shape=(size, size), dtype=np.int32
        )
        
        self.size = size
        self.board = None
        self.score = 0
        self.done = False
        
        # Initialize the game
        self.reset()
    
    def reset(self):
        """Reset the game state, return the initial observation"""
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        self.score = 0
        self.done = False
        
        # Place two initial tiles
        self._place_random_tile()
        self._place_random_tile()
        
        return self.board
    
    def step(self, action):
        """
        Take an action and return next state, reward, done, info
        
        Args:
            action: 0 = UP, 1 = RIGHT, 2 = DOWN, 3 = LEFT
            
        Returns:
            observation (np.array): The state of the game board
            reward (float): The reward for this step
            done (bool): Whether the game is over
            info (dict): Additional information
        """
        if self.done:
            return self.board, 0, True, {"score": self.score}
        
        old_board = self.board.copy()
        old_score = self.score
        
        # Move tiles
        valid_move = self._move(action)
        
        # Calculate reward
        reward = 0
        
        if valid_move:
            # Basic reward: increase in score
            reward += (self.score - old_score)
            
            # Add a small bonus for keeping empty tiles
            empty_tiles_before = np.count_nonzero(old_board == 0)
            empty_tiles_after = np.count_nonzero(self.board == 0)
            reward += (empty_tiles_after - empty_tiles_before) * 0.1
            
            # Place a new random tile (2 or 4)
            self._place_random_tile()
            
            # Check if game is over
            if not self._has_valid_moves():
                self.done = True
                # Penalty for ending the game
                reward -= 10
        else:
            # Penalty for invalid move
            reward -= 1
        
        return self.board, reward, self.done, {"score": self.score}
    
    def render(self, mode='human'):
        """Render the game board"""
        if mode == 'human' or mode == 'ansi':
            result = ""
            for row in self.board:
                result += " | ".join([str(int(cell)).rjust(4) if cell != 0 else "    " for cell in row])
                result += "\n" + "-" * (self.size * 6 - 1) + "\n"
            result += f"Score: {self.score}\n"
            
            if mode == 'human':
                print(result)
            return result
        else:
            raise NotImplementedError
    
    def _place_random_tile(self):
        """Place a random tile (2 or 4) on an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            row, col = random.choice(empty_cells)
            # 90% chance of placing 2, 10% chance of placing 4
            self.board[row, col] = 2 if random.random() < 0.9 else 4
    
    def _has_valid_moves(self):
        """Check if there are any valid moves left"""
        # If there are empty cells, there are valid moves
        if np.any(self.board == 0):
            return True
        
        # Check for adjacent cells with same values
        for row in range(self.size):
            for col in range(self.size):
                cell = self.board[row, col]
                # Check right
                if col < self.size - 1 and cell == self.board[row, col + 1]:
                    return True
                # Check down
                if row < self.size - 1 and cell == self.board[row + 1, col]:
                    return True
        
        return False
    
    def _move(self, action):
        """
        Move tiles in the specified direction
        
        Args:
            action: 0 = UP, 1 = RIGHT, 2 = DOWN, 3 = LEFT
            
        Returns:
            valid_move (bool): Whether a valid move was made
        """
        # Save old board for comparison
        old_board = self.board.copy()
        
        # Rotate board to handle all movements in the same way (left)
        self.board = np.rot90(self.board, action)
        
        # Process each row for left movement
        for row in range(self.size):
            # Compact the row (slide to the left)
            self.board[row] = self._compact_row(self.board[row])
            
            # Merge adjacent cells with same values
            self.board[row], score_increment = self._merge_row(self.board[row])
            self.score += score_increment
            
            # Compact again after merging
            self.board[row] = self._compact_row(self.board[row])
        
        # Rotate the board back to its original orientation
        self.board = np.rot90(self.board, (4 - action) % 4)
        
        # Check if the board has changed (valid move)
        return not np.array_equal(old_board, self.board)
    
    def _compact_row(self, row):
        """
        Compact a row by sliding all non-zero elements to the left
        """
        # Filter out zeros and build a new row
        non_zero = row[row != 0]
        # Pad with zeros at the end
        return np.pad(non_zero, (0, self.size - len(non_zero)), 'constant')
    
    def _merge_row(self, row):
        """
        Merge adjacent tiles with the same value
        
        Returns:
            merged_row (np.array): The merged row
            score_increase (int): Points gained from merging
        """
        score_increase = 0
        for i in range(self.size - 1):
            if row[i] != 0 and row[i] == row[i + 1]:
                # Merge tiles
                row[i] *= 2
                row[i + 1] = 0
                score_increase += row[i]
        return row, score_increase