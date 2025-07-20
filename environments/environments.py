import random
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from numpy.typing import NDArray
from enum import Enum


class Action(Enum):
    """Actions available in the Tetris environment"""
    LEFT = 0
    RIGHT = 1
    DOWN = 2
    ROTATE = 3
    DROP = 4
    NOTHING = 5


class TetrisShape:
    """Represents a Tetris piece with its rotations"""
    
    SHAPES = {
        'I': [
            [[1, 1, 1, 1]],
            [[1], [1], [1], [1]]
        ],
        'O': [
            [[1, 1], [1, 1]]
        ],
        'T': [
            [[0, 1, 0], [1, 1, 1]],
            [[1, 0], [1, 1], [1, 0]],
            [[1, 1, 1], [0, 1, 0]],
            [[0, 1], [1, 1], [0, 1]]
        ],
        'S': [
            [[0, 1, 1], [1, 1, 0]],
            [[1, 0], [1, 1], [0, 1]]
        ],
        'Z': [
            [[1, 1, 0], [0, 1, 1]],
            [[0, 1], [1, 1], [1, 0]]
        ],
        'J': [
            [[1, 0, 0], [1, 1, 1]],
            [[1, 1], [1, 0], [1, 0]],
            [[1, 1, 1], [0, 0, 1]],
            [[0, 1], [0, 1], [1, 1]]
        ],
        'L': [
            [[0, 0, 1], [1, 1, 1]],
            [[1, 0], [1, 0], [1, 1]],
            [[1, 1, 1], [1, 0, 0]],
            [[1, 1], [0, 1], [0, 1]]
        ]
    }
    
    def __init__(self, shape_type: str):
        self.shape_type = shape_type
        self.rotations = self.SHAPES[shape_type]
        self.current_rotation = 0
        self.x = 0
        self.y = 0
    
    def get_shape(self) -> List[List[int]]:
        return self.rotations[self.current_rotation]
    
    def rotate(self) -> None:
        self.current_rotation = (self.current_rotation + 1) % len(self.rotations)
    
    def copy(self) -> 'TetrisShape':
        new_shape = TetrisShape(self.shape_type)
        new_shape.current_rotation = self.current_rotation
        new_shape.x = self.x
        new_shape.y = self.y
        return new_shape


class TetrisEnvironment:
    """Tetris environment for RL training and human play"""
    
    def __init__(self, width: int = 10, height: int = 20, seed: Optional[int] = None):
        self.width = width
        self.height = height
        self.board = np.zeros((height, width), dtype=int)
        self.current_piece = None
        self.next_piece = None
        self.score = 0
        self.lines_cleared = 0
        self.level = 1
        self.fall_time = 0
        self.fall_speed = 500  # milliseconds
        self.game_over = False
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.reset()
    
    def reset(self) -> NDArray[np.int32]:
        """Reset the game to initial state"""
        self.board = np.zeros((self.height, self.width), dtype=int)
        self.score = 0
        self.lines_cleared = 0
        self.level = 1
        self.fall_time = 0
        self.game_over = False
        self.current_piece = self._spawn_piece()
        self.next_piece = self._spawn_piece()
        return self.get_state()
    
    def _spawn_piece(self) -> TetrisShape:
        """Spawn a new random piece"""
        shape_type = random.choice(list(TetrisShape.SHAPES.keys()))
        piece = TetrisShape(shape_type)
        piece.x = self.width // 2 - len(piece.get_shape()[0]) // 2
        piece.y = 0
        return piece
    
    def _is_valid_position(self, piece: TetrisShape, dx: int = 0, dy: int = 0) -> bool:
        """Check if piece position is valid"""
        shape = piece.get_shape()
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    new_x = piece.x + x + dx
                    new_y = piece.y + y + dy
                    
                    if (new_x < 0 or new_x >= self.width or 
                        new_y >= self.height or 
                        (new_y >= 0 and self.board[new_y][new_x])):
                        return False
        return True
    
    def _place_piece(self, piece: TetrisShape) -> None:
        """Place piece on the board"""
        shape = piece.get_shape()
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    board_y = piece.y + y
                    board_x = piece.x + x
                    if 0 <= board_y < self.height and 0 <= board_x < self.width:
                        self.board[board_y][board_x] = 1
    
    def _clear_lines(self) -> int:
        """Clear completed lines and return number cleared"""
        lines_to_clear = []
        for y in range(self.height):
            if np.all(self.board[y]):
                lines_to_clear.append(y)
        
        for y in lines_to_clear:
            self.board = np.delete(self.board, y, axis=0)
            self.board = np.vstack([np.zeros((1, self.width)), self.board])
        
        lines_cleared = len(lines_to_clear)
        self.lines_cleared += lines_cleared
        
        # Update score based on lines cleared
        if lines_cleared > 0:
            points = [0, 40, 100, 300, 1200][lines_cleared]
            self.score += points * self.level
        
        # Update level
        self.level = self.lines_cleared // 10 + 1
        self.fall_speed = max(50, 500 - (self.level - 1) * 50)
        
        return lines_cleared
    
    def _get_drop_position(self, piece: TetrisShape) -> int:
        """Get the Y position where piece would land if dropped"""
        test_piece = piece.copy()
        while self._is_valid_position(test_piece, dy=1):
            test_piece.y += 1
        return test_piece.y
    
    def step(self, action: Action, dt: int = 0) -> Tuple[NDArray[np.int32], float, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        if self.game_over:
            return self.get_state(), 0, True, {'game_over': True}
        
        reward = 0
        
        # Handle action
        if action == Action.LEFT and self._is_valid_position(self.current_piece, dx=-1):
            self.current_piece.x -= 1
        elif action == Action.RIGHT and self._is_valid_position(self.current_piece, dx=1):
            self.current_piece.x += 1
        elif action == Action.DOWN and self._is_valid_position(self.current_piece, dy=1):
            self.current_piece.y += 1
            reward += 1  # Small reward for soft drop
        elif action == Action.ROTATE:
            # Try to rotate
            test_piece = self.current_piece.copy()
            test_piece.rotate()
            if self._is_valid_position(test_piece):
                self.current_piece.rotate()
        elif action == Action.DROP:
            # Hard drop
            old_y = self.current_piece.y
            self.current_piece.y = self._get_drop_position(self.current_piece)
            reward += 2 * (self.current_piece.y - old_y)  # Reward for hard drop
        
        # Handle automatic falling
        self.fall_time += dt
        if self.fall_time >= self.fall_speed:
            self.fall_time = 0
            if self._is_valid_position(self.current_piece, dy=1):
                self.current_piece.y += 1
            else:
                # Piece has landed
                self._place_piece(self.current_piece)
                lines_cleared = self._clear_lines()
                
                # Reward for clearing lines
                if lines_cleared > 0:
                    reward += [0, 40, 100, 300, 1200][lines_cleared] * self.level
                
                # Spawn next piece
                self.current_piece = self.next_piece
                self.next_piece = self._spawn_piece()
                
                # Check game over
                if not self._is_valid_position(self.current_piece):
                    self.game_over = True
                    reward -= 500  # Penalty for game over
        
        return self.get_state(), reward, self.game_over, {
            'score': self.score,
            'lines_cleared': self.lines_cleared,
            'level': self.level
        }
    
    def get_state(self) -> NDArray[np.int32]:
        """Get current state representation for RL"""
        # Create a copy of the board with current piece
        state_board = self.board.copy()
        
        if self.current_piece and not self.game_over:
            shape = self.current_piece.get_shape()
            for y, row in enumerate(shape):
                for x, cell in enumerate(row):
                    if cell:
                        board_y = self.current_piece.y + y
                        board_x = self.current_piece.x + x
                        if 0 <= board_y < self.height and 0 <= board_x < self.width:
                            state_board[board_y][board_x] = 2  # Different value for current piece
        
        return state_board
    
    def get_game_features(self) -> NDArray[np.float32]:
        """Get engineered features for RL training"""
        features = []
        
        # Board height features
        heights = []
        for x in range(self.width):
            height = 0
            for y in range(self.height):
                if self.board[y][x]:
                    height = self.height - y
                    break
            heights.append(height)
        
        features.extend(heights)
        
        # Height differences (bumpiness)
        height_diffs = [abs(heights[i] - heights[i+1]) for i in range(len(heights)-1)]
        features.extend(height_diffs)
        
        # Holes count
        holes = 0
        for x in range(self.width):
            block_found = False
            for y in range(self.height):
                if self.board[y][x]:
                    block_found = True
                elif block_found and not self.board[y][x]:
                    holes += 1
        features.append(holes)
        
        # Max height
        features.append(max(heights) if heights else 0)
        
        # Number of complete lines
        complete_lines = sum(1 for y in range(self.height) if np.all(self.board[y]))
        features.append(complete_lines)
        
        return np.array(features, dtype=np.float32)

if __name__ == "__main__":
    # Example of using the environment for RL
    print("Example RL interaction:")
    env = TetrisEnvironment()
    state = env.reset()
    
    for step in range(100):
        action = Action(random.randint(0, 5))  # Random action
        state, reward, done, info = env.step(action, dt=500)  # Simulate 500ms
        
        if done:
            print(f"Game over! Final score: {info['score']}")
            break
        
        if step % 20 == 0:
            print(f"Step {step}: Score={info['score']}, Lines={info['lines_cleared']}")