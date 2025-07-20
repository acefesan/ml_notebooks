import pygame
import random
from typing import Dict, Tuple
from environments import TetrisEnvironment, Action

class TetrisGame:
    """Pygame wrapper for visual Tetris gameplay"""
    
    def __init__(self, width: int = 10, height: int = 20) -> None:
        pygame.init()
        self.env = TetrisEnvironment(width, height)
        
        # Display settings
        self.cell_size = 30
        self.board_width = width * self.cell_size
        self.board_height = height * self.cell_size
        self.sidebar_width = 200
        self.screen_width = self.board_width + self.sidebar_width
        self.screen_height = self.board_height
        
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Tetris RL Environment")
        
        # Colors
        self.colors: Dict[int, Tuple[int, int, int]] = {
            0: (0, 0, 0),      # Empty
            1: (255, 255, 255), # Placed piece
            2: (255, 0, 0)      # Current piece
        }
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
    
    def run(self) -> None:
        """Run the game loop"""
        running = True
        
        while running:
            dt = self.clock.tick(60)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    action = Action.NOTHING
                    if event.key == pygame.K_LEFT:
                        action = Action.LEFT
                    elif event.key == pygame.K_RIGHT:
                        action = Action.RIGHT
                    elif event.key == pygame.K_DOWN:
                        action = Action.DOWN
                    elif event.key == pygame.K_UP:
                        action = Action.ROTATE
                    elif event.key == pygame.K_SPACE:
                        action = Action.DROP
                    elif event.key == pygame.K_r and self.env.game_over:
                        self.env.reset()
                    
                    if action != Action.NOTHING:
                        self.env.step(action, dt)
            
            # Auto-fall
            if not self.env.game_over:
                self.env.step(Action.NOTHING, dt)
            
            self._draw()
        
        pygame.quit()
    
    def _draw(self) -> None:
        """Draw the game state"""
        self.screen.fill((50, 50, 50))
        
        # Draw board
        state = self.env.get_state()
        for y in range(self.env.height):
            for x in range(self.env.width):
                color = self.colors[state[y][x]]
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                                 self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (128, 128, 128), rect, 1)
        
        # Draw sidebar
        sidebar_x = self.board_width + 10
        
        # Score
        score_text = self.font.render(f"Score: {self.env.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (sidebar_x, 50))
        
        # Lines
        lines_text = self.font.render(f"Lines: {self.env.lines_cleared}", True, (255, 255, 255))
        self.screen.blit(lines_text, (sidebar_x, 100))
        
        # Level
        level_text = self.font.render(f"Level: {self.env.level}", True, (255, 255, 255))
        self.screen.blit(level_text, (sidebar_x, 150))
        
        # Game over
        if self.env.game_over:
            game_over_text = self.font.render("GAME OVER", True, (255, 0, 0))
            restart_text = self.font.render("Press R to restart", True, (255, 255, 255))
            self.screen.blit(game_over_text, (sidebar_x, 250))
            self.screen.blit(restart_text, (sidebar_x, 300))
        
        pygame.display.flip()


if __name__ == "__main__":
    print("Starting Tetris game...")
    print("Controls: Arrow keys to move, Up to rotate, Space to drop, R to restart")
    
    game = TetrisGame()
    game.run()