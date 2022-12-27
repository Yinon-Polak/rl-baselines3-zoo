import sys
from rl_zoo3.train import train

# sys.argv = ["python", "--algo", "a2c", "--env", "SnakeGameAIGym-v0", "-n", "10_000_000", "-optimize", "--n-trials", "2", "--track", "--wandb-project-name", "test-zoo-sb3", "--study-name", "test", "--storage", "sqlite:///example.db"]
sys.argv = ["python", "--algo", "ppo", "--env", "SnakeGameAIGym-v0", "-n", "10_000_000", "-optimize", "--n-trials", "2", "--track", "--wandb-project-name", "test-ppo-hp-opt", "--study-name", "test-ppo-hp-opt", "--storage", "sqlite:///test.db"]

train()