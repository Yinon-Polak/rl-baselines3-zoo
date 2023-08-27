import sys
from rl_zoo3.train import train


# # sys.argv = ["python", "--algo", "a2c", "--env", "SnakeGameAIGym-v0", "-n", "10_000_000", "-optimize", "--n-trials", "2", "--track", "--wandb-project-name", "test-zoo-sb3", "--study-name", "test", "--storage", "sqlite:///example.db"]
# sys.argv = ["python", "--algo", "ppo", "--env", "SnakeGameAIGym-v0", "-n", "10_000_000", "-optimize", "--n-trials", "2", "--track", "--wandb-project-name", "test-123", "--study-name", "test-123", "--storage", "sqlite:///test-123.db"] #  "--track", "--wandb-project-name", "test-ppo-hp-opt", "--study-name", "test-ppo-hp-opt", "--storage", "sqlite:///test.db"

# test callbacks, max episode callback and max is_looping termination
project_name = "ppo-transformer-v-0.1"
sys.argv = ["python", "--algo", "ppo", "--env", "SnakeGameAIGym-v0", "--track", "--wandb-project-name", project_name]

train()