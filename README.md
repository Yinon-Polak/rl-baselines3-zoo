# Snake RL
Solving Snake game using RL training framework [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) and snake environment from [snake-ai-pytorch](https://github.com/patrickloeber/snake-ai-pytorch).


## Installation

### Minimal installation

From source:
```
pip install -e .
```

As a python package:
```
pip install rl_zoo3
```

Note: you can do `python -m rl_zoo3.train` from any folder and you have access to `rl_zoo3` command line interface, for instance, `rl_zoo3 train` is equivalent to `python train.py`

### Full installation (with extra envs and test dependencies)

```
apt-get install swig cmake ffmpeg
pip install -r requirements.txt
```

Please see [Stable Baselines3 documentation](https://stable-baselines3.readthedocs.io/en/master/) for alternatives to install stable baselines3.
