import warnings
from typing import Optional, List, Union, Dict, Type, Any

import gym
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from torch import nn
import torch as th


class NatureSmallCNN(BaseFeaturesExtractor):
    """
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
            self,
            observation_space: gym.spaces.Box,
            features_dim: int = 512,
            initial_layer_out_channels: int = 16,
            deep_layers_out_channels: int = 32,
            kernel_size_1: int = 4,
            kernel_size_2: int = 3,
            kernel_size_3: int = 2,
            padding_1: int = 1,
            padding_2: int = 0,
            padding_3: int = 0,
            stride_1: int = 1,
            stride_2: int = 1,
            stride_3: int = 1,
            activation_func: nn.Module = nn.ReLU
    ):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
        )

        print(
            f"initial_layer_out_channels ={initial_layer_out_channels} ; "
            f"deep_layers_out_channels = {deep_layers_out_channels} ; "
            f"kernel_size_1={kernel_size_1} ; "
            f"kernel_size_2={kernel_size_2} ; "
            f"kernel_size_3={kernel_size_3} ; "
            f"padding_1={padding_1} ; "
            f"padding_2={padding_2} ; "
            f"padding_3={padding_3} ; "
            f"stride_1={stride_1} ; "
            f"stride_2={stride_2} ; "
            f"stride_3={stride_3} ; "
            f"activation func: {activation_func.__name__} ; "
        )

        n_input_channels = observation_space.shape[0]
        modules = [
            nn.Conv2d(n_input_channels, initial_layer_out_channels, kernel_size=kernel_size_1, stride=stride_1, padding=padding_1),
            activation_func(),
            nn.Conv2d(initial_layer_out_channels, deep_layers_out_channels, kernel_size=kernel_size_2, stride=stride_2, padding=padding_2),
            activation_func(),
        ]
        if kernel_size_3:
            modules.extend(
                [
                    nn.Conv2d(deep_layers_out_channels, deep_layers_out_channels, kernel_size=kernel_size_3, stride=stride_3, padding=padding_3),
                    activation_func(),
                ]
            )
        modules.append(nn.Flatten())
        self.cnn = nn.Sequential(*modules)

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), activation_func())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class ActorCriticSmallCnnPolicy(ActorCriticPolicy):
    """
    CNN policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = [],
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        share_features_extractor: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureSmallCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = {},
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        warnings.warn("setting constant net_arch=[]")
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            [],
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            # sde_net_arch,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
        print("net_arch:", net_arch)




