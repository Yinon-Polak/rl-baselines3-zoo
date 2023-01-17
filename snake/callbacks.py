import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class StopTrainingOnMaxIsLoopingCount(BaseCallback):
    """
    Stop the training once a maximum number of looping is passed.
    For multiple environments presumes that, the desired behavior is that the max_is_looping_count = ``max_is_looping_count * n_envs``
    :param max_is_looping_ratio_threshold: Maximum number of episodes to stop training.
    :param min_episodes_count_before_termination: Minimum number of episodes before exiting.
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about when training ended by
        reaching ``max_episodes``
    """

    def __init__(self, max_is_looping_ratio_threshold: int, min_episodes_count_before_termination: int, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.max_is_looping_ratio_threshold = max_is_looping_ratio_threshold
        self.min_episodes_count_before_termination = min_episodes_count_before_termination
        self.n_is_looping = 0
        self.n_episodes = 1
        self.step_count = 0

    def _on_step(self) -> bool:
        self.step_count += 1
        # Check that the `dones`, `infos` local variables are defined
        assert "dones" in self.locals, "`dones` variable is not defined, please check your code next to `callback.on_step()`"
        assert "infos" in self.locals, "`infos` variable is not defined, please check your code next to `callback.on_step()`"
        self.n_episodes += np.sum(self.locals["dones"]).item()
        self.n_is_looping += [done and info["is_looping"] for info, done in zip(self.locals["infos"], self.locals["dones"])].count(True)
        is_looping_ratio = self.n_is_looping / self.n_episodes
        continue_training = not (self.n_episodes > self.min_episodes_count_before_termination and is_looping_ratio > self.max_is_looping_ratio_threshold)

        if self.verbose >= 1 and self.step_count % 1_000 == 0:
            print(
                f"StopTrainingOnMaxIsLoopingCount:"
                f" n_episodes: {self.n_episodes} ; "
                f"n_is_looping: {self.n_is_looping} ; "
                f"is_looping_ratio: {is_looping_ratio} ; "
            )

        if self.verbose >= 1 and not continue_training:
            mean_is_looping_per_env = self.n_is_looping / self.training_env.num_envs
            mean_is_looping_str = (
                f"with an average of {mean_is_looping_per_env:.2f} is_looping per env" if self.training_env.num_envs > 1 else ""
            )

            print(
                f"Stopping training with a total of {self.num_timesteps} steps because the "
                f"{self.locals.get('tb_log_name')} model reached max_is_looping_ratio_threshold={self.max_is_looping_ratio_threshold}, "
                f"by playing for {self.n_episodes} episodes "
                f"{mean_is_looping_str}"
            )

        return continue_training


class StopTrainingOnMaxEpisodes(BaseCallback):
    """
    Stop the training once a maximum number of episodes are played.
    For multiple environments presumes that, the desired behavior is that the agent trains on each env for ``max_episodes``
    and in total for ``max_episodes * n_envs`` episodes.
    :param max_episodes: Maximum number of episodes to stop training.
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about when training ended by
        reaching ``max_episodes``
    """

    def __init__(self, max_episodes: int, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.max_episodes = max_episodes
        self._total_max_episodes = max_episodes
        self.n_episodes = 0

    def _init_callback(self) -> None:
        # At start set total max according to number of envirnments
        self._total_max_episodes = self.max_episodes * self.training_env.num_envs

    def _on_step(self) -> bool:
        # Check that the `dones` local variable is defined
        assert "dones" in self.locals, "`dones` variable is not defined, please check your code next to `callback.on_step()`"
        self.n_episodes += np.sum(self.locals["dones"]).item()

        continue_training = self.n_episodes < self._total_max_episodes

        if self.verbose >= 1 and not continue_training:
            mean_episodes_per_env = self.n_episodes / self.training_env.num_envs
            mean_ep_str = (
                f"with an average of {mean_episodes_per_env:.2f} episodes per env" if self.training_env.num_envs > 1 else ""
            )

            print(
                f"Stopping training with a total of {self.num_timesteps} steps because the "
                f"{self.locals.get('tb_log_name')} model reached max_episodes={self.max_episodes}, "
                f"by playing for {self.n_episodes} episodes "
                f"{mean_ep_str}"
            )
        return continue_training


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.n_episodes = 0
        self.sum_scores = 0

    def _on_step(self) -> bool:
        # Check that the `dones` local variable is defined
        assert "dones" in self.locals, "`dones` variable is not defined, please check your code next to `callback.on_step()`"
        n_step_dones = np.sum(self.locals["dones"]).item()
        self.n_episodes += n_step_dones
        self.sum_scores += sum([info["score"] for info, done in zip(self.locals["infos"], self.locals["dones"]) if done])

        if n_step_dones > 0:
            self.logger.record("train/score", self.sum_scores / self.n_episodes)

        return True
