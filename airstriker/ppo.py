# """
# Train an agent using Proximal Policy Optimization from Stable Baselines 3
# """

import argparse

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps
# import Livesbonuswrapper as LBTracker

import retro


class StochasticFrameSkip(gym.Wrapper):
    def __init__(self, env, n, stickprob):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env, "supports_want_render")

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        terminated = False
        truncated = False
        totrew = 0
        for i in range(self.n):
            if self.curac is None:
                self.curac = ac
            elif i == 0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            elif i == 1:
                self.curac = ac
            if self.supports_want_render and i < self.n - 1:
                ob, rew, terminated, truncated, info = self.env.step(
                    self.curac,
                    want_render=False,
                )
            else:
                ob, rew, terminated, truncated, info = self.env.step(self.curac)
            totrew += rew
            if terminated or truncated:
                break
        return ob, totrew, terminated, truncated, info


def make_retro(*, game, state=None, max_episode_steps=4500, vis, **kwargs):
    if state is None:
        state = retro.State.DEFAULT
    if vis == 1:
        env = retro.make(game, state ,**kwargs)
    else:
        env = retro.make(game, state, render_mode='rgb_array', **kwargs)
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def wrap_deepmind_retro(env):
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="Airstriker-Genesis")
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--ckpt_dir", default="./checkpoints3", help="Directory to save checkpoints")
    parser.add_argument("--ckpt_freq", type=int, default=500_000, help="Timesteps between checkpoints")
    # parser.add_argument("--load_ckpt", type=str, default="") # Continue training a model
    parser.add_argument("--load_ckpt", type=str, default= "./checkpoints3/ppo_agent_85499744_steps.zip") # Continue training a model
    parser.add_argument('--num_env', type=int, default=32)
    parser.add_argument('--vis', type=int, default=0)
    args = parser.parse_args()

    def make_env():
        env = make_retro(game=args.game, state=args.state, vis=args.vis, scenario=args.scenario)
        env = wrap_deepmind_retro(env)
        # env = LBTracker.LivesBonusWrapper(env)
        # env = LBTracker.OptimizedRewardWrapper(env)
        return env

    n_envs = args.num_env
    venv = VecTransposeImage(VecFrameStack(SubprocVecEnv([make_env] * n_envs), n_stack=4))

    # Setup checkpoint callback: save every ckpt_freq timesteps
    checkpoint_callback = CheckpointCallback(
        save_freq=args.ckpt_freq // n_envs,  # Adjust for vectorized envs
        save_path=args.ckpt_dir,
        name_prefix="ppo_agent",
        save_replay_buffer=True,
        save_vecnormalize=True,
        verbose=2,
    )

    # Infinite training (very large number)
    total_timesteps = int(1e12)

    if args.load_ckpt is not None:
        print(f"Loading model from checkpoint: {args.load_ckpt}")
        model = PPO.load(args.load_ckpt, env=venv, device="cuda")
        print(model.device)  # Should print 'cuda:0' if on GPU
        # Optionally, you can reload replay buffer and vecnormalize if needed
    else:
        print("Initializing new model.")
        model = PPO(
            policy="CnnPolicy",
            env=venv,
            learning_rate=lambda f: f * 2.5e-4,
            n_steps=128,
            batch_size=32,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=0.01,
            verbose=1,
            device="cuda"
        )

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        log_interval=1,
        reset_num_timesteps=False
    )


if __name__ == "__main__":
    main()
