import argparse
import time

import gymnasium as gym
import numpy as np
import retro
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

class StochasticFrameSkip(gym.Wrapper):
    # Same as training code
    def __init__(self, env, n, stickprob):
        super().__init__(env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()

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
            ob, rew, terminated, truncated, info = self.env.step(self.curac)
            totrew += rew
            if terminated or truncated:
                break
        return ob, totrew, terminated, truncated, info

def make_env(game, state):
    # Create environment with human rendering
    env = retro.make(
        game=game,
        state=state,
        render_mode='human',  # Changed to human-readable rendering
        use_restricted_actions=retro.Actions.FILTERED
    )
    
    # Apply same wrappers as training
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    env = WarpFrame(env)        # 84x84 grayscale
    env = ClipRewardEnv(env)    # Maintain reward clipping
    
    return env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="Airstriker-Genesis")
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--model-path", default='./checkpoints3/ppo_agent_75999744_steps', help="Path to trained model .zip file")
    args = parser.parse_args()

    # Create environment with same preprocessing as training
    env = make_env(args.game, args.state)
    
    # Apply vectorization and frame stacking to match training setup
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    # Load trained model
    model = PPO.load(args.model_path, env=env, device="cuda")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        
        # Render the game (automatically handled by render_mode='human')
        # Add small delay to make game visible
        time.sleep(0.02)
        
        # Reset on episode end
        if dones.any():
            print("Episode finished, resetting...")
            obs = env.reset()

if __name__ == "__main__":
    main()
