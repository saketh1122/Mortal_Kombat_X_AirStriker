import numpy as np
import gymnasium as gym
from gymnasium import Wrapper  # Changed from gym to gymnasium
import numpy as np

class LivesBonusWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_lives = None
        self.prev_score = None
        self.max_score = None

    def reset(self, **kwargs):
        self.prev_lives = None
        self.prev_score = None
        self.max_score = None
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        print(info)
        
        # Get current lives from info dict
        current_lives = info.get('lives', None)
        current_score = info.get('score', None)
        # Check if lives increased from previous state
        if self.max_score is None:
            self.max_score = current_score
            print(f"New max score: {self.max_score}")

        if self.prev_lives is not None and current_lives is not None:
            if current_lives > self.prev_lives:
                # Add large reward bonus
                reward += 100
                print(f"Lives increased! Old: {self.prev_lives}, New: {current_lives} - Bonus awarded!")
            elif current_lives < self.prev_lives:
                #Add large negative reward bonus
                reward -= 10
                print(f"Lives decreased! Old: {self.prev_lives}, New: {current_lives} - Penalty applied!")
        # Check if score increased from previous state
        if self.prev_score is not None and current_score is not None:
            if current_score > self.prev_score:
                # Add small reward bonus
                reward += (current_score - self.prev_score)
                print(f"Score increased! Old: {self.prev_score}, New: {current_score} - Small bonus awarded!")
            elif current_score < self.prev_score:
                # Add small negative reward bonus
                reward -= 5
                print(f"Score decreased! Old: {self.prev_score}, New: {current_score} - Small penalty applied!")
        
        # if current_score > self.max_score:
        #     # Add large reward for new max score
        #     reward += 50
        #     print(f"New max score! Old: {self.max_score}, New: {current_score} - Large bonus awarded!")
        
        # gameover = info.get('gameover', False)
        # if gameover:
        #     # Add large negative reward for game over
        #     reward -= 20
        #     print("Game Over! Large penalty applied!")
        print(f"final reward: {reward}")
        # Store current lives for next comparison
        self.prev_lives = current_lives
        self.prev_score = current_score
        
        return obs, reward, terminated, truncated, info
    


class OptimizedRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.episode_count = 0
        self.prev_score = 0
        self.prev_lives = 3
        self.current_max_score = 0
        self.time_penalty_factor = 0.01
        self.consecutive_score_steps = 0
        # Logging variables
        self.initial_score = 0
        self.final_score = 0
        self.total_score_delta = 0
        self.episode_steps = 0
        self.episode_reward = 0

    def reset(self, **kwargs):
        # Log previous episode summary
        if self.episode_steps > 0:
            self._log_episode_summary()
        self.episode_count += 1
        obs, info = self.env.reset(**kwargs)
        self.prev_score = info.get('score', 0)
        self.prev_lives = info.get('lives', 3)
        self.current_max_score = self.prev_score
        self.consecutive_score_steps = 0
        self.initial_score = self.prev_score
        self.final_score = 0
        self.total_score_delta = 0
        self.episode_steps = 0
        self.episode_reward = 0
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_steps += 1

        # Extract game state
        current_score = info.get('score', 0)
        current_lives = info.get('lives', 3)
        gameover = info.get('gameover', 0)
        self.final_score = current_score

        # Track score changes
        score_delta = current_score - self.prev_score
        self.total_score_delta += score_delta

        # Reward calculation (same as before)
        score_reward = life_reward = survival_reward = time_penalty = gameover_penalty = combo_bonus = 0

        if score_delta > 0:
            score_reward = score_delta * (1 + (self.current_max_score / 10000))
            self.consecutive_score_steps += 1
        else:
            self.consecutive_score_steps = 0

        if current_score > self.current_max_score:
            self.current_max_score = current_score

        if current_lives < self.prev_lives:
            life_reward = -25 * (current_lives + 1)
        elif current_lives > self.prev_lives:
            life_reward = 50 * (1 + (self.current_max_score / 5000))

        survival_reward = 0.1
        time_penalty = -self.time_penalty_factor
        gameover_penalty = -50 + (self.current_max_score * 0.1) if gameover else 0
        combo_bonus = min(self.consecutive_score_steps * 0.5, 10)

        total_reward = (
            score_reward +
            life_reward +
            survival_reward +
            time_penalty +
            gameover_penalty +
            combo_bonus
        )

        self.episode_reward += total_reward

        # Store current state
        self.prev_score = current_score
        self.prev_lives = current_lives

        # Log summary if episode ended
        if terminated or truncated:
            self._log_episode_summary()

        return obs, total_reward, terminated, truncated, info

    def _log_episode_summary(self):
        header = f"\n🔍 Episode {self.episode_count} Breakdown 🔍"
        divider = "═" * 60
        metrics = [
            ("Initial Score", self.initial_score),
            ("Final Score", self.final_score),
            ("Score Change", f"{self.total_score_delta:+}"),
            ("Max Score", self.current_max_score),
            ("Final Lives", self.prev_lives),
            ("Lives Lost", 3 - self.prev_lives),
            ("Total Reward", f"{self.episode_reward:+.2f}"),
            ("Steps", self.episode_steps)
        ]

        print(header)
        print(divider)
        for label, value in metrics:
            print(f"│ {label:<20} │ {str(value):>30} │")
        print(divider)
        if self.total_score_delta < 0:
            print("⚠️  Score decreased! Check game variables")
        if self.current_max_score == 0:
            print("⚠️  Agent failed to score points")
        print("\n")

    def close(self):
        if self.episode_steps > 0:
            self._log_episode_summary()
        super().close()
# class OptimizedRewardWrapper(gym.Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         self.episode_count = 0
#         self.prev_score = 0
#         self.prev_lives = 3
#         self.current_max_score = 0
#         self.time_penalty_factor = 0.01
#         self.consecutive_score_steps = 0
        
#         # Enhanced logging variables
#         self.initial_score = 0
#         self.final_score = 0
#         self.total_score_delta = 0
#         self.verbose = True  # Toggle for step-by-step logging

#     def reset(self, **kwargs):
#         self._log_episode_summary()
#         self.episode_count += 1
        
#         # Reset environment and get initial info
#         obs, info = self.env.reset(**kwargs)
        
#         # Initialize tracking variables
#         self.prev_score = info.get('score', 0)
#         self.prev_lives = info.get('lives', 3)
#         self.current_max_score = self.prev_score
#         self.consecutive_score_steps = 0
#         self.initial_score = self.prev_score
#         self.final_score = 0
#         self.total_score_delta = 0
        
#         return obs

#     def step(self, action):
#         obs, reward, terminated, truncated, info = self.env.step(action)
        
#         # Extract game state
#         current_score = info.get('score', 0)
#         current_lives = info.get('lives', 3)
#         gameover = info.get('gameover', 0)
#         self.final_score = current_score

#         # Track score changes
#         score_delta = current_score - self.prev_score
#         self.total_score_delta += score_delta

#         # Debug logging
#         if self.verbose and score_delta != 0:
#             print(f"\n[Step] Score: {self.prev_score} → {current_score} (Δ{score_delta})")
#         if self.verbose and current_lives != self.prev_lives:
#             print(f"[Step] Lives: {self.prev_lives} → {current_lives}")

#         # Original reward calculation
#         score_reward = life_reward = survival_reward = time_penalty = gameover_penalty = combo_bonus = 0

#         if score_delta > 0:
#             score_reward = score_delta * (1 + (self.current_max_score / 10000))
#             self.consecutive_score_steps += 1
#         else:
#             self.consecutive_score_steps = 0
            
#         if current_score > self.current_max_score:
#             self.current_max_score = current_score

#         if current_lives < self.prev_lives:
#             life_reward = -25 * (current_lives + 1)
#         elif current_lives > self.prev_lives:
#             life_reward = 50 * (1 + (self.current_max_score / 5000))

#         survival_reward = 0.1
#         time_penalty = -self.time_penalty_factor
#         gameover_penalty = -50 + (self.current_max_score * 0.1) if gameover else 0
#         combo_bonus = min(self.consecutive_score_steps * 0.5, 10)

#         total_reward = (
#             score_reward +
#             life_reward +
#             survival_reward +
#             time_penalty +
#             gameover_penalty +
#             combo_bonus
#         )

#         # Store current state
#         self.prev_score = current_score
#         self.prev_lives = current_lives

#         if terminated or truncated:
#             self._log_episode_summary()

#         return obs, total_reward, terminated, truncated, info

#     def _log_episode_summary(self):
#         if self.episode_count == 0:
#             return

#         header = f"\n🔍 Episode {self.episode_count} Breakdown 🔍"
#         divider = "═" * 60
#         metrics = [
#             ("Initial Score", self.initial_score),
#             ("Final Score", self.final_score),
#             ("Score Change", f"{self.total_score_delta:+}"),
#             ("Max Score", self.current_max_score),
#             ("Final Lives", self.prev_lives),
#             ("Lives Lost", 3 - self.prev_lives),
#             ("Total Reward", f"{self._get_episode_reward():+.2f}")
#         ]

#         print(header)
#         print(divider)
#         for label, value in metrics:
#             print(f"│ {label:<20} │ {str(value):>30} │")
#         print(divider)
        
#         # Add diagnostic warnings
#         if self.total_score_delta < 0:
#             print("⚠️  Score decreased! Check game variables")
#         if self.current_max_score == 0:
#             print("⚠️  Agent failed to score points")
#         print("\n")

#     def _get_episode_reward(self):
#         # Implement actual reward summation if needed
#         return 0.0  # Placeholder for actual calculation

#     def close(self):
#         self._log_episode_summary()
#         super().close()
