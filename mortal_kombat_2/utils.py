# utils.py

import gymnasium as gym
from gymnasium import Wrapper, spaces
import numpy as np

class Discretizer(gym.ActionWrapper):
    """
    Custom action discretizer for Mortal Kombat II
    Based on original retro implementation but self-contained
    """
    def __init__(self, env, buttons, combos):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.MultiBinary), \
            f"Expected MultiBinary, got {type(env.action_space)}"
        
        self._buttons = buttons
        self._combos = []  # Renamed for clarity
        for combo in combos:
            arr = np.array([False] * len(buttons))
            for button in combo:
                arr[self._buttons.index(button)] = True
            self._combos.append(arr)
        
        self.action_space = spaces.Discrete(len(self._combos))

    def action(self, act):
        return self._combos[act].copy()  # Now matches variable name

class LiuKangDiscretizer(Discretizer):
    def __init__(self, env):
        combos = [
            [], ['X'], ['A'], ['Z'], ['C'], ['Y'], ['START'],
            ['UP'], ['DOWN'], ['LEFT'], ['RIGHT'],
            ['RIGHT', 'RIGHT', 'X'],       # High Fireball
            ['RIGHT', 'RIGHT', 'A'],       # Low Fireball
            ['UP', 'RIGHT', 'RIGHT', 'X'], # Air Fireball
            ['RIGHT', 'RIGHT', 'Z'],       # Flying Kick
            ['C'],                         # Bicycle Kick
            ['UP', 'Z'],                   # Jump High Kick
            ['UP', 'X'],                   # Jump High Punch
            ['DOWN', 'Z'],                 # Crouch High Kick
            ['DOWN', 'A'],                 # Crouch Low Punch
            ['DOWN', 'C'],                 # Crouch Low Kick
            ['LEFT', 'A'],                 # Back + Low Punch
            ['LEFT', 'Z'],                 # Back + High Kick
            ['RIGHT', 'A'],                # Forward + Low Punch
            ['RIGHT', 'Z']                 # Forward + High Kick
        ]
        
        self.combo_names = [
            "No Action", "High Punch", "Low Punch", "High Kick", "Low Kick", "Block", "Start",
            "Up", "Down", "Left", "Right",
            "High Fireball", "Low Fireball", "Air Fireball", "Flying Kick", "Bicycle Kick",
            "Jump High Kick", "Jump High Punch", "Crouch High Kick", "Crouch Low Punch",
            "Crouch Low Kick", "Back+Low Punch", "Back+High Kick", "Forward+Low Punch", "Forward+High Kick"
        ]
        super().__init__(env=env, buttons=env.unwrapped.buttons, combos=combos)

    def get_combo_name(self, action):
        if 0 <= action < len(self.combo_names):
            return self.combo_names[action]
        return "Unknown"

class SubzeroDiscretizer(Discretizer):
    """
    Custom action space for Sub-Zero's moves in MKII
    Includes special moves and fatalities
    """
    def __init__(self, env):
        super().__init__(
            env=env,
            buttons=env.unwrapped.buttons,
            # combos=[
            #     # Basic moves
            #     [],
            #     ['X'],  # High Punch
            #     ['A'],  # Low Punch
            #     ['Z'],  # Block
            #     ['C'],  # High Kick
            #     ['Y'],  # Low Kick
            #     ['START'],
                
            #     # Special moves
            #     ['DOWN', 'RIGHT', 'X'],  # Freeze (D,F+LP)
            #     ['LEFT', 'RIGHT', 'C'],  # Ground Freeze (B,F+LK)
            #     ['LEFT', 'DOWN', 'A'],   # Slide (B,D+LP)
                
            #     # Kombos
            #     ['UP', 'Z'],             # Jump Kick
            #     ['RIGHT', 'RIGHT', 'X'], # Dash Freeze
            #     ['LEFT', 'DOWN', 'RIGHT', 'C'], # Ground Freeze Combo
                
            #     # Fatalities
            #     ['DOWN', 'DOWN', 'LEFT', 'RIGHT', 'Y'],  # Spine Rip
            #     ['DOWN', 'FORWARD', 'DOWN', 'FORWARD', 'B']  # Ice Statue
            # ]
            combos=[[], ['X'], ['A'], ['Z'], ['C'], ['Y'], ['START'], 
                    ['UP'], ['DOWN'], ['LEFT'], ['RIGHT'], 
                
                ['LEFT', 'UP'], ['LEFT', 'DOWN'], ['RIGHT', 'UP'], ['RIGHT', 'DOWN'],
                ['UP', 'Z'], ['LEFT', 'UP', 'Z'], ['RIGHT', 'UP', 'Z'],
                ['UP', 'X'], ['LEFT', 'UP', 'X'], ['RIGHT', 'UP', 'X'],
                ['LEFT', 'X'], ['RIGHT', 'X'], ['LEFT', 'DOWN', 'X'], ['RIGHT', 'DOWN', 'X'],
                ['LEFT', 'A'], ['RIGHT', 'A'], ['LEFT', 'UP', 'A'], ['RIGHT', 'UP', 'A'], ['LEFT', 'DOWN', 'A'], ['RIGHT', 'DOWN', 'A'],
                ['DOWN', 'X'],
                ['DOWN', 'A'], 
                ['DOWN', 'C'], ['LEFT', 'DOWN', 'C'], ['RIGHT', 'DOWN', 'C'],
                ['DOWN', 'Y'], ['DOWN', 'LEFT', 'Y'], ['DOWN', 'RIGHT', 'Y'],
                ['LEFT', 'C', 'Z'], ['RIGHT', 'C', 'Z']
                ]
        )


    def get_move_name(self, action):
        move_names = {
            0: "Block",
            1: "High Punch",
            2: "Low Punch",
            3: "Block",
            4: "High Kick",
            5: "Low Kick",
            6: "Start",
            7: "Freeze",
            8: "Ground Freeze",
            9: "Slide",
            10: "Jump Kick",
            11: "Dash Freeze",
            12: "Ground Freeze Combo",
            13: "Spine Rip Fatality",
            14: "Ice Statue Fatality",
            15: "Unknown",  # Add fallback
            16: "Unknown"
        }
        return move_names.get(action, "Unknown")

# Add this to your utils.py or a new reward_wrappers.py file
class MK2RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_health = 100
        self.prev_enemy_health = 100
        self.rounds_won = 0
        self.rounds_lost = 0
        self.episode_steps = 0

        # Define health thresholds (descending order)
        self.health_triggers = [100, 80, 60, 50, 40, 30, 20, 10, 0]
        self.enemy_health_triggers = [100, 80, 60, 50, 40, 30, 20, 10, 0]

        # Track which triggers have fired per episode
        self.fired_health_triggers = set()
        self.fired_enemy_health_triggers = set()

        # Reward/penalty for each threshold (index matches trigger)
        self.health_trigger_penalties = [5, 10, 25, 50, 75, 100, 150, 300, 500]
        self.enemy_health_trigger_rewards = [5, 10, 25, 50, 75, 100, 150, 300, 500]

        self.episode_rewards = []

    def reset(self, **kwargs):
        # if self.episode_rewards and np.random.rand() < 1e-2:
        #     print(f"Episode ended. Total reward: {sum(self.episode_rewards):.2f}")
        #     # print(f"Steps: {self.episode_steps}")
        #     # print(f"Rounds won: {self.rounds_won}, Rounds lost: {self.rounds_lost}")
        #     print(f"Health change: {self.prev_health - 120:.2f}, "
        #           f"Enemy health change: {self.prev_enemy_health - 120:.2f}")
        
        self.episode_rewards = []
        self.episode_steps = 0
        self.prev_health = 120
        self.prev_enemy_health = 120
        self.fired_health_triggers = set()
        self.fired_enemy_health_triggers = set()
        
        obs, info = self.env.reset(**kwargs)
        return obs, info 
        

    def step(self, action):
        self.episode_steps += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Get current game state
        current_health = info.get('health', 100)
        current_enemy_health = info.get('enemy_health', 100)
        
        # Calculate deltas
        health_delta = current_health - self.prev_health
        enemy_health_delta = self.prev_enemy_health - current_enemy_health
        
        # Base rewards
        reward = 0
        reward += enemy_health_delta * 5.0
        reward += health_delta * -3.75
        
        for idx, threshold in enumerate(self.health_triggers):
            if current_health <= threshold and threshold not in self.fired_health_triggers:
                reward -= 1.0 * self.health_trigger_penalties[idx]
                self.fired_health_triggers.add(threshold)
                # print(f"Player health dropped below {threshold}: penalty {self.health_trigger_penalties[idx]} (triggered once)")

        # Stepwise health triggers (enemy)
        for idx, threshold in enumerate(self.enemy_health_triggers):
            if current_enemy_health <= threshold and threshold not in self.fired_enemy_health_triggers:
                reward += 2.0 * self.enemy_health_trigger_rewards[idx]
                self.fired_enemy_health_triggers.add(threshold)
                # if current_enemy_health <= 60:
                    # print(f"Enemy health dropped below {threshold}: reward {self.enemy_health_trigger_rewards[idx]} (triggered once)")
        
        # Update previous values
        self.prev_health = current_health
        self.prev_enemy_health = current_enemy_health

        # Logging few steps
        # if np.random.rand() < 1e-6:
            # print(
                #   f"[Step {self.episode_steps}] "
                #   f"Damage: {enemy_health_delta*2.0:.2f}, "
                #   f"Taken: {health_delta*-1.5:.2f}, "
                #   f"Total: {reward:.2f}")
        
        self.episode_rewards.append(reward)

        return obs, reward, terminated, truncated, info
