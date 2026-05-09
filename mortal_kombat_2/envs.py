import os
import numpy as np
from stable_baselines3.common.atari_wrappers import WarpFrame, ClipRewardEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import retro
import game_wrappers_mgr as games
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecTransposeImage
from utils import SubzeroDiscretizer, MK2RewardWrapper  # Import custom discretizer
# In envs.py (modify make_env function)

from utils import LiuKangDiscretizer, MK2RewardWrapper
# MenuSkipper  # Import custom discretizer

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
        obs, info = self.env.reset(**kwargs)
        return obs, info
        # return self.env.reset(**kwargs)

    def step(self, ac):
        done = False
        totrew = 0
        for i in range(self.n):
            # First step after reset, use action
            if self.curac is None:
                self.curac = ac
            # First substep, delay with probability=stickprob
            elif i==0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            # Second substep, new action definitely kicks in
            elif i==1:
                self.curac = ac
            if self.supports_want_render and i<self.n-1:
                ob, rew, terminated, truncated, info = self.env.step(self.curac, want_render=False)
            else:
                ob, rew, terminated, truncated, info = self.env.step(self.curac)
            totrew += rew
            if terminated: break
        return ob, totrew, terminated, truncated, info

    def seed(self, s):
        self.rng.seed(s)

def make_retro(*, game, state=None, num_players, max_episode_steps=4500, **kwargs):
    import retro  # pylint: disable=import-outside-toplevel,reimported
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(game, state, **kwargs, players=num_players, render_mode="rgb_array")
    #env = NHL94Discretizer(env)
    #if max_episode_steps is not None:
    #    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def init_env(output_path, num_env, state, num_players, args, use_sticky_action=True, use_display=False, use_frame_skip=True):
    #if wrapper_kwargs is None:
    wrapper_kwargs = {}
    #wrapper_kwargs['scenario'] = 'test'

    seed = 0
    start_index = 0
    start_method=None
    allow_early_resets=True

    def make_env(rank):
        def _thunk():
            games.wrappers.init(args)

            env = make_retro(game=args.env,
                             use_restricted_actions=retro.Actions.FILTERED, 
                             state=state, 
                             num_players=num_players)

            env.action_space.seed(seed + rank)
            env = LiuKangDiscretizer(env)
            # env = SubzeroDiscretizer(env)
            env = MK2RewardWrapper(env)

            if args.nn == 'MlpPolicy':
                env = games.wrappers.obs_env(env, args, num_players, args.rf)
                #if args.rf != '':
                #    env.set_reward_function(args.rf)

            env = Monitor(env, output_path and os.path.join(output_path, str(rank)), allow_early_resets=allow_early_resets)

            # TOFIX
            #if use_display:
            #    env = GameDisplayEnv(env, args, 17, 'CNN', None)
            if use_frame_skip:
                if use_sticky_action:
                    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
                else:
                    env = StochasticFrameSkip(env, n=4, stickprob=-1)

            if args.nn != 'MlpPolicy':
                env = WarpFrame(env)

            env = ClipRewardEnv(env)

            return env
        return _thunk

    env = SubprocVecEnv([make_env(i + start_index) for i in range(num_env)], start_method=start_method)

    env.seed(seed)

    if args.nn != 'MlpPolicy':
        env = VecFrameStack(env, n_stack=4)
        #env = VecTransposeImage(env)

    return env

def get_button_names(args):
    env = retro.make(
        game=args.env, 
        state=args.state, 
        use_restricted_actions=retro.Actions.FILTERED, 
        players=args.num_players
    )
    buttons = env.buttons
    env.close()  # Critical: Close the temporary environment
    return buttons

def init_play_env(args, num_players, is_pvp_display=False, need_display=True, use_frame_skip=True):
    button_names = get_button_names(args)
    games.wrappers.init(args)

    # 1. Create base environment
    env = retro.make(
        game=args.env,
        state=args.state,
        use_restricted_actions=retro.Actions.FILTERED,
        players=num_players,
        obs_type=retro.Observations.IMAGE,
        render_mode="human" if need_display else "rgb_array"
    )
    
    # 2. Apply core wrappers
    # env = MenuSkipper(env)
    env = LiuKangDiscretizer(env)
    # env = SubzeroDiscretizer(env)
    env = MK2RewardWrapper(env)
    
    # # 3. Apply display wrapper BEFORE vectorization
    # if need_display:
    #     if is_pvp_display:
    #         env = games.wrappers.pvp_display_env(env, args, args.model1_desc, args.model2_desc, None, None, button_names)
    #     else:
    #         env = games.wrappers.sp_display_env(env, args, 0, args.model1_desc, button_names)
    
    # # 4. Apply observation processing
    # if args.nn != 'MlpPolicy':
    #     env = WarpFrame(env)
    #     env = ClipRewardEnv(env)
    
    # # 5. Vectorize environment
    # env = DummyVecEnv([lambda: env])
    env = WarpFrame(env)  # Outputs (84,84,1)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)  # Now (84,84,4)
    env = VecTransposeImage(env)  # Converts to (4,84,84)
    
    return env


# def init_env(output_path, num_env, state, num_players, args, use_sticky_action=True, use_display=False, use_frame_skip=True):
#     def make_env(rank):
#         def _thunk():
#             games.wrappers.init(args)
            
#             # 1. Create base environment
#             env = retro.make(
#                 game=args.env,
#                 state=state,
#                 use_restricted_actions=retro.Actions.FILTERED,
#                 players=num_players,
#                 obs_type=retro.Observations.IMAGE,
#                 render_mode="rgb_array"
#             )
            
#             # 2. Apply action discretizer FIRST
#             env = SubzeroDiscretizer(env)
            
#             # 3. Seed action space
#             env.action_space.seed(seed + rank)
            
#             # 4. Apply observation wrappers
#             if args.nn != 'MlpPolicy':
#                 env = WarpFrame(env)
#                 env = ClipRewardEnv(env)
            
#             # 5. Apply frame skipping
#             if use_frame_skip:
#                 env = StochasticFrameSkip(env, n=4, stickprob=0.25 if use_sticky_action else -1)
            
#             # 6. Monitoring
#             env = Monitor(env, output_path and os.path.join(output_path, str(rank)), 
#                         allow_early_resets=True)
            
#             return env
#         return _thunk

#     # Create vectorized environment
#     env = SubprocVecEnv([make_env(i) for i in range(num_env)])
    
#     # Frame stacking for CNN policies
#     if args.nn != 'MlpPolicy':
#         env = VecFrameStack(env, n_stack=4)
#         env = VecTransposeImage(env)
    
#     return env



# def get_button_names(args):
#     env = retro.make(game=args.env, state=args.state, use_restricted_actions=retro.Actions.FILTERED, players=args.num_players)
#     print(env.buttons)
#     return env.buttons



# def init_play_env(args, num_players, is_pvp_display=False, need_display=True, use_frame_skip=True):
#     button_names = get_button_names(args)

#     env = init_env(None, 1, args.state, num_players, args, use_sticky_action=False, use_display=False, use_frame_skip=use_frame_skip)

#     if not need_display:
#         return env

#     games.wrappers.init(args)

#     if is_pvp_display:
#         display_env = env = games.wrappers.pvp_display_env(env, args, args.model1_desc, args.model2_desc, None, None, button_names)
#     else:
#         display_env = env = games.wrappers.sp_display_env(env, args, 0, args.model1_desc, button_names)

#     return display_env

# def init_play_env(args, num_players, is_pvp_display=False, need_display=True, use_frame_skip=True):
#     button_names = get_button_names(args)
#     games.wrappers.init(args)

#     # 1. Create base environment
#     env = retro.make(
#         game=args.env,
#         state=args.state,
#         use_restricted_actions=retro.Actions.FILTERED,
#         players=num_players,
#         obs_type=retro.Observations.IMAGE,
#         render_mode="human" if need_display else "rgb_array" 
#     )
#     env = StochasticFrameSkip(env, n=4, stickprob=0.25)
#     # liu kang descritiser wapper
#     env = LiuKangDiscretizer(env)
#     env = MK2RewardWrapper(env) 
#     # 2. Apply necessary wrappers FIRST
#     if args.nn != 'MlpPolicy':
#         env = WarpFrame(env)  # Applies 84x84 grayscale transformation
#     env = ClipRewardEnv(env)
    
#     # 3. Apply display wrapper BEFORE vectorization
#     if need_display:
#         if is_pvp_display:
#             env = games.wrappers.pvp_display_env(env, args, args.model1_desc, args.model2_desc, None, None, button_names)
#         else:
#             env = games.wrappers.sp_display_env(env, args, 0, args.model1_desc, button_names)

#     # 4. Convert to vectorized environment (single env)
#     env = DummyVecEnv([lambda: env])
    
#     # 5. Add frame stacking if needed
#     if args.nn != 'MlpPolicy':
#         env = VecFrameStack(env, n_stack=4)
#         env = VecTransposeImage(env)

#     return env


