"""
Play a pre-trained model on NHL 94
"""

import sys
import argparse
from common import com_print, init_logger
from envs import init_env, init_play_env

import game_wrappers_mgr as games

def parse_cmdline(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--alg', type=str, default='ppo2')
    parser.add_argument('--model1_desc', type=str, default='CNN')
    parser.add_argument('--nn', type=str, default='CnnPolicy')  
    parser.add_argument('--nnsize', type=int, default='256')
    parser.add_argument('--env', type=str, default='MortalKombatII-Genesis')
    parser.add_argument('--state', type=str, default='VeryEasy.LiuKang-02')
    parser.add_argument('--num_players', type=int, default='1')
    parser.add_argument('--num_env', type=int, default=1)
    parser.add_argument('--num_timesteps', type=int, default=0)
    parser.add_argument('--output_basedir', type=str, default='./logs') 
    parser.add_argument('--model_1', type=str, default='./ckpt/model_checkpointVeryEasy.LiuKang-03_19825664_steps') #load model_1
    # parser.add_argument('--model_1', type=str, default='./ckpt_subzero/rl_model_6000000_steps')
    parser.add_argument('--model_2', type=str, default='')
    parser.add_argument('--display_width', type=int, default='1440')
    parser.add_argument('--display_height', type=int, default='810')
    parser.add_argument('--deterministic', default=True, action='store_true')
    parser.add_argument('--fullscreen', default=False, action='store_true')
    parser.add_argument('--rf', type=str, default='')
    #parser.add_argument('--useframeskip', default=False, action='store_true')

    args = parser.parse_args(argv)

    return args

class ModelVsGame:
    def __init__(self, args, logger, need_display=True):

        self.p1_env = init_env(None, 1, args.state, 1, args, True)
        self.display_env = init_play_env(args, 1, False, need_display, False)
        print("Observation space(init_play_env called):", self.display_env.observation_space)

        self.ai_sys = games.wrappers.ai_sys(args, self.p1_env, logger)
        if args.model_1 != '' or args.model_2 != '':
            models = [args.model_1, args.model_2]
            self.ai_sys.SetModels(models)

        self.need_display = need_display
        self.args = args

    # def play(self, continuous=True, need_reset=True):
    #     state = self.display_env.reset()

    #     total_rewards = 0
    #     skip_frames = 0
    #     p1_actions = []
    #     info = None

    #     while True:
    #         p1_actions = self.ai_sys.predict(state, info=info, deterministic=self.args.deterministic)
    #         # print(f"Executing action: {p1_actions} -> {self.p1_env.envs[0].env.unwrapped._decode_discrete_action[p1_actions]}")
    #         # Print the combo name for the chosen action
    #         # combo_name = self.display_env.envs[0].env.get_combo_name(p1_actions)
    #         # print(f"Action {p1_actions}: {combo_name}")

    #         self.display_env.action_probabilities = []

    #         for i in range(4):
    #             if self.need_display:
    #                 # self.display_env.set_ai_sys_info(self.ai_sys)
    #                 self.display_env.env_method('set_ai_sys_info', self.ai_sys)
    #             state, reward, done, info = self.display_env.step(p1_actions)
                
    #             total_rewards += reward

    #         if done:
    #             if continuous:
    #                 if need_reset:
    #                     state = self.display_env.reset()
    #             else:
    #                 return info, total_rewards
    def play(self, continuous=True, need_reset=True):
        state = self.display_env.reset()
        total_rewards = 0
        info = {}  # Initialize info as empty dict
        
        while True:
            # Pass info to predict()
            p1_actions = self.ai_sys.predict(state, info=info, deterministic=self.args.deterministic)
            
            # Get new state and info
            state, reward, done, info = self.display_env.step(p1_actions)
            total_rewards += reward
            
            if done:
                if continuous and need_reset:   
                    state = self.display_env.reset()
                    info = {}  # Reset info on new episode
                else:
                    return info, total_rewards



       


def main(argv):
    args = parse_cmdline(argv[1:])

    logger = init_logger(args)

    games.wrappers.init(args)

    player = ModelVsGame(args, logger)

    com_print('========= Start of Game Loop ==========')
    com_print('Press ESC or Q to quit')
    player.play(need_reset=False)

if __name__ == '__main__':
    main(sys.argv)