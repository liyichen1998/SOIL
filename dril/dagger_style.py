import gym, os
import numpy as np
import argparse
import random
import pandas as pd

import sys
import torch
from gym import wrappers
import random
import torch.nn.functional as F
import torch.nn as nn
import torch as th

from dril.a2c_ppo_acktr.envs import make_vec_envs
from dril.a2c_ppo_acktr.model import Policy
from dril.a2c_ppo_acktr.arguments import get_args
import os

from small_test import dagger
from copy import deepcopy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

args = get_args()

args.recurrent_policy = False
args.load_expert = True

os.system(f'mkdir -p {args.demo_data_dir}')
os.system(f'mkdir -p {args.demo_data_dir}/tmp/gym')
sys.path.insert(1,os.path.join(args.rl_baseline_zoo_dir, 'utils'))
from a2c_ppo_acktr.utils import get_saved_hyperparams

#device = torch.device("cpu")
device = torch.device("cuda:0" if args.cuda else "cpu")
print(f'device: {device}')
seed = args.seed
print(f'seed: {seed}')

if args.env_name in ['highway-v0']:
    import highway_env
    from rl_agents.agents.common.factory import agent_factory

    env = make_vec_envs(args.env_name, seed, 1, 0.99, f'{args.emo_data_dir}/tmp/gym', device,\
                       True, stats_path=stats_path, hyperparams=hyperparams, time=time,
                       atari_max_steps=args.atari_max_steps)
    # envs = make_vec_envs(args.env_name, args.seed, args.num_processes, args.gamma,
    #                      args.log_dir, device, False, use_obs_norm=args.use_obs_norm,
    #                      max_steps=args.atari_max_steps)
    agent_config = {
        "__class__": "<class 'rl_agents.agents.tree_search.deterministic.DeterministicPlannerAgent'>",
        "budget": args.data_per_round,
        "gamma": 0.7,
    }
    th_model = agent_factory(gym.make(args.env_name), agent_config)
    time = False
elif args.env_name in ['duckietown']:
    from a2c_ppo_acktr.duckietown.env import launch_env
    from a2c_ppo_acktr.duckietown.wrappers import NormalizeWrapper, ImgWrapper,\
         DtRewardWrapper, ActionWrapper, ResizeWrapper
    from a2c_ppo_acktr.duckietown.teacher import PurePursuitExpert
    env = launch_env()
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = ImgWrapper(env)
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)

     # Create an imperfect demonstrator
    expert = PurePursuitExpert(env=env)
    time = False
else:
    print('[Setting environemnt hyperparams variables]')
    stats_path = os.path.join(args.rl_baseline_zoo_dir, 'trained_agents', f'{args.expert_algo}',\
                        f'{args.env_name}')
    hyperparams, stats_path = get_saved_hyperparams(stats_path, test_mode=True,\
                                         norm_reward=args.norm_reward_stable_baseline)

    ## Load saved policy

    # subset of the environments have time wrapper
    time_wrapper_envs = ['HalfCheetahBulletEnv-v0', 'Walker2DBulletEnv-v0', 'AntBulletEnv-v0']
    if args.env_name in time_wrapper_envs:
        time=True
    else:
        time = False

    env = make_vec_envs(args.env_name, seed, 1, 0.99, f'{args.demo_data_dir}/tmp/gym', device,\
                       True, stats_path=stats_path, hyperparams=hyperparams, time=time)

    th_model = Policy(
           env.observation_space.shape,
           env.action_space,
           load_expert=True,
           env_name=args.env_name,
           rl_baseline_zoo_dir=args.rl_baseline_zoo_dir,
           expert_algo=args.expert_algo,
           # [Bug]: normalize=False,
           normalize=True if hasattr(gym.envs, 'atari') else False,
           base_kwargs={'recurrent': args.recurrent_policy}).to(device)

rtn_obs, rtn_acs, rtn_lens, ep_rewards = [], [], [], []
obs = env.reset()
if args.env_name in ['duckietown']:
    obs = torch.FloatTensor([obs])

save = True
print(f'[running]')

step = 0
args.seed = args.seed
idx = random.randint(1,args.subsample_frequency)

obs_path_suffix = f'{args.demo_data_dir}/obs_{args.env_name}_seed={args.seed}'
acs_path_suffix = f'{args.demo_data_dir}/acs_{args.env_name}_seed={args.seed}'

bc_result_list = []
bc_std_list = []
bc_rollout_std = []
covariance_list = []

#initialization of agent
actor_critic = Policy(
        env.observation_space.shape,
        env.action_space,
        load_expert= False,
        env_name=args.env_name,
        rl_baseline_zoo_dir=args.rl_baseline_zoo_dir,
        expert_algo=args.expert_algo,
        base_kwargs={'recurrent': False})
actor_critic.to(device)

while True:
    with torch.no_grad():
        if args.env_name in ['highway-v0']:
            action = torch.tensor([[th_model.act(obs)]])
        elif args.env_name in ['duckietown']:
            action = torch.FloatTensor([expert.predict(None)])
        elif hasattr(gym.envs, 'atari'):
            _, actor_features, _ = th_model.base(obs, None, None)
            #action = th.argmax(th_model.dist.linear(actor_features)).reshape(-1,1)
            dist = th_model.dist(actor_features)
            action = dist.sample()
        else:
            _, action, _, _ = th_model.act(obs, None, None, deterministic=True)

    with torch.no_grad():
            _, agent_action, _, eval_recurrent_hidden_states = actor_critic.act(obs, None, None, deterministic=True)


    if isinstance(env.action_space, gym.spaces.Box):
        clip_action = np.clip(action.cpu(), env.action_space.low, env.action_space.high)
        clip_agent_action = np.clip(agent_action.cpu(), env.action_space.low, env.action_space.high)
        # torch.clamp(action, float(eval_envs.action_space.low[0]),float(eval_envs.action_space.high[0]))
    else:
        clip_action = action
        clip_agent_action = agent_action.cpu()
    # print(step)
    # print(idx)
    activate = False
    if (step == idx and args.subsample) or not args.subsample:
        #if args.env_name in env_hyperparam:
        # print(len(rtn_obs))
        if time:
            try: # If vectornormalize is on
                rtn_obs.append(env.venv.get_original_obs())
            except: # if vectornormalize is off
                rtn_obs.append(env.venv.envs[0].get_original_obs())
        else:
            try: # If time is on and vectornormalize is on
                rtn_obs.append(env.venv.get_original_obs())
            except: # If time is off and vectornormalize is off
                rtn_obs.append(obs.cpu().numpy().copy())

        rtn_acs.append(action.cpu().numpy().copy())
        idx += args.subsample_frequency
        activate = True
    #original BC
    # if args.env_name in ['duckietown']:
    #     obs, reward, done, infos = env.step(clip_action.squeeze())
    #     obs = torch.FloatTensor([obs])
    # else:
    #     obs, reward, done, infos = env.step(clip_action)

    if args.dagger:
        if args.env_name in ['duckietown']:
            obs, reward, done, infos = env.step(clip_agent_action.squeeze())
            obs = torch.FloatTensor([obs])
        else:
            obs, reward, done, infos = env.step(clip_agent_action)
    else:
        if args.env_name in ['duckietown']:
            obs, reward, done, infos = env.step(clip_action.squeeze())
            obs = torch.FloatTensor([obs])
        else:
            obs, reward, done, infos = env.step(clip_action)

    step += 1
    if args.env_name in ['duckietown']:
        if done:
            # print(f"reward: {reward}")
            ep_rewards.append(reward)
            save = True
            obs = env.reset()
            obs = torch.FloatTensor([obs])
            step = 0
            idx=random.randint(1,args.subsample_frequency)
    else:
        for info in infos or done:
            if 'episode' in info.keys():
                # print(f"reward: {info['episode']['r']}")
                ep_rewards.append(info['episode']['r'])
                save = True
                obs = env.reset()
                step = 0
                idx=random.randint(1,args.subsample_frequency)
                
    # if step!= 1 and step == idx-args.subsample_frequency+1 and len(rtn_obs) % args.data_per_round == 0:   
    if activate and len(rtn_obs) % args.data_per_round == 0:
        if int(len(rtn_obs)/args.data_per_round) in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
        # [1, 3, 5, 10, 15, 20]:
        # [1,2,3,4,5,6,7,8,9,10,15,20]
        # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
            #and save
            obs = env.reset()
            step = 0
            idx=random.randint(1,args.subsample_frequency)
            
            # print('bc',int(len(rtn_obs)/args.data_per_round))
            print('sample size',len(rtn_obs))
            rtn_obs_ = np.concatenate(rtn_obs)
            rtn_acs_ = np.concatenate(rtn_acs)

           
            ens_result_list = []
            ens_std_list = []
            ens_loss_list = []

            for ensembles in range(3):
                bc_param, bc_result,bc_std,bc_loss = dagger(args,env,deepcopy(rtn_obs_),deepcopy(rtn_acs_),len(rtn_obs))
                # bc_result,bc_std,bc_loss = logger(args,env,deepcopy(rtn_obs_),deepcopy(rtn_acs_),len(rtn_obs))
                actor_critic.load_state_dict(bc_param)
                ens_result_list.append(bc_result)
                ens_std_list.append(bc_std)
                ens_loss_list.append(bc_loss)
            # print(bc_result_list)

            # obs_path = f'{obs_path_suffix}_ntraj={len(ep_rewards)}.npy'
            # acs_path = f'{acs_path_suffix}_ntraj={len(ep_rewards)}.npy'

            # print(f'saving to: {obs_path}')
            # print(f'saving to: {acs_path}')

            # np.save(obs_path, rtn_obs_)
            # np.save(acs_path, rtn_acs_)
            # print(f'done, length :{len(ep_rewards)}')
            # save = False
            print(ens_result_list)
            print(ens_loss_list)

            covariance = np.cov(ens_result_list, ens_loss_list)[0][1]
            print(covariance)
            bc_result_list.append(np.mean(ens_result_list))
            bc_std_list.append(np.std(ens_result_list))
                            #    /np.mean(ens_result_list))
            bc_rollout_std.append(np.mean(ens_std_list))
            covariance_list.append(covariance)

            if int(len(rtn_obs)/args.data_per_round) % 20 == 0:
                print('policy return, std across policies, std across rollouts, cov_reward_trainloss')
                id_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
                # [1,2,3,4,5,6,7,8,9,10,15,20]
                # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
                # [1, 3, 5, 10, 15, 20]
                for i in range(len(id_list)):
                    print(id_list[i], bc_result_list[i],bc_std_list[i],bc_rollout_std[i],covariance_list[i])
                # print(bc_result_list)) #
                # print('bc optimize std')
                # print(bc_std_list)
                # print('bc rollout std')
                # print(bc_rollout_std)
                break

# print(f'expert: {np.mean(ep_rewards)}')
# results_save_path = os.path.join(args.save_results_dir, 'expert', f'expert_{args.env_name}_seed={args.seed}.perf')
# results = [{'total_num_steps':0 , 'train_loss': 0, 'test_loss': 0, 'num_trajs': 0 ,\
#     'test_reward':np.mean(ep_rewards), 'u_reward': 0}]
# df = pd.DataFrame(results, columns=np.hstack(['x', 'steps', 'train_loss', 'test_loss',\
#                  'train_reward', 'test_reward', 'label', 'u_reward']))
# df.to_csv(results_save_path)
