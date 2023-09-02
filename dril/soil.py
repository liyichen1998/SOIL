import gym, os
import numpy as np
import argparse
import random
import pandas as pd
import copy

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
import dril.a2c_ppo_acktr.ensemble_models as ensemble_models
import os

from small_test import dagger
from mftpl import mftpl
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
    expert_param = copy.deepcopy(th_model.state_dict())

    # test_state = torch.tensor( [7.6010e-02,  0.0000e+00,  1.0000e+00,  7.1802e-01,  0.0000e+00, -1.9221e-01,  0.0000e+00, -4.3330e-01,  1.0630e+00,  0.0000e+00, -3.2823e-02, -1.4766e-03, -1.3534e-01,  6.7750e-01,  0.0000e+00]).float().to(device)
    # # print('state',test_state.cpu())
    # print(test_state)
    # _, test_action, _, _ = th_model.act(test_state,None,None,deterministic=True) 
    # print('1 expert action',test_action.cpu().detach().numpy())

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

saved_param = None

if args.dagger or args.behavior_cloning:
#initialization of dagger agent
    actor_critic = Policy(
            env.observation_space.shape,
            env.action_space,
            load_expert= False,#True False
            partial_load = False,#True
            env_name=args.env_name,
            rl_baseline_zoo_dir=args.rl_baseline_zoo_dir,
            expert_algo=args.expert_algo,
            base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)
    saved_param = copy.deepcopy(actor_critic.state_dict())
    
    # test_state = torch.tensor( [7.6010e-02,  0.0000e+00,  1.0000e+00,  7.1802e-01,  0.0000e+00, -1.9221e-01,  0.0000e+00, -4.3330e-01,  1.0630e+00,  0.0000e+00, -3.2823e-02, -1.4766e-03, -1.3534e-01,  6.7750e-01,  0.0000e+00]).float().to(device)
    # # print('state',test_state.cpu())
    # _, test_action, _, _ = actor_critic.act(test_state,None,None,deterministic=True) 
    # print('2 dagger actor_action',test_action.cpu().detach().numpy())



if args.logger:
#define ensemble policy    
    ensemble_size = args.ensemble_size
    try:
        num_actions = env.action_space.n
    except:
        num_actions = env.action_space.shape[0]
    ensemble_args = (env.observation_space.shape[0], num_actions, args.ensemble_hidden_size, ensemble_size)
    if len(env.observation_space.shape) == 3:
        if args.env_name in ['duckietown']:
            policy_def = ensemble_models.PolicyEnsembleDuckieTownCNN
        else:
            policy_def = ensemble_models.PolicyEnsembleCNN
    else:
        policy_def = ensemble_models.PolicyEnsembleMLP_simple
    ensemble_policy = policy_def(*ensemble_args).to(device)
    
    saved_param = copy.deepcopy(ensemble_policy.state_dict())

    random_selection_list = np.random.randint(low=0, high=ensemble_size, size=1000)

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
    
        if args.dagger:    
            _, agent_action, _, _ = actor_critic.act(obs,None,None,deterministic=True)
        if args.logger:    
            ensemble_obs = torch.unsqueeze(obs, dim=0)
            ensemble_obs = torch.cat([ensemble_obs.repeat(ensemble_size, *[1]*len(ensemble_obs.shape[1:]))], dim=0)
            ensemble_actions = ensemble_policy(ensemble_obs).squeeze(1)
            # .view(ensemble_size, -1)
            # print(ensemble_actions.shape)

    if isinstance(env.action_space, gym.spaces.Box):
        # clip_action = action.cpu()
        clip_action = np.clip(action.cpu(), env.action_space.low, env.action_space.high)
        # print(clip_action,agent_action.cpu()) #for debug
        if args.dagger:
            clip_agent_action = np.clip(agent_action.cpu(), env.action_space.low, env.action_space.high)
        if args.logger:
            clip_ensemble_actions = np.clip(ensemble_actions.cpu(), env.action_space.low, env.action_space.high)
            selected_action = clip_ensemble_actions[random_selection_list[step]]   
            # print(selected_action.shape)
            # print(random_selection_list[step])
        # torch.clamp(action, float(eval_envs.action_space.low[0]),float(eval_envs.action_space.high[0]))
    else:
        clip_action = action.cpu()
        if args.dagger:
            clip_agent_action = agent_action.cpu()
        if args.logger:
            clip_ensemble_actions = ensemble_actions.cpu()
            selected_action = clip_ensemble_actions[random_selection_list[step]]
            

    

    # print(step)
    # print(idx)
    activate = False
    if (step == idx and args.subsample) or not args.subsample:
        #if args.env_name in env_hyperparam:
        # print(len(rtn_obs))
        # # realize that obs may need to normalize
        # print('actual_obs',obs)
        # print('dataset_obs',env.venv.get_original_obs())

        # # todo: understand why they need original observation
        # if time:
        #     try: # If vectornormalize is on
        #         rtn_obs.append(env.venv.get_original_obs())
        #     except: # if vectornormalize is off
        #         rtn_obs.append(env.venv.envs[0].get_original_obs())
        # else:
        #     try: # If time is on and vectornormalize is on
        #         rtn_obs.append(env.venv.get_original_obs())
        #     except: # If time is off and vectornormalize is off
        #         rtn_obs.append(obs.cpu().numpy().copy())

        rtn_obs.append(obs.cpu().numpy().copy())    #make sure the rollout have the same observation with the training

        rtn_acs.append(action.cpu().numpy().copy())
        idx += args.subsample_frequency
        activate = True
    #original BC
    # if args.env_name in ['duckietown']:
    #     obs, reward, done, infos = env.step(clip_action.squeeze())
    #     obs = torch.FloatTensor([obs])
    # else:
    #     obs, reward, done, infos = env.step(clip_action)
    if args.behavior_cloning:
        if args.env_name in ['duckietown']:
            obs, reward, done, infos = env.step(clip_action.squeeze())
            obs = torch.FloatTensor([obs])
        else:
            obs, reward, done, infos = env.step(clip_action)
    elif args.dagger:
        if args.env_name in ['duckietown']:
            obs, reward, done, infos = env.step(clip_agent_action.squeeze())
            obs = torch.FloatTensor([obs])
        else:
            obs, reward, done, infos = env.step(clip_agent_action)
    elif args.logger:
        if args.env_name in ['duckietown']:
            obs, reward, done, infos = env.step(selected_action.squeeze())
            obs = torch.FloatTensor([obs])
        else:
            obs, reward, done, infos = env.step(selected_action)
    else:
        raise NotImplementedError
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
            random_selection_list = np.random.randint(low=0, high=args.ensemble_size, size=1000)
            
    else:
        for info in infos or done:
            if 'episode' in info.keys():
                # print(f"reward: {info['episode']['r']}")
                ep_rewards.append(info['episode']['r'])
                save = True
                obs = env.reset()
                step = 0
                idx=random.randint(1,args.subsample_frequency)
                random_selection_list = np.random.randint(low=0, high=args.ensemble_size, size=1000)
                
    # if step!= 1 and step == idx-args.subsample_frequency+1 and len(rtn_obs) % args.data_per_round == 0:   
    if activate and len(rtn_obs) % args.data_per_round == 0:
        if int(len(rtn_obs)/args.data_per_round) in range(args.rounds+1):
        # [1, 3, 5, 10, 15, 20]:
        # [1,2,3,4,5,6,7,8,9,10,15,20]
        # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
            #and save
            obs = env.reset()
            step = 0
            idx=random.randint(1,args.subsample_frequency)
            random_selection_list = np.random.randint(low=0, high=args.ensemble_size, size=1000)
            # print('bc',int(len(rtn_obs)/args.data_per_round))
            print('sample size',len(rtn_obs))
            rtn_obs_ = np.concatenate(rtn_obs)
            rtn_acs_ = np.concatenate(rtn_acs)


            if args.logger:
                ensemble_param, result, std, loss = mftpl(args,env,deepcopy(rtn_obs_),deepcopy(rtn_acs_),len(rtn_obs), stats_path=stats_path, hyperparams=hyperparams, time=time )
                # bc_result,bc_std,bc_loss = logger(args,env,deepcopy(rtn_obs_),deepcopy(rtn_acs_),len(rtn_obs))
                saved_param = deepcopy(ensemble_param)
                ensemble_policy.load_state_dict(saved_param)

                print(result,std,loss)
                bc_result_list.append(result)
                bc_std_list.append(0)
                bc_rollout_std.append(std)
            else:
                ens_result_list = []
                ens_std_list = []
                ens_loss_list = []
                
                for ensembles in range(1):
                    bc_param, bc_result,bc_std,bc_loss = dagger(args,env,deepcopy(rtn_obs_),deepcopy(rtn_acs_),deepcopy(saved_param),len(rtn_obs),fix_feature_layer=True, stats_path=stats_path, hyperparams=hyperparams, time=time) 
                    # bc_result,bc_std,bc_loss = logger(args,env,deepcopy(rtn_obs_),deepcopy(rtn_acs_),len(rtn_obs))
                    
                    ens_result_list.append(bc_result)
                    ens_std_list.append(bc_std)
                    ens_loss_list.append(bc_loss)

                # update model
                
                #saved_param = deepcopy(expert_param)  # just for test
                saved_param = deepcopy(bc_param)
                actor_critic.load_state_dict(saved_param)
                print(ens_result_list)
                print(ens_loss_list)

                # covariance = np.cov(ens_result_list, ens_loss_list)[0][1]
                # print(covariance)
                bc_result_list.append(np.mean(ens_result_list))
                # bc_std_list.append(np.std(ens_result_list))
                                #    /np.mean(ens_result_list))
                bc_rollout_std.append(np.mean(ens_std_list))
                # covariance_list.append(covariance)

            if int(len(rtn_obs)/args.data_per_round) % args.rounds == 0:
                id_list = list(range(args.rounds))
                    # [1,2,3,4,5,6,7,8,9,10,15,20]
                    # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
                    # [1, 3, 5, 10, 15, 20]
                if args.logger:
                    print('round, policy return, std across rollouts')
                    for i in range(len(id_list)):
                        print(id_list[i], bc_result_list[i],bc_rollout_std[i])
                else:
                    print('round, policy return, std across rollouts')
                    # print('round, policy return, std across policies, std across rollouts, cov_reward_trainloss')
                    for i in range(len(id_list)):
                        print(id_list[i], bc_result_list[i], bc_rollout_std[i])
                        # print(id_list[i], bc_result_list[i],bc_std_list[i],bc_rollout_std[i],covariance_list[i])
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
