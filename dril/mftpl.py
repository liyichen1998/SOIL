#!/usr/bin/env python

import copy
import glob
import os
import time
from collections import deque
import sys
import warnings

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random

from dril.a2c_ppo_acktr import algo, utils
from dril.a2c_ppo_acktr.algo import gail
from dril.a2c_ppo_acktr.algo.behavior_cloning import BehaviorCloning
from dril.a2c_ppo_acktr.algo.ensemble import Ensemble
from dril.a2c_ppo_acktr.algo.dril import DRIL
from dril.a2c_ppo_acktr.arguments import get_args
from dril.a2c_ppo_acktr.envs import make_vec_envs
from dril.a2c_ppo_acktr.model import Policy
from dril.a2c_ppo_acktr.expert_dataset import ExpertDataset
from dril.a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
from eval_ensemble import eval_ensemble
import dril.a2c_ppo_acktr.ensemble_models as ensemble_models

import pandas as pd


from instant_dataset import InstantDataset

def mftpl(args,envs,obs,acs,sample_size, stats_path=None, hyperparams=None, time=False):
    # args = get_args()
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)

    if args.system == 'philly':
        args.demo_data_dir = os.getenv('PT_OUTPUT_DIR') + '/demo_data/'
        args.save_model_dir = os.getenv('PT_OUTPUT_DIR') + '/trained_models/'
        args.save_results_dir = os.getenv('PT_OUTPUT_DIR') + '/trained_results/'


    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    # utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")


    # envs = make_vec_envs(args.env_name, args.seed, args.num_processes, args.gamma,
    #                      args.log_dir, device, False, use_obs_norm=args.use_obs_norm,
    #                      max_steps=args.atari_max_steps)
    
    # actor_critic = Policy(
    #     envs.observation_space.shape,
    #     envs.action_space,
    #     load_expert= False,
    #     env_name=args.env_name,
    #     rl_baseline_zoo_dir=args.rl_baseline_zoo_dir,
    #     expert_algo=args.expert_algo,
    #     base_kwargs={'recurrent': args.recurrent_policy})
    # actor_critic.to(device)
    
    print('logger!')
    ensemble_size = args.ensemble_size
    try:
        num_actions = envs.action_space.n
    except:
        num_actions = envs.action_space.shape[0]
    ensemble_args = (envs.observation_space.shape[0], num_actions, args.ensemble_hidden_size, ensemble_size)
    if len(envs.observation_space.shape) == 3:
        if args.env_name in ['duckietown']:
            policy_def = ensemble_models.PolicyEnsembleDuckieTownCNN
        else:
            policy_def = ensemble_models.PolicyEnsembleCNN
    else:
        policy_def = ensemble_models.PolicyEnsembleMLP_simple
    ensemble_policy = policy_def(*ensemble_args).to(device)

    # stores results
    # main_results = []
    # args.load_expert,
    # bc_results = None
    bc_model_reward = None
    if True: #args.behavior_cloning or args.dril or args.warm_start:
        # random_num = random.randint(1,123456)
        # print(random_num)
        # expert_dataset = InstantDataset(obs, acs, args.env_name,\
        #                    args.num_trajs, random_num, args.ensemble_shuffle_type)
        expert_dataset = InstantDataset(obs, acs, args.env_name,\
                           args.num_trajs, args.seed, args.ensemble_shuffle_type)
        
        # expert_dataset = ExpertDataset(args.demo_data_dir, args.env_name,\
        #                    args.num_trajs, args.seed, args.ensemble_shuffle_type)
        # bc_model_save_path = os.path.join(args.save_model_dir, 'bc')
        # bc_file_name = f'bc_{args.env_name}_policy_ntrajs={args.num_trajs}_seed={args.seed}'
        #bc_file_name = f'{args.env_name}_bc_policy_ntraj={args.num_trajs}_seed={args.seed}'
        # bc_model_path = os.path.join(bc_model_save_path, f'{bc_file_name}.model.pth')
        # bc_results_save_path = os.path.join(args.save_results_dir, 'bc', f'{bc_file_name}.perf')

        # bc_model = BehaviorCloning(actor_critic, device, batch_size=args.bc_batch_size,\
        #         lr=args.bc_lr, training_data_split=args.training_data_split,
        #         expert_dataset=expert_dataset, envs=envs)

        ensemble = BehaviorCloning(ensemble_policy,device, batch_size=args.ensemble_batch_size,\
                   lr=args.ensemble_lr, envs=envs, training_data_split=args.training_data_split,\
                   expert_dataset=expert_dataset,ensemble_size=ensemble_size )

        # Check if model already exist
        test_reward = None
        # if os.path.exists(bc_model_path):
        #     best_test_params = torch.load(bc_model_path, map_location=device)
        #     print(f'*** Loading behavior cloning policy: {bc_model_path} ***')
        # else:
        bc_results = []
        best_test_loss, best_test_model = np.float('inf'), None
        for bc_epoch in range(args.num_ensemble_train_epoch):
            train_loss = ensemble.update(update=True, data_loader_type='train')
            # train_loss = bc_model.update(update=True, data_loader_type='train')
            # with torch.no_grad():
            #     test_loss = bc_model.update(update=False, data_loader_type='test')
            
            #if test_loss < best_test_loss:
            #    best_test_loss = test_loss
            #    best_test_params = copy.deepcopy(actor_critic.state_dict())

            # #early stop by validation does not fit when dataset is small
            # if test_loss < best_test_loss:
            #     # print('model has improved')
            #     best_test_loss = test_loss
            #     best_test_params = copy.deepcopy(actor_critic.state_dict())
            #     patience = 20
            # else:
            #     patience -= 1
            #     # print('model has not improved')
            #     if patience == 0:
            #         print('model has not improved in 20 epochs, breaking')
            #         break

            # if bc_epoch>0 and bc_epoch % int(0.5*args.bc_train_epoch-1) == 0:
            # if bc_epoch in [int(0.5*args.bc_train_epoch),args.bc_train_epoch -1]:
            #     test_reward,bc_model_std = evaluate(actor_critic, None, args.env_name, args.seed,
            #                     args.num_processes, eval_log_dir, device, num_episodes=5,
            #                     atari_max_steps=args.atari_max_steps)
            #     print(f'epoch {bc_epoch}/{args.bc_train_epoch} | train loss: {train_loss:.4f}, test return: {test_reward:.4f}')
            

            # if bc_epoch % int(100*args.bc_train_epoch) == 0:
            #     print(f'bc-epoch {bc_epoch}/{args.bc_train_epoch} | train loss: {train_loss:.4f}, test loss: {test_loss:.4f}')
            
            # Save the Behavior Cloning model and training results
            # test_reward = evaluate(actor_critic, None, args.env_name, args.seed,
            #                  args.num_processes, eval_log_dir, device, num_episodes=10,
            #                  atari_max_steps=args.atari_max_steps)
            # bc_results.append({'epoch': bc_epoch, 'trloss':train_loss, 'teloss': test_loss,\
            #             'test_reward': test_reward})

            # torch.save(best_test_params, bc_model_path)
            # df = pd.DataFrame(bc_results, columns=np.hstack(['epoch', 'trloss', 'teloss', 'test_reward']))
            # df.to_csv(bc_results_save_path)

        # Load Behavior cloning model

        # actor_critic.load_state_dict(best_test_params)
        last_params = copy.deepcopy(ensemble_policy.state_dict())
        # last_params = copy.deepcopy(actor_critic.state_dict())
        

        if test_reward is None:
            bc_model_reward,bc_model_std = eval_ensemble(ensemble_policy,ensemble_size, None, args.env_name, args.seed,
                                args.num_processes, eval_log_dir, device, num_episodes=5,
                                stats_path=stats_path, hyperparams=hyperparams, time=time)
            # print(f'test_{sample_size}')
            # bc_model_reward = evaluate(actor_critic, None, args.env_name, args.seed,
            #                     args.num_processes, None, device, num_episodes=10,
            #                     atari_max_steps=args.atari_max_steps)
            # bc_model_reward,bc_model_std = evaluate(actor_critic, None, args.env_name, args.seed,
            #                     args.num_processes, eval_log_dir, device, num_episodes=5,
            #                     atari_max_steps=args.atari_max_steps)
        else:
            bc_model_reward = test_reward
        print(f'{sample_size} samples model performance: {bc_model_reward}')
        # bc_results=(bc_model_reward)
# If behavior cloning terminate the script early
        # if args.behavior_cloning:
        #      sys.exit()
        # Reset the behavior cloning optimizer
        # bc_model.reset()
        ensemble.reset()
        # mean_by_std = bc_model_std  #/bc_model_reward
    return last_params, bc_model_reward, bc_model_std, train_loss #, best_test_loss

    # if args.dril:
    #     expert_dataset = ExpertDataset(args.demo_data_dir, args.env_name,
    #                        args.num_trajs, args.seed, args.ensemble_shuffle_type)

        # # Train or load ensemble policy
        # ensemble_policy = Ensemble(device=device, envs=envs,
        #    expert_dataset=expert_dataset,
        #    uncertainty_reward=args.dril_uncertainty_reward,
        #    ensemble_hidden_size=args.ensemble_hidden_size,
        #    ensemble_drop_rate=args.ensemble_drop_rate,
        #    ensemble_size=args.ensemble_size,
        #    ensemble_batch_size=args.ensemble_batch_size,
        #    ensemble_lr=args.ensemble_lr,
        #    num_ensemble_train_epoch=args.num_ensemble_train_epoch,
        #    num_trajs=args.num_trajs,
        #    seed=args.seed,
        #    env_name=args.env_name,
        #    training_data_split=args.training_data_split,
        #    save_model_dir=args.save_model_dir,
        #    save_results_dir=args.save_results_dir)

        # # If only training ensemble
        # if args.pretrain_ensemble_only:
        #     sys.exit()


# if __name__ == "__main__":
#     main()
