# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to run the RL environment for the cartpole balancing task.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tutorials/03_envs/run_cartpole_rl_env.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import random
import copy
import datetime
import platform
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import deque

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleEnvCfg

# DDPG 파라미터 설정
state_size = 4
action_size = 1
load_model = False
train_mode = True
batch_size = 128
mem_maxlen = 30000
discount_factor = 0.9
actor_lr = 1e-4
critic_lr = 5e-4
tau = 1e-3

# OU noise 파라미터
mu = 0
theta = 1e-3
sigma = 2e-3

run_step = 50000 if train_mode else 0
train_start_step = 5000

# 연산 장치
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# OU_noise 클래스
class OU_noise:
    def __init__(self):
        self.reset()

    def reset(self):
        self.X = torch.full((action_size,), mu, device=device, dtype=torch.float32)

    def sample(self):
        dx = theta * (mu - self.X) + sigma * torch.randn_like(self.X)
        self.X += dx
        return self.X

# Actor 클래스
class Actor(torch.nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.mu = torch.nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.mu(x))

# Critic 클래스
class Critic(torch.nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, 128)
        self.fc2 = torch.nn.Linear(128 + action_size, 128)
        self.q = torch.nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.relu(self.fc1(state))
        x = torch.cat((x, action), dim=-1)
        x = torch.relu(self.fc2(x))
        return self.q(x)

# DDPGAgent 클래스
class DDPGAgent:
    def __init__(self):
        self.actor = Actor().to(device)
        self.target_actor = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic = Critic().to(device)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.OU = OU_noise()
        self.memory = deque(maxlen=mem_maxlen)

        # 파라미터 상태 확인
        for name, param in self.critic.named_parameters():
            print(f"Critic {name} requires_grad: {param.requires_grad}")
        for name, param in self.actor.named_parameters():
            print(f"Actor {name} requires_grad: {param.requires_grad}")

    def get_action(self, state, training=True):
        self.actor.train(training)
        action = self.actor(state)
        return action + self.OU.sample() if training else action

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state.clone().detach(), action.clone().detach(), 
                           reward.clone().detach(), next_state.clone().detach(), 
                           done.clone().detach()))

    def train_model(self):
        self.actor.train()
        self.critic.train()

        batch = random.sample(self.memory, batch_size)
        state = torch.stack([b[0] for b in batch]).view(-1, state_size).to(device)
        action = torch.stack([b[1] for b in batch]).view(-1, action_size).to(device)
        reward = torch.stack([b[2] for b in batch]).view(-1, 1).to(device)
        next_state = torch.stack([b[3] for b in batch]).view(-1, state_size).to(device)
        done = torch.stack([b[4] for b in batch]).view(-1, 1).to(device).float()  # bool -> float 변환

        # Critic 업데이트
        next_actions = self.target_actor(next_state)
        next_q = self.target_critic(next_state, next_actions)
        target_q = reward + (1 - done) * discount_factor * next_q
        q = self.critic(state, action)

        # 디버깅 출력
        print(f"state shape: {state.shape}, requires_grad: {state.requires_grad}")
        print(f"action shape: {action.shape}, requires_grad: {action.requires_grad}")
        print(f"q shape: {q.shape}, requires_grad: {q.requires_grad}")
        print(f"target_q shape: {target_q.shape}, requires_grad: {target_q.requires_grad}")

        critic_loss = F.mse_loss(target_q.detach(), q)
        print(f"critic_loss requires_grad: {critic_loss.requires_grad}")

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor 업데이트
        action_pred = self.actor(state)
        actor_loss = -self.critic(state, action_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update_target()
        return actor_loss.item(), critic_loss.item()

    def soft_update_target(self):
        for target_param, local_param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        for target_param, local_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    # 네트워크 모델 저장
    def save_model(self):
        print(f"... Save Model to {save_path}/ckpt ...")
        torch.save({
            "actor" : self.actor.state_dict(),
            "actor_optimizer" : self.actor_optimizer.state_dict(),
            "critic" : self.critic.state_dict(),
            "critic_optimizer" : self.critic_optimizer.state_dict(),
        }, save_path+'/ckpt')

    # 학습 기록
    def write_summray(self, score, actor_loss, critic_loss, step):
        self.writer.add_scalar("run/score", score, step)
        self.writer.add_scalar("model/actor_loss", actor_loss, step)
        self.writer.add_scalar("model/critic_loss", critic_loss, step)

def main():
    """Main function."""
    # create environment configuration
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # DDPGAgent 클래스를 agent로 정의
    agent = DDPGAgent()

    # sample random actions -> 액션 코드를 수정해야함
    joint_efforts = torch.randn_like(env.action_manager.action)
    # step the environment
    obs, rew, terminated, truncated, info = env.step(joint_efforts)

    # simulate physics
    count = 0
    actor_losses, critic_losses, scores, episode, score = [], [], [], 0, 0
    for step in range(run_step):
        with torch.inference_mode():
            # reset
            # if count % 300 == 0:
            #     count = 0
            #     env.reset()
            #     print("-" * 80)
            #     print("[INFO]: Resetting environment...")

            # 상태 : 카트의 위치, 폴의 각도, 카트의 속도, 폴의 각속도
            state = obs["policy"]
            # print(state)
            action = agent.get_action(state, train_mode)

            # step the environment
            obs, rew, terminated, truncated, info = env.step(action)
            next_state = obs["policy"]
            # print current orientation of pole
            # joint_pos={"slider_to_cart": 0.0, "cart_to_pole": 0.0}
            # joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
            # joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
            # 그래서 Pole joint의 배열 값이 0,1로 나옴 (0: num_env의 0번째, 0~1: 카트(1축)와 폴 각도, 2~3: 카트(속도)와 폴(각속도))
            # print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            score += rew[0].item()
            
            if train_mode:
                # 병렬 환경 데이터를 메모리에 저장
                for i in range(args_cli.num_envs):
                    agent.append_sample(state[i], action[i], rew[i], next_state[i], terminated[i])

            if train_mode and step > max(batch_size, train_start_step):
                # 학습 수행
                actor_loss, critic_loss = agent.train_model()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

                # 타겟 네트워크 소프트 업데이트
                agent.soft_update_target()

            if terminated.all():
                episode += 1
                scores.append(score)
                score = 0

                # 게임 진행 상황 출력 및 텐서 보드에 보상과 손실함수 값 기록
                # if episode % print_interval == 0:
                #     mean_score = np.mean(scores)
                #     mean_actor_loss = np.mean(actor_losses)
                #     mean_critic_loss = np.mean(critic_losses)
                #     agent.write_summray(mean_score, mean_actor_loss, mean_critic_loss, step)
                #     actor_losses, critic_losses, scores = [], [], []

                #     print(f"{episode} Episode / Step: {step} / Score: {mean_score:.2f} / " +\
                #         f"Actor loss: {mean_actor_loss:.2f} / Critic loss: {mean_critic_loss:.4f}")

            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
