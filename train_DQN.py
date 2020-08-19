import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque
from collections import namedtuple
from torch.distributions import Categorical

import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
import argparse

from knowledge_graph import KnowledgeGraph
from kg_env import BatchKGEnvironment
from utils import *

logger = None

SavedAction = namedtuple('SavedAction', ['log_prob'])#, 'value'])

# Hyper Parameters for DQN
GAMMA = 0.9 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch
REPLACE_TARGET_FREQ = 10 # frequency to update target Q network

class DQN(nn.Module):
    # DQN Agent
    def __init__(self, env, state_dim, act_dim, gamma=0.99,hidden_sizes=[512,256]):
        super(DQN, self).__init__()
        self.replay_buffer = deque() #双向队列 可以从左append些什么
        # init some parameters
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.state_dim = 400 #state_dim
        self.action_dim = 251       
        
        self.current_net1 = nn.Linear(self.state_dim,20)
        self.current_net2 = nn.Linear(20,self.action_dim)
        
        self.target_net1 = nn.Linear(self.state_dim,20)
        self.target_net2 = nn.Linear(20,self.action_dim)
        
        '''state_dim 要改!!!'''
         #act_dim
        #env.action_space.n
        
        self.saved_actions = []        
        self.rewards = []
        self.entropy = []

    
    def forward(self,inputs):
        state,act_mask = inputs
        #act_mask = a_m
        h_1 = self.current_net1(state)
        
        h_1 = F.dropout(F.relu(h_1))
        self.Q_value = self.current_net2(h_1)
        
        h_2 = self.target_net1(state)
        h_2 = F.dropout(F.relu(h_2))
        self.target_Q_value = self.target_net2(h_2)

        #optimizer = optim.Adam(model.parameters(), lr=args.lr)
        

    def perceive(self,state,action,reward,next_state,done,optimizer):
        one_hot_action = np.zeros(self.action_dim)
        #应该要 32*action_dim
        one_hot_action[action] = 1
        self.replay_buffer.append((state,one_hot_action,reward,next_state,done))
        self.optimizer = optimizer
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()
    
    
    def train_Q_network(self):
        self.time_step += 1
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        reward_batch = torch.tensor(reward_batch)
        # Step 2: calculate y
        y_batch = []
        current_Q_batch = self.Q_value
        #max_action_next = np.argmax(current_Q_batch, axis=1)
        max_action_next=torch.argmax(current_Q_batch ,dim=1)
        target_Q_batch = self.target_Q_value

        for i in range(0,BATCH_SIZE):
            done = minibatch[i][4]
            if i == 0:
                if done:
                    y_batch = torch.tensor(reward_batch[i])
                else:
                    #print('2:',len(target_Q_batch),len(target_Q_batch[0]))
                    target_Q_value = target_Q_batch[i, max_action_next[i]] #[i, i, max_action_next[i]]
                    #print('type:',type(reward_batch),type(target_Q_value))
                    #print(reward_batch)
                    #print(target_Q_value)
                    y_batch = torch.tensor(torch.tensor(reward_batch[i]) + GAMMA * target_Q_value)
            else:
                if done:
                    #y_batch = np.append(y_batch,reward_batch[i])
                    y_batch = torch.tensor(y_batch)
                    app = torch.tensor(reward_batch[i])
                    y_batch = torch.cat((y_batch,app),0)
                else :
                    #print('1:',len(target_Q_batch),len(target_Q_batch[0]))
                    #print(len(target_Q_batch[0][0]))
                    target_Q_value = target_Q_batch[i, max_action_next[i]] #[i, i,max_action_next[i]]
                    #print('type:',type(reward_batch),type(target_Q_value))
                    y_batch = torch.tensor(y_batch)
                    app = torch.tensor(reward_batch[i]+GAMMA*target_Q_value)
                    y_batch = torch.cat((y_batch,app),0)
                    #y_batch = np.append(y_batch, reward_batch[i] + GAMMA * target_Q_value)
                    
        y_batch = y_batch.reshape(32,32)
        #print(self.entropy)
        self.loss = self.entropy.mean()
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        
    def egreedy_action(self,batch_state,act_mask):#batch_state = state
        state = torch.FloatTensor(batch_state)#.to(device)  # Tensor [bs, state_dim]
        act_mask = torch.FloatTensor(act_mask)#.to(device)  # Tensor of [bs, act_dim]
        self((state,act_mask))
        ''''''
        act_mask = act_mask.type(torch.uint8)        
        self.Q_value[1-act_mask] = -999999.0
        self.Q_value = F.softmax(self.Q_value, dim=-1)
        
        m = Categorical(self.Q_value)#加起来应该不是1？说不准
        acts = m.sample()
        # [CAVEAT] If sampled action is out of action_space, choose the first action in action_space.
        self.entropy = torch.cat((torch.tensor(self.entropy),m.entropy()))
        #self.entropy.append(m.entropy())
        
        if random.random() <= self.epsilon:
            acts = []
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            for i in range(BATCH_SIZE):
                acts.append(random.randint(0,self.action_dim - 1))
            acts_log = torch.tensor(acts)
            self.saved_actions.append(SavedAction(m.log_prob(acts_log)))#, value))
            return acts #要return一行，不能只有一个值吧!!
        else:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            acts_log = torch.tensor(acts)
            self.saved_actions.append(SavedAction(m.log_prob(acts_log)))#, value))
            return acts #np.argmax(Q_value)

        
    def action(self,batch_state,act_mask):
        batch_state = torch.tensor(batch_state,dtype=torch.float32)
        act_mask = torch.tensor(act_mask,dtype=torch.float32)
        
        self((batch_state,act_mask))
        #self.forward(batch_state,act_mask)
        ''''''
        act_mask = act_mask.type(torch.uint8)
        #self((batch_state, act_mask))        
        self.Q_value[1-act_mask] = -999999.0
        self.Q_value = F.softmax(self.Q_value, dim=-1)

        m = Categorical(self.Q_value)#加起来应该不是1？说不准 
        acts = m.sample()
        # [CAVEAT] If sampled action is out of action_space, choose the first action in action_space.
        self.entropy = torch.cat((torch.tensor(self.entropy),m.entropy()))
        #self.entropy.append(m.entropy())
        acts_log = torch.tensor(acts)
        self.saved_actions.append(SavedAction(m.log_prob(acts_log)))#, value))
        return acts
		
class ACDataLoader(object):
    def __init__(self, uids, batch_size):
        self.uids = np.array(uids)
        self.num_users = len(uids)
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        self._rand_perm = np.random.permutation(self.num_users)
        self._start_idx = 0
        self._has_next = True

    def has_next(self):
        return self._has_next


    #直接把训练集和测试机给改了就行了 序号都是 32的倍数就可
    def get_batch(self):
        if not self._has_next:
            return None
        # Multiple users per batch
        end_idx = min(self._start_idx + self.batch_size, self.num_users)
        #print('get_batch,end_idx:',end_idx,';',self.num_users)
        batch_idx = self._rand_perm[self._start_idx:end_idx]
        batch_uids = self.uids[batch_idx]
        self._has_next = self._has_next and end_idx < self.num_users
        self._start_idx = end_idx
        
        return batch_uids.tolist()

def train(args):
    
    # initialize OpenAI Gym env and dqn agent
    #env = gym.make(ENV_NAME)
    env = BatchKGEnvironment(args.dataset, args.max_acts, max_path_len=args.max_path_len, state_history=args.state_history)
    uids = list(env.kg(USER).keys())
    print('uids:',len(uids))
    uids = np.arange(19488).tolist()
    agent = DQN(env,env.state_dim,env.act_dim,gamma = args.gamma,hidden_sizes = args.hidden)
    dataloader = ACDataLoader(uids, args.batch_size)
    logger.info('Parameters:' + str([i[0] for i in agent.named_parameters()]))
    optimizer = optim.Adam(agent.parameters(), lr=args.lr)
    #model = ActorCritic(env.state_dim, env.act_dim, gamma=args.gamma, hidden_sizes=args.hidden).to(args.device)
    #logger.info('Parameters:' + str([i[0] for i in model.named_parameters()]))

    '''    
    uids = list(env.kg(USER).keys())
    dataloader = ACDataLoader(uids, args.batch_size)
    model = ActorCritic(env.state_dim, env.act_dim, gamma=args.gamma, hidden_sizes=args.hidden).to(args.device)
    logger.info('Parameters:' + str([i[0] for i in model.named_parameters()]))
    '''
    episode = 1
    for epoch in range(0, args.epochs):
        ### Start epoch ###
        if epoch % 10 == 0:
            print('epoch:',epoch)
        dataloader.reset()
        while dataloader.has_next():
            batch_uids = dataloader.get_batch()
            ### Start batch episodes ###
            #print('batch_uids:',batch_uids,';',len(batch_uids))
            batch_state1 = env.reset(batch_uids)  # numpy array of [bs, state_dim]
            #print('egreedy_action, batch_state1:',batch_state1,';',len(batch_state1),';',len(batch_state1[0]))
            for step in range(STEP):#while not done:

                batch_act_mask = env.batch_action_mask(dropout=args.act_dropout)  # numpy array of size [bs, act_dim]
                '''select action'''
                #print('shape of batch state1:',batch_state1.shape)
                batch_act_idx = agent.egreedy_action(batch_state1,batch_act_mask)#batch_act_mask
                batch_state2, batch_reward, done = env.batch_step(batch_act_idx)
                batch_state2 = batch_state2.reshape((-1,400))
                agent.perceive(batch_state1,batch_act_idx,batch_reward,batch_state2,done,optimizer)
                agent.rewards.append(batch_reward)
                #print('batch_state2:',batch_state2,';',len(batch_state2),';',len(batch_state2[0]))
                batch_state1 = batch_state2
                #train Q network
                if done:
                    break
                # Test every 100 episodes
            if epoch % 100 == 0:
                total_reward = 0
                for i in range(TEST):
                    #batch_uids = dataloader.get_batch()
                    #print('action batch_uids:',batch_uids)
                    ### Start batch episodes ###
                    batch_state1 = env.reset(batch_uids)  # numpy array of [bs, state_dim]
                    #state = env.reset()
                    for j in range(STEP):
                        #env.render()#在屏幕上显示画面，不需要
                        batch_act_mask = env.batch_action_mask(dropout=args.act_dropout)
                        #print('before action, batch_state1:',batch_state1,';',len(batch_state1),';',len(batch_state1[0]))
                        action = agent.action(batch_state1,batch_act_mask) # direct action for test
                        batch_state2, batch_reward, done = env.batch_step(action)
                        batch_state1 = batch_state2
                        total_reward += batch_reward
                        if done:
                            break
                ave_reward = total_reward/TEST
                if episode % 100 == 0:
                    print ('episode: ',episode,'Evaluation Average Reward:',sum(ave_reward)/len(ave_reward))
                episode = episode + 1 
        #agent.update_target_q_network(epoch)
        
        #########
        if epoch % 10 == 0:
            policy_file = '{}/dqn3_model_epoch_{}.ckpt'.format(args.log_dir, epoch)
            logger.info("Save model to " + policy_file)
            print('epoch:',epoch,',episode:',episode)
            torch.save(agent.state_dict(), policy_file)
