import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import os

from wind_farm_env import WindFarmEnvPandapower 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(state_size, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000) 

        self.gamma = 0.99 
        self.learning_rate = 0.001 
        self.epsilon = 0.3 
        self.epsilon_min = 0.01 
        self.epsilon_decay = (0.3 - 0.01) / 5000 
        self.batch_size = 64 
        self.target_update_freq = 100 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = QNetwork(state_size, action_size).to(self.device)
        self.target_model = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.update_target_model() 

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        self.model.eval() 
        with torch.no_grad(): 
            act_values = self.model(state_tensor)
        self.model.train() 
        
        return np.argmax(act_values.cpu().data.numpy())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        
        states = torch.from_numpy(np.vstack([e[0] for e in minibatch])).float().to(self.device)
        actions = torch.from_numpy(np.array([e[1] for e in minibatch])).long().to(self.device).unsqueeze(-1)
        rewards = torch.from_numpy(np.array([e[2] for e in minibatch])).float().to(self.device).unsqueeze(-1)
        next_states = torch.from_numpy(np.vstack([e[3] for e in minibatch])).float().to(self.device)
        dones = torch.from_numpy(np.array([e[4] for e in minibatch]).astype(np.uint8)).float().to(self.device).unsqueeze(-1)

        Q_targets_next = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        Q_expected = self.model(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        
        self.optimizer.zero_grad() 
        loss.backward() 
        self.optimizer.step() 

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

def main():
    csv_path = ' ' 
    
    try:
        env = WindFarmEnvPandapower(data_path=csv_path)
    except ImportError as e:
        print(f"오류: {e}")
        print("Pandapower가 설치되어 있는지 확인하세요. (pip install pandapower)")
        return

    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)

    num_episodes = 5000 
    total_steps = 0
    
    episode_rewards_list = []

    print(f"고충실도 시뮬레이션 학습 시작 (Device: {agent.device})")

    for e in range(num_episodes):
        state = env.reset() 
        episode_reward = 0 

        for time in range(env.max_steps):
            action = agent.act(state)
            
            next_state, reward, done = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            total_steps += 1

            if len(agent.memory) > agent.batch_size:
                agent.replay()

            if total_steps % agent.target_update_freq == 0:
                agent.update_target_model()

            if done:
                print(f"에피소드 {e+1} 완료. (1년 시뮬레이션 종료)")
                break
        
        agent.update_epsilon()

        print(f"Episode: {e+1}/{num_episodes}, Score: {episode_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
        
        episode_rewards_list.append(episode_reward)

    print("학습 완료. 결과를 'episode_rewards.npy' 파일로 저장합니다.")
    np.save("episode_rewards.npy", np.array(episode_rewards_list))
    
    model_path = "dqn_model.pth"
    torch.save(agent.model.state_dict(), model_path)
    print(f"학습된 모델을 '{model_path}' 파일로 저장했습니다.")

    print("저장 완료.")

if __name__ == "__main__":
    main()
# 24/02/19 Init Main 
# 24/02/23 Add QNetwork 
# 24/02/27 Add Agent 
# 24/02/29 Add Replay 
# 24/03/04 Add act 
# 24/03/08 Start loop 
# 24/03/22 Debug training 
# 24/03/28 Refactor agent 
# 24/04/03 Add save model 
# 24/04/15 Adjust params 
# 24/04/25 Cleanup main 
# 24/05/01 Adjust target update 
# 24/05/15 Update training params 
