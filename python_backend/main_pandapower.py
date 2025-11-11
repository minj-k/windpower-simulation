import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import os

# Pandapower 환경 클래스를 불러옵니다.
from windfarm_env_pandapower import WindFarmEnvPandapower 

# TensorFlow 관련 로그를 숨깁니다 (Pandapower가 내부적으로 사용할 수 있음)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class QNetwork(nn.Module):
    """
    DQN 에이전트가 사용할 신경망 모델 (PyTorch)
    """
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        # 상태 크기를 입력으로, 64개 노드를 출력으로 하는 첫 번째 완전 연결 계층
        self.layer1 = nn.Linear(state_size, 64)
        # 64개 노드를 입력으로, 64개 노드를 출력으로 하는 두 번째 완전 연결 계층
        self.layer2 = nn.Linear(64, 64)
        # 64개 노드를 입력으로, 행동 크기(Q-value)를 출력으로 하는 세 번째 완전 연결 계층
        self.layer3 = nn.Linear(64, action_size)

    def forward(self, x):
        """신경망의 순전파를 정의합니다."""
        # ReLU 활성화 함수 적용
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        # 마지막 계층은 활성화 함수 없음 (Q-value는 선형 값)
        return self.layer3(x)

class DQNAgent:
    """
    DQN 에이전트 (학습 로직 담당)
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        # 리플레이 버퍼 (메모리)
        self.memory = deque(maxlen=20000) 

        # DQN 하이퍼파라미터 (논문 Table 2 참조)
        self.gamma = 0.99           # 감가율 (미래 보상 가중치)
        self.learning_rate = 0.001  # 학습률
        self.epsilon = 0.3          # 초기 Epsilon (탐험 확률)
        self.epsilon_min = 0.01     # 최소 Epsilon
        self.epsilon_decay = (0.3 - 0.01) / 5000 # 5000 에피소드에 걸쳐 감소
        self.batch_size = 64        # 미니배치 크기
        self.target_update_freq = 100 # 타겟 네트워크 업데이트 주기 (스텝 기준)

        # GPU 사용 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q-Network(메인 모델)와 Target-Network(타겟 모델) 생성
        self.model = QNetwork(state_size, action_size).to(self.device)
        self.target_model = QNetwork(state_size, action_size).to(self.device)
        # Adam 옵티마이저 설정
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # 타겟 네트워크 가중치를 메인 네트워크와 동일하게 초기화
        self.update_target_model() 

    def update_target_model(self):
        """타겟 네트워크의 가중치를 메인 네트워크와 동일하게 복사"""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """경험(S, A, R, S', Done)을 리플레이 버퍼(메모리)에 저장"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Epsilon-Greedy 정책에 따라 행동 결정"""
        # Epsilon 확률로 무작위 행동 (탐험)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # (1-Epsilon) 확률로 Q-value가 가장 높은 행동 (활용)
        # 상태를 PyTorch 텐서로 변환 (GPU로 이동)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        self.model.eval() # 평가 모드 (Dropout 등 비활성화)
        with torch.no_grad(): # 그래디언트 계산 비활성화
            act_values = self.model(state_tensor)
        self.model.train() # 다시 학습 모드로 전환
        
        # Q-value가 가장 큰 행동의 인덱스를 반환
        return np.argmax(act_values.cpu().data.numpy())

    def replay(self):
        """리플레이 버퍼에서 미니배치를 샘플링하여 모델을 1회 학습"""
        # 버퍼에 데이터가 충분히 쌓이지 않았으면 학습하지 않음
        if len(self.memory) < self.batch_size:
            return

        # 미니배치 랜덤 샘플링
        minibatch = random.sample(self.memory, self.batch_size)
        
        # 데이터를 PyTorch 텐서로 변환 (GPU로 이동)
        states = torch.from_numpy(np.vstack([e[0] for e in minibatch])).float().to(self.device)
        actions = torch.from_numpy(np.array([e[1] for e in minibatch])).long().to(self.device).unsqueeze(-1)
        rewards = torch.from_numpy(np.array([e[2] for e in minibatch])).float().to(self.device).unsqueeze(-1)
        next_states = torch.from_numpy(np.vstack([e[3] for e in minibatch])).float().to(self.device)
        dones = torch.from_numpy(np.array([e[4] for e in minibatch]).astype(np.uint8)).float().to(self.device).unsqueeze(-1)

        # --- 타겟 Q-value 계산 (논문 식 (11) 부분) ---
        # 다음 상태(next_states)에 대한 Q-value를 타겟 모델에서 계산
        # .detach()로 그래디언트 흐름 차단
        Q_targets_next = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
        # 타겟 Q-value: R + gamma * max_a' Q_target(S', a')
        # (단, 에피소드가 종료(done=1)된 경우, 미래 가치(Q_targets_next)는 0으로)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # --- 현재 Q-value 계산 ---
        # 현재 상태(states)에서 실제 선택했던 행동(actions)의 Q-value를 메인 모델에서 계산
        Q_expected = self.model(states).gather(1, actions)

        # --- 손실함수 계산 (논문 식 (12) 부분) ---
        # 타겟 Q-value와 현재 Q-value 간의 MSE (Mean Squared Error) 손실
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # --- 모델 업데이트 (경사 하강법) ---
        self.optimizer.zero_grad() # 그래디언트 초기화
        loss.backward()            # 역전파
        self.optimizer.step()      # 가중치 업데이트

    def update_epsilon(self):
        """Epsilon 값을 선형적으로 감소시킴 (탐험 확률 감소)"""
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

def main():
    """메인 학습 루프"""
    
    # 이 경로에 실제 CSV 파일 위치를 지정해야 합니다.
    csv_path = ' ' 
    
    try:
        # 고충실도 Pandapower 환경 생성
        env = WindFarmEnvPandapower(data_path=csv_path)
    except ImportError as e:
        print(f"오류: {e}")
        print("Pandapower가 설치되어 있는지 확인하세요. (pip install pandapower)")
        return

    # DQN 에이전트 생성
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)

    num_episodes = 5000 # 총 학습 에피소드 수 (논문과 동일)
    total_steps = 0
    
    # 결과 저장을 위한 리스트
    episode_rewards_list = []

    print(f"고충실도 시뮬레이션 학습 시작 (Device: {agent.device})")

    # 에피소드 루프
    for e in range(num_episodes):
        state = env.reset() # 환경 초기화
        episode_reward = 0  # 현재 에피소드 보상 초기화

        # 1 에피소드 = 1년 (8760 스텝)
        for time in range(env.max_steps):
            # 1. 행동 결정
            action = agent.act(state)
            
            # 2. 환경에서 행동 수행
            next_state, reward, done = env.step(action)
            
            # 3. 경험(S, A, R, S', Done)을 리플레이 버퍼에 저장
            agent.remember(state, action, reward, next_state, done)
            
            # 4. 상태 업데이트
            state = next_state
            episode_reward += reward
            total_steps += 1

            # 5. 미니배치 크기만큼 경험이 쌓이면 학습(replay) 시작
            if len(agent.memory) > agent.batch_size:
                agent.replay()

            # 6. 일정 스텝마다 타겟 네트워크 업데이트
            if total_steps % agent.target_update_freq == 0:
                agent.update_target_model()

            # 7. 에피소드 종료 (1년 시뮬레이션 완료)
            if done:
                print(f"에피소드 {e+1} 완료. (1년 시뮬레이션 종료)")
                break
        
        # 에피소드 종료 시 Epsilon 감소 (탐험 확률 줄이기)
        agent.update_epsilon()

        # 학습 결과 출력
        print(f"Episode: {e+1}/{num_episodes}, Score: {episode_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
        
        # 결과 그래프를 그리기 위해 보상 저장
        episode_rewards_list.append(episode_reward)

    print("학습 완료. 결과를 'episode_rewards.npy' 파일로 저장합니다.")
    np.save("episode_rewards.npy", np.array(episode_rewards_list))
    
    # --- 학습된 모델 저장 ---
    # 학습된 모델의 '두뇌'(가중치)를 파일로 저장합니다.
    # 이 파일은 'agent_controller.py'에서 불러와 사용됩니다.
    model_path = "dqn_model.pth"
    torch.save(agent.model.state_dict(), model_path)
    print(f"학습된 모델을 '{model_path}' 파일로 저장했습니다.")
    # ---------------------------------

    print("저장 완료.")

if __name__ == "__main__":
    main()
