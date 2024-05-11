import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests 
import time
import os

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

STATE_SIZE = 43 
ACTION_SIZE = 3 
MODEL_PATH = "dqn_model.pth" 
API_URL = "http://127.0.0.1:5000/wind_data" 
CONTROL_INTERVAL_SECONDS = 3600 

def load_trained_model(model_path, state_size, action_size):
    if not os.path.exists(model_path):
        print(f"오류: 모델 파일 '{model_path}'을 찾을 수 없습니다.")
        print("먼저 'main_pandapower.py'를 실행하여 모델을 학습/저장해야 합니다.")
        return None

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = QNetwork(state_size, action_size).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() 
        print(f"모델 로드 완료 (Device: {device})")
        return model, device
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        return None, None

def get_realtime_state(api_url):
    try:
        response = requests.get(api_url)
        response.raise_for_status() 
        
        data = response.json()
        wind_power = np.array(data['wind_power'])
        topology = np.array(data['topology'])
        
        state = np.concatenate((wind_power, topology))
        return state
        
    except requests.exceptions.RequestException as e:
        print(f"API 데이터 수신 오류: {e}")
        return None

def predict_best_action(model, state, device):
    with torch.no_grad(): 
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        act_values = model(state_tensor)
        
        return np.argmax(act_values.cpu().data.numpy())

def execute_action(action):
    print(f"  -> [제어 명령] {action}번 차단기를 조작합니다.")


def main_controller():
    print("--- 실시간 자동 제어 프로그램 (Agent Controller) ---")
    
    model, device = load_trained_model(MODEL_PATH, STATE_SIZE, ACTION_SIZE)
    if model is None:
        return

    while True:
        print(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} - 다음 제어 사이클 시작...")
        
        print(f"  - '{API_URL}'에서 실시간 상태 데이터를 요청합니다...")
        current_state = get_realtime_state(API_URL)
        
        if current_state is not None:
            print(f"  - 현재 상태 수신 완료 (터빈 {current_state[0]:.2f}MW...)")
            best_action = predict_best_action(model, current_state, device)
            print(f"  - AI가 최적 행동을 '{best_action}' (으)로 결정했습니다.")
            
            execute_action(best_action)
            
        else:
            print("  - 데이터 수신에 실패하여 이번 사이클을 건너뜁니다.")

        print(f"  - {CONTROL_INTERVAL_SECONDS}초 후 다음 제어를 실행합니다.")
        time.sleep(CONTROL_INTERVAL_SECONDS)

if __name__ == "__main__":
    main_controller()
# 24/03/14 Init Controller 
# 24/03/20 Add API request 
# 24/04/01 Add load model 
# 24/04/09 Test controller 
# 24/04/17 Add error handling 
# 24/04/23 Refine controller loop 
# 24/05/03 Fix controller predict 
# 24/05/13 Update controller state 
# 24/05/23 Add docs 
# 24/05/30 Final check 
# 24/06/17 Final fix 
# 24/03/13 Init Controller 
# 24/03/20 Add API request 
# 24/04/02 Add load model 
# 24/04/13 (Sat) Add error handling 
# 24/04/22 Refine controller loop 
# 24/05/11 (Sat) Update controller state 
# 24/05/24 Add docs 
# 24/03/13 Init Controller 
# 24/03/20 Add API request 
# 24/04/02 Add load model 
# 24/04/13 (Sat) Add error handling 
# 24/04/22 Refine controller loop 
# 24/05/11 (Sat) Update controller state 
