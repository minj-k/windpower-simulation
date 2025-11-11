import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests # API 요청을 위한 라이브러리
import time
import os

# --- QNetwork 정의 ---
# main_pandapower.py 에 있는 QNetwork 클래스 정의를 그대로 가져옵니다.
# 모델 구조가 일치해야 저장된 가중치를 불러올 수 있습니다.
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

# --- 설정값 ---
STATE_SIZE = 43  # (터빈 40개 + 차단기 3개)
ACTION_SIZE = 3  # (차단기 3개 조작)
MODEL_PATH = "dqn_model.pth" # main_pandapower.py가 저장한 '두뇌 파일'
API_URL = "http://127.0.0.1:5000/wind_data" # api_mock.py 가 실행 중인 주소
CONTROL_INTERVAL_SECONDS = 3600 # 1시간 (3600초) 마다 제어 (테스트 시 10초로 변경)

def load_trained_model(model_path, state_size, action_size):
    """학습된 '두뇌 파일'(.pth)을 불러와 모델을 준비합니다."""
    
    if not os.path.exists(model_path):
        print(f"오류: 모델 파일 '{model_path}'을 찾을 수 없습니다.")
        print("먼저 'main_pandapower.py'를 실행하여 모델을 학습/저장해야 합니다.")
        return None

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = QNetwork(state_size, action_size).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # *** 매우 중요: 추론(Inference) 모드로 설정 ***
        print(f"모델 로드 완료 (Device: {device})")
        return model, device
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        return None, None

def get_realtime_state(api_url):
    """API 서버에 접속하여 현재 상태(풍황, 토폴로지)를 받아옵니다."""
    try:
        response = requests.get(api_url)
        response.raise_for_status() # 오류가 났으면 예외 발생
        
        data = response.json()
        wind_power = np.array(data['wind_power'])
        topology = np.array(data['topology'])
        
        # main_pandapower.py가 학습한 '상태'와 동일한 형태로 조합
        state = np.concatenate((wind_power, topology))
        return state
        
    except requests.exceptions.RequestException as e:
        print(f"API 데이터 수신 오류: {e}")
        return None

def predict_best_action(model, state, device):
    """학습된 모델을 이용해 현재 상태에서 최적의 행동을 결정합니다."""
    with torch.no_grad(): # (epsilon-greedy 탐험 없이) 순수하게 최적의 행동만 선택
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        act_values = model(state_tensor)
        
        # Q-value가 가장 높은 행동(action)을 반환
        return np.argmax(act_values.cpu().data.numpy())

def execute_action(action):
    """
    결정된 행동을 실제 제어 시스템(SCADA)에 전달합니다.
    (여기서는 시뮬레이션을 위해 화면에 출력)
    """
    
    # (실제 구현 시)
    # scada.set_breaker_state(breaker_id=action, state='toggle')
    
    print(f"  -> [제어 명령] {action}번 차단기를 조작합니다.")


def main_controller():
    print("--- 실시간 자동 제어 프로그램 (Agent Controller) ---")
    
    # 1. 학습된 모델('두뇌') 로드
    model, device = load_trained_model(MODEL_PATH, STATE_SIZE, ACTION_SIZE)
    if model is None:
        return

    # 2. 무한 루프를 돌며 실시간 제어 시작
    while True:
        print(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} - 다음 제어 사이클 시작...")
        
        # 3. API로부터 실시간 데이터(상태) 수신
        print(f"  - '{API_URL}'에서 실시간 상태 데이터를 요청합니다...")
        current_state = get_realtime_state(API_URL)
        
        if current_state is not None:
            # 4. 모델을 이용해 최적 행동 결정
            print(f"  - 현재 상태 수신 완료 (터빈 {current_state[0]:.2f}MW...)")
            best_action = predict_best_action(model, current_state, device)
            print(f"  - AI가 최적 행동을 '{best_action}' (으)로 결정했습니다.")
            
            # 5. 결정된 행동 실행
            execute_action(best_action)
            
        else:
            print("  - 데이터 수신에 실패하여 이번 사이클을 건너뜁니다.")

        # 6. 다음 제어 주기까지 대기
        print(f"  - {CONTROL_INTERVAL_SECONDS}초 후 다음 제어를 실행합니다.")
        time.sleep(CONTROL_INTERVAL_SECONDS)

if __name__ == "__main__":
    main_controller()
