import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_training_results(filepath="episode_rewards.npy", window_size=100):
    """
    저장된 'episode_rewards.npy' 파일을 읽어 학습 결과 그래프를 그립니다.
    
    Args:
        filepath (str): 저장된 보상 데이터 파일 경로
        window_size (int): 이동 평균을 계산할 윈도우 크기
    """
    
    # 1. 파일이 있는지 확인
    if not os.path.exists(filepath):
        print(f"오류: '{filepath}' 파일을 찾을 수 없습니다.")
        print("먼저 'main_pandapower.py'를 실행하여 학습 결과를 저장해야 합니다.")
        return

    # 2. 데이터 로드
    try:
        rewards = np.load(filepath)
    except Exception as e:
        print(f"파일을 로드하는 중 오류가 발생했습니다: {e}")
        return

    print(f"총 {len(rewards)} 에피소드의 데이터를 로드했습니다.")

    # 3. 이동 평균 계산 (그래프를 부드럽게 보기 위함)
    if len(rewards) >= window_size:
        moving_avg = pd.Series(rewards).rolling(window=window_size).mean()
    else:
        # 데이터가 윈도우 크기보다 작으면 이동 평균을 계산하지 않음
        moving_avg = pd.Series(rewards) # 원본 데이터 사용

    # 4. 그래프 그리기
    plt.figure(figsize=(12, 7))
    
    # 원본 데이터 (흐리게)
    plt.plot(rewards, color='blue', alpha=0.3, label="Raw Episode Reward")
    
    # 이동 평균 데이터 (진하게)
    plt.plot(moving_avg, color='red', linewidth=2, label=f"Moving Average (window={window_size})")
    
    # 그래프 설정
    plt.title("DQN Training Progress (학습 진행도)", fontsize=16)
    plt.xlabel("Episode (에피소드)", fontsize=12)
    plt.ylabel("Total Reward (총 보상)", fontsize=12)
    plt.legend()
    plt.grid(True)
    
    # 한글 폰트 설정 (Windows, Mac, Linux 순서)
    # 폰트가 없는 경우, 깨짐 방지를 위해 영어로만 표시
    try:
        plt.rcParams['font.family'] = 'Malgun Gothic' # Windows
        plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호
    except:
        try:
            plt.rcParams['font.family'] = 'AppleGothic' # Mac
            plt.rcParams['axes.unicode_minus'] = False 
        except:
            pass # (기본 폰트 사용)

    print("그래프를 표시합니다...")
    plt.show()

if __name__ == "__main__":
    plot_training_results()
