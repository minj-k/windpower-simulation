from flask import Flask, jsonify
import numpy as np

app = Flask(__name__)

@app.route('/wind_data')
def get_wind_data():
    """
    현재 시간의 풍력 발전량(40개)과 차단기 상태(3개)를 반환합니다.
    (현재는 랜덤 데이터로 시뮬레이션)
    """
    try:
        # 1. 40개 터빈의 현재 발전량 (0~5MW 사이 랜덤)
        current_wind_power = (np.random.rand(40) * 5.0).tolist()
        
        # 2. 3개 차단기의 현재 상태 (0 또는 1 랜덤)
        current_topology = (np.random.randint(0, 2, 3)).tolist()
        
        # 3. JSON 형식으로 상태 반환
        state_data = {
            "wind_power": current_wind_power,
            "topology": current_topology
        }
        return jsonify(state_data)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("실시간 데이터 모의 API 서버를 시작합니다...")
    print("Agent Controller가 http://127.0.0.1:5000/wind_data 로 접속할 것입니다.")
    # Flask 라이브러리가 필요합니다. (pip install Flask)
    app.run(host='127.0.0.1', port=5000)
# 24/03/12 Init API 
# 24/03/18 Add data endpoint 
# 24/04/05 Fix API data 
# 24/04/19 Add docs 
# 24/05/07 Update API random 
# 24/06/03 Cleanup 
# 24/03/12 Init API 
# 24/03/16 (Sat) Add data endpoint 
# 24/04/05 Fix API data 
# 24/04/17 Add docs 
# 24/05/07 Update API random 
# 24/06/01 (Sat) Cleanup 
# 24/03/12 Init API 
# 24/03/16 (Sat) Add data endpoint 
# 24/04/05 Fix API data 
# 24/04/17 Add docs 
# 24/05/07 Update API random 
