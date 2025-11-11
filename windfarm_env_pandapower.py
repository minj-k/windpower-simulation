import numpy as np
import pandas as pd
import random
try:
    import pandapower as pp
    import pandapower.converter as pc
    PANDAPOWER_INSTALLED = True
except ImportError:
    PANDAPOWER_INSTALLED = False

class WindFarmEnvPandapower:
    """
    Pandapower 기반 고충실도 해상풍력발전단지 환경 시뮬레이터
    - 논문의 그림 1과 수식 (10)을 기반으로 물리적 계통을 모델링
    - 전력 조류 계산(Newton-Raphson)을 통해 실제 손실, 전압, 부하율을 계산
    """
    def __init__(self, data_path):
        if not PANDAPOWER_INSTALLED:
            raise ImportError("Pandapower 라이브러리가 필요합니다. 'pip install pandapower'로 설치해주세요.")

        self.num_turbines = 40
        self.num_strings = 4
        self.turbines_per_string = 10
        self.num_breakers = self.num_strings - 1 # 3 (String 1-2, 2-3, 3-4 연결)
        
        self.max_steps = 8760 # 1년 (시간 단위)

        # 1. 시계열 데이터 로드 (풍력 발전량)
        try:
            self.data = pd.read_csv(data_path)
            self.wind_data = self.data.iloc[:, :self.num_turbines].values
            if len(self.wind_data) < self.max_steps:
                print(f"경고: 데이터가 {self.max_steps} 스텝보다 부족합니다.")
                self.max_steps = len(self.wind_data)
        except (FileNotFoundError, pd.errors.EmptyDataError, OSError):
            print(f"'{data_path}' 경로에 파일이 없거나 비어있어 임시 더미 데이터를 생성합니다.")
            self.wind_data = np.random.rand(self.max_steps, self.num_turbines) * 5.0 # 0~5MW

        # 상태: (터빈 발전량 40개 + 차단기 상태 3개)
        self.state_size = self.num_turbines + self.num_breakers
        # 행동: 3개의 차단기 중 하나를 조작
        self.action_size = self.num_breakers
        
        # 2. Pandapower 계통 생성
        self.net = self._create_grid_topology()
        
        self.reset()

    def _create_grid_topology(self):
        """
        논문(그림 1) 기반으로 Pandapower 계통 모델 생성
        - 40 터빈 (0.69kV) -> 40 변압기 -> 4 스트링 (33kV) -> 해상변전소 (33/220kV) -> 육지 (220kV)
        """
        net = pp.create_empty_network(f_hz=60.0) # 60Hz 계통

        # --- 1. 전기 파라미터 정의 (중요: 이 값들은 가정치이며 실제 데이터로 교체 필요) ---
        
        # 33kV 400mm^2 XLPE 해저 케이블 (논문 참조)
        # (가정치: km당 R=0.06 옴, X=0.12 옴, 최대 허용 전류 820A)
        pp.create_std_type(net, name="33kV_400mm_XLPE", 
                           r_ohm_per_km=0.06, x_ohm_per_km=0.12, 
                           c_nf_per_km=150, max_i_ka=0.820, type="cs") # cs=cable system

        # 5MW 터빈 승압 변압기 (0.69kV / 33kV)
        # (가정치: 6MVA, 6% 임피던스)
        trafo_5mw_spec = {"sn_mva": 6.0, "vn_hv_kv": 33.0, "vn_lv_kv": 0.69, 
                          "vk_percent": 6.0, "vkr_percent": 0.6, 
                          "pfe_kw": 9, "i0_percent": 0.1, "shift_degree": 0}
        pp.create_std_type(net, name="5MW_Trafo", **trafo_5mw_spec, type="trafo")

        # 200MVA 해상 변전소 변압기 (33kV / 220kV)
        # (가정치: 2x 100MVA = 200MVA, 12% 임피던스)
        trafo_200mva_spec = {"sn_mva": 200.0, "vn_hv_kv": 220.0, "vn_lv_kv": 33.0, 
                             "vk_percent": 12.0, "vkr_percent": 0.3,
                             "pfe_kw": 50, "i0_percent": 0.05, "shift_degree": 0}
        pp.create_std_type(net, name="200MVA_Trafo", **trafo_200mva_spec, type="trafo")

        # --- 2. 모선(Bus) 생성 ---
        # 육지 계통 (Slack Bus: 전압과 위상의 기준점)
        b_grid = pp.create_bus(net, vn_kv=220.0, name="External Grid")
        pp.create_ext_grid(net, bus=b_grid, vm_pu=1.00, va_degree=0.0)

        # 해상 변전소 (HV/LV)
        b_hv_offshore = pp.create_bus(net, vn_kv=220.0, name="Offshore Sub HV")
        b_lv_offshore = pp.create_bus(net, vn_kv=33.0, name="Offshore Sub LV")
        
        # 해상 변전소 변압기 연결
        pp.create_transformer_from_parameters(net, hv_bus=b_grid, lv_bus=b_lv_offshore, 
                                              sn_mva=200.0, vn_hv_kv=220.0, vn_lv_kv=33.0,
                                              vkr_percent=0.3, vk_percent=12.0,
                                              pfe_kw=50, i0_percent=0.05)
        # (논문 그림대로라면 20km 220kV 케이블이 있으나, 여기선 변전소를 바로 육지 그리드에 연결)

        # --- 3. 터빈 및 내부망 생성 (4 스트링) ---
        self.sgen_indices = [] # 터빈 발전기(sgen) ID 저장
        self.breaker_indices = [] # 차단기(switch) ID 저장
        last_string_head_bus = None

        for s in range(self.num_strings):
            string_head_bus = pp.create_bus(net, vn_kv=33.0, name=f"String {s+1} Head")
            
            # 스트링 헤드(Head)를 해상변전소 LV 모선에 연결 (0.5km 가정)
            pp.create_line(net, from_bus=b_lv_offshore, to_bus=string_head_bus, 
                           length_km=0.5, std_type="33kV_400mm_XLPE")
            
            # (차단기) 이전 스트링과 현재 스트링 연결 (Breaker)
            if last_string_head_bus is not None:
                # 차단기는 'line'으로 모델링하되, 'switch'로 제어
                sw_bus = pp.create_bus(net, vn_kv=33.0)
                pp.create_line(net, from_bus=last_string_head_bus, to_bus=sw_bus, length_km=0.1, std_type="33kV_400mm_XLPE")
                # 'closed=False' (기본 개방 상태), et='l' (line switch)
                sw_id = pp.create_switch(net, bus=sw_bus, element=net.line.index[-1], et='l', closed=False)
                self.breaker_indices.append(sw_id)

            last_string_head_bus = string_head_bus
            last_33kv_bus = string_head_bus

            # 스트링 내부 터빈 10개 생성 (논문 그림 참조)
            for t in range(self.turbines_per_string):
                b_069kv = pp.create_bus(net, vn_kv=0.69, name=f"T_{s*10+t+1}_0.69kV")
                b_33kv = pp.create_bus(net, vn_kv=33.0, name=f"T_{s*10+t+1}_33kV")

                # 터빈 발전기 (Static Generator)
                sgen_id = pp.create_sgen(net, bus=b_069kv, p_mw=0.0, q_mvar=0.0, sn_mva=5.0)
                self.sgen_indices.append(sgen_id)

                # 터빈 변압기
                pp.create_transformer(net, hv_bus=b_33kv, lv_bus=b_069kv, std_type="5MW_Trafo")
                
                # 터빈 간 케이블 (논문: x축 0.5km)
                pp.create_line(net, from_bus=last_33kv_bus, to_bus=b_33kv, 
                               length_km=0.5, std_type="33kV_400mm_XLPE")
                last_33kv_bus = b_33kv
        
        return net

    def reset(self):
        self.current_step = 0
        
        # 1. 토폴로지(차단기) 초기화 (모두 개방)
        self.topology = np.zeros(self.num_breakers)
        for sw_id in self.breaker_indices:
            self.net.switch.closed[sw_id] = False
            
        # 2. 발전량 초기화
        initial_wind = self.wind_data[self.current_step]
        for i, sgen_id in enumerate(self.sgen_indices):
            self.net.sgen.p_mw[sgen_id] = initial_wind[i]
            # (무효전력(q_mvar)은 0으로 설정, 전압 제어는 slack bus가 담당)

        # 3. 초기 상태 반환
        state = np.concatenate((initial_wind, self.topology))
        return state

    def step(self, action):
        # 1. 행동(차단기 조작) 수행
        breaker_id = self.breaker_indices[action]
        current_state = self.net.switch.closed[breaker_id]
        self.net.switch.closed[breaker_id] = not current_state
        self.topology[action] = 1.0 if not current_state else 0.0

        self.current_step += 1
        done = self.current_step == self.max_steps - 1

        # 2. 다음 시간대 발전량 업데이트
        next_wind = self.wind_data[self.current_step]
        for i, sgen_id in enumerate(self.sgen_indices):
            self.net.sgen.p_mw[sgen_id] = next_wind[i]

        # 3. *** 전력 조류 계산 (Newton-Raphson) ***
        try:
            pp.runpp(self.net, algorithm='nr', numba=True, enforce_q_lims=False, enforce_v_lims=False)
            calculation_success = True
        except Exception as e:
            # (예: 과부하, 전압 붕괴 등으로 조류 계산 실패 시)
            # print(f"Warning: Power flow failed at step {self.current_step}. {e}")
            calculation_success = False

        # 4. 보상 계산
        reward = self.calculate_reward_from_net(calculation_success)
        
        # 5. 다음 상태 반환
        next_state = np.concatenate((next_wind, self.topology))
        
        return next_state, reward, done

    def calculate_reward_from_net(self, calculation_success):
        """
        Pandapower 계산 결과를 바탕으로 논문(수식 10) 기반의 보상 계산
        """
        
        # 조류 계산 실패 시 (치명적 페널티)
        if not calculation_success:
            return -10000.0 # 매우 큰 음수 보상

        # 1. P_loss (유효전력 손실)
        total_generation = self.net.res_sgen.p_mw.sum()
        total_load_at_grid = self.net.res_ext_grid.p_mw.sum() # 음수로 나옴
        
        # 총 손실 = (총 발전량) - (육지로 송전된 량)
        p_loss_mw = total_generation + total_load_at_grid 
        
        # 2. 전압 제약 (Voltage Constraints)
        v_pu = self.net.res_bus.vm_pu
        v_min_penalty = np.sum(np.maximum(0, 0.95 - v_pu)) # 0.95 p.u. 이하
        v_max_penalty = np.sum(np.maximum(0, v_pu - 1.05)) # 1.05 p.u. 이상
        v_penalty = (v_min_penalty + v_max_penalty) * 100 # (가중치 100)

        # 3. 선로 용량 제약 (Loading Constraints)
        loading_pct = self.net.res_line.loading_percent
        overload_penalty = np.sum(np.maximum(0, loading_pct - 100.0)) # 100% 초과
        
        # 4. 차단기 조작 비용 (Switching Cost)
        switching_cost = 5.0 # (가정치: 1회 조작 비용 5.0)

        # 최종 보상 (논문 수식 10과 유사하게, 비용 최소화 = 보상 최대화)
        # (w1 * P_loss) + (w2 * V_penalty) + (w3 * Overload_penalty) + (w5 * C)
        total_cost = (p_loss_mw * 10) + v_penalty + overload_penalty + switching_cost
        
        return -total_cost
