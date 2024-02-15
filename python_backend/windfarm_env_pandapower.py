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
    def __init__(self, data_path):
        if not PANDAPOWER_INSTALLED:
            raise ImportError("Pandapower 라이브러리가 필요합니다. 'pip install pandapower'로 설치해주세요.")

        self.num_turbines = 40
        self.num_strings = 4
        self.turbines_per_string = 10
        self.num_breakers = self.num_strings - 1 
        
        self.max_steps = 8760 

        try:
            self.data = pd.read_csv(data_path)
            self.wind_data = self.data.iloc[:, :self.num_turbines].values
            if len(self.wind_data) < self.max_steps:
                print(f"경고: 데이터가 {self.max_steps} 스텝보다 부족합니다.")
                self.max_steps = len(self.wind_data)
        except (FileNotFoundError, pd.errors.EmptyDataError, OSError):
            print(f"'{data_path}' 경로에 파일이 없거나 비어있어 임시 더미 데이터를 생성합니다.")
            self.wind_data = np.random.rand(self.max_steps, self.num_turbines) * 5.0 

        self.state_size = self.num_turbines + self.num_breakers
        self.action_size = self.num_breakers
        
        self.net = self._create_grid_topology()
        
        self.reset()

    def _create_grid_topology(self):
        net = pp.create_empty_network(f_hz=60.0) 

        pp.create_std_type(net, name="33kV_400mm_XLPE", 
                           r_ohm_per_km=0.06, x_ohm_per_km=0.12, 
                           c_nf_per_km=150, max_i_ka=0.820, type="cs") 

        trafo_5mw_spec = {"sn_mva": 6.0, "vn_hv_kv": 33.0, "vn_lv_kv": 0.69, 
                          "vk_percent": 6.0, "vkr_percent": 0.6, 
                          "pfe_kw": 9, "i0_percent": 0.1, "shift_degree": 0}
        pp.create_std_type(net, name="5MW_Trafo", **trafo_5mw_spec, type="trafo")

        trafo_200mva_spec = {"sn_mva": 200.0, "vn_hv_kv": 220.0, "vn_lv_kv": 33.0, 
                             "vk_percent": 12.0, "vkr_percent": 0.3,
                             "pfe_kw": 50, "i0_percent": 0.05, "shift_degree": 0}
        pp.create_std_type(net, name="200MVA_Trafo", **trafo_200mva_spec, type="trafo")

        b_grid = pp.create_bus(net, vn_kv=220.0, name="External Grid")
        pp.create_ext_grid(net, bus=b_grid, vm_pu=1.00, va_degree=0.0)

        b_hv_offshore = pp.create_bus(net, vn_kv=220.0, name="Offshore Sub HV")
        b_lv_offshore = pp.create_bus(net, vn_kv=33.0, name="Offshore Sub LV")
        
        pp.create_transformer_from_parameters(net, hv_bus=b_grid, lv_bus=b_lv_offshore, 
                                              sn_mva=200.0, vn_hv_kv=220.0, vn_lv_kv=33.0,
                                              vkr_percent=0.3, vk_percent=12.0,
                                              pfe_kw=50, i0_percent=0.05)

        self.sgen_indices = [] 
        self.breaker_indices = [] 
        last_string_head_bus = None

        for s in range(self.num_strings):
            string_head_bus = pp.create_bus(net, vn_kv=33.0, name=f"String {s+1} Head")
            
            pp.create_line(net, from_bus=b_lv_offshore, to_bus=string_head_bus, 
                           length_km=0.5, std_type="33kV_400mm_XLPE")
            
            if last_string_head_bus is not None:
                sw_bus = pp.create_bus(net, vn_kv=33.0)
                pp.create_line(net, from_bus=last_string_head_bus, to_bus=sw_bus, length_km=0.1, std_type="33kV_400mm_XLPE")
                sw_id = pp.create_switch(net, bus=sw_bus, element=net.line.index[-1], et='l', closed=False)
                self.breaker_indices.append(sw_id)

            last_string_head_bus = string_head_bus
            last_33kv_bus = string_head_bus

            for t in range(self.turbines_per_string):
                b_069kv = pp.create_bus(net, vn_kv=0.69, name=f"T_{s*10+t+1}_0.69kV")
                b_33kv = pp.create_bus(net, vn_kv=33.0, name=f"T_{s*10+t+1}_33kV")

                sgen_id = pp.create_sgen(net, bus=b_069kv, p_mw=0.0, q_mvar=0.0, sn_mva=5.0)
                self.sgen_indices.append(sgen_id)

                pp.create_transformer(net, hv_bus=b_33kv, lv_bus=b_069kv, std_type="5MW_Trafo")
                
                pp.create_line(net, from_bus=last_33kv_bus, to_bus=b_33kv, 
                               length_km=0.5, std_type="33kV_400mm_XLPE")
                last_33kv_bus = b_33kv
        
        return net

    def reset(self):
        self.current_step = 0
        
        self.topology = np.zeros(self.num_breakers)
        for sw_id in self.breaker_indices:
            self.net.switch.closed[sw_id] = False
            
        initial_wind = self.wind_data[self.current_step]
        for i, sgen_id in enumerate(self.sgen_indices):
            self.net.sgen.p_mw[sgen_id] = initial_wind[i]

        state = np.concatenate((initial_wind, self.topology))
        return state

    def step(self, action):
        breaker_id = self.breaker_indices[action]
        current_state = self.net.switch.closed[breaker_id]
        self.net.switch.closed[breaker_id] = not current_state
        self.topology[action] = 1.0 if not current_state else 0.0

        self.current_step += 1
        done = self.current_step == self.max_steps - 1

        next_wind = self.wind_data[self.current_step]
        for i, sgen_id in enumerate(self.sgen_indices):
            self.net.sgen.p_mw[sgen_id] = next_wind[i]

        try:
            pp.runpp(self.net, algorithm='nr', numba=True, enforce_q_lims=False, enforce_v_lims=False)
            calculation_success = True
        except Exception as e:
            calculation_success = False

        reward = self.calculate_reward_from_net(calculation_success)
        
        next_state = np.concatenate((next_wind, self.topology))
        
        return next_state, reward, done

    def calculate_reward_from_net(self, calculation_success):
        if not calculation_success:
            return -10000.0 

        total_generation = self.net.res_sgen.p_mw.sum()
        total_load_at_grid = self.net.res_ext_grid.p_mw.sum() 
        
        p_loss_mw = total_generation + total_load_at_grid 
        
        v_pu = self.net.res_bus.vm_pu
        v_min_penalty = np.sum(np.maximum(0, 0.95 - v_pu)) 
        v_max_penalty = np.sum(np.maximum(0, v_pu - 1.05)) 
        v_penalty = (v_min_penalty + v_max_penalty) * 100 

        loading_pct = self.net.res_line.loading_percent
        overload_penalty = np.sum(np.maximum(0, loading_pct - 100.0)) 
        
        switching_cost = 5.0 

        total_cost = (p_loss_mw * 10) + v_penalty + overload_penalty + switching_cost
        
        return -total_cost
# 24/02/15 Init Env 
# 24/02/21 Refine Env 
# 24/03/06 Fix reward 
# 24/03/26 Refactor env 
# 24/04/11 Fix env step 
# 24/04/29 Cleanup env 
# 24/05/10 Fix state size 
# 24/05/21 Add docs 
# 24/06/05 Cleanup 
# 24/06/27 Final doc 
# 24/02/15 Init Env 
