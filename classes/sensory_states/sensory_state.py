import numpy as np

class Sensory_state():
    def __init__(self, N, K, observe_func, observe_func_args=[]):
        self._N = N
        self._n = 0
        self._K = K

        # p(s_t|e_t): must be a function of external and internal states
        self._p_s_g_e = observe_func
        self._p_s_g_e_params = observe_func_args

        self._observations = np.zeros((N+1, K))

        self._obs_alt_record = False
        self._observations_alt = np.zeros((N+1, K))

    
    def observe(self, external_state, internal_state):
        obs, obs_alt = self._p_s_g_e(external_state, internal_state, *self._p_s_g_e_params)
        self._n += 1
        self._observations[self._n] = obs
        if type(obs_alt) == np.ndarray:
            self._observations_alt[self._n] = obs_alt
        return obs

    # Used mostly for action selection
    def rollback(self, back=np.Inf):
        if back > self._N or back > self._n:
            self._n = 0
        else:
            self._n -= int(back)

        # Roll back observations   
        self._observations[self._n+1:, :] = 0  
        self._observations_alt[self._n+1:, :] = 0
    
    @property
    def s(self):
        return self._observations[self._n]

    @property
    def s_prev(self):
        return self._observations[self._n - 1]

    @property
    def obs(self):
        return self._observations[:self._n, :]

    @property
    def s_alt(self):
        if not self._obs_alt_record:
            return None
        else:
            return self._observations_alt[self._n]

    @property
    def s_alt_prev(self):
        if not self._obs_alt_record:
            return None
        else:
            return self._observations_alt[self._n - 1]

    @property
    def obs_alt(self):
        if not self._obs_alt_record:
            return None
        else:
            return self._observations_alt[:self._n, :]

