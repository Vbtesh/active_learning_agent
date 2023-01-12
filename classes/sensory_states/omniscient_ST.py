from classes.sensory_states.sensory_state import Sensory_state
import numpy as np

# Oniscient sensor
## Returns extactly the external states

# Free parameter:
## alpha: change "smoothing" rate, represents the memory of recent change and serves to smooth out one step variance or noise from the data

class Omniscient_ST(Sensory_state):
    def __init__(self, N, K, noise_std=None, change_memory=0.5, change='relative', value_range=(-100, 100)):
        super().__init__(N, K, self.omniscient_observation)
        self._alpha = change_memory
        self._obs_alt_record = True

        self.change_summary = change
        
        if change == 'relative':
            self._change_function = self._relative_change
            self.alt_range = (-5, 5) # Simply for plotting purposes
        elif change == 'normalised':
            self._change_function = self._normalised_change
            self.alt_range = (value_range[0]/100, value_range[1]/100) # Simply for plotting purposes
        else:
            self._change_function = self._raw_change
            self.alt_range = (value_range[0]/4, value_range[1]/4) # Simply for plotting purposes


        if noise_std:
            self._noisy = 1
            self._noise_std = noise_std
        else:
            self._noisy = 0
            self._noise_std = 1


    def omniscient_observation(self, external_state, internal_state):
        change_update = self._change_function(external_state)
        noisy_obs = self._noisy_observation(external_state)
        return noisy_obs, change_update


    def _noisy_observation(self, external_state):
        return external_state.x + self._noisy * np.random.normal(scale=self._noise_std)


    def _raw_change(self, external_state):
        sense = self.s
        change_update = self.s_alt + self._alpha * ((external_state.x - sense) - self.s_alt)
        return change_update


    def _relative_change(self, external_state):
        sense = self.s
        if np.sum(self.s == 0):
            sense[self.s == 0] = 1
        change_update = self.s_alt + self._alpha * ((external_state.x - sense)/sense - self.s_alt)
        return change_update


    def _normalised_change(self, external_state):
        sense = self.s
        bound = external_state._range[1]
        change_update = self.s_alt + self._alpha * ((external_state.x - sense)/bound - self.s_alt)
        return change_update