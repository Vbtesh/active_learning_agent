from classes.sensory_states.sensory_state import Sensory_state

# Oniscient sensor
## Returns extactly the external states
class Omniscient_ST(Sensory_state):
    def __init__(self, N, K, gamma=0.5):
        super().__init__(N, K, self.omniscient_observation)
        self._gamma = gamma
        self._obs_alt_record = True

    def omniscient_observation(self, external_state, internal_state):
        alt = self.s_alt
        change_update = (external_state.x - self.s) + self._gamma * self.s_alt
        return external_state.x, change_update