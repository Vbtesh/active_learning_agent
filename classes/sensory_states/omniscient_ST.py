from classes.sensory_states.sensory_state import Sensory_state

# Oniscient sensor
## Returns extactly the external states
class Omniscient_ST(Sensory_state):
    def __init__(self, N, K):
        super().__init__(N, K, self.omniscient_observation)

    def omniscient_observation(self, external_state, internal_state):
        return external_state.x