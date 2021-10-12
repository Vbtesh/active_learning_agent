import numpy as np

# Simplest omniscient sensor, return the external states
def omniscient_sensor(external_state, internal_state):
    return external_state.x