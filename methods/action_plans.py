import numpy as np


def generate_action_plan(N, K=3, abs_value=90, time=2/7):
    split = time*np.ones((2, K))

    split[1, :] = ( 1 - K*time) / K

    split_units = (N * split).astype(int)
    split_units
    done_counts = np.zeros(split_units.shape)

    actions = np.empty(N)
    actions[:] = np.nan
    values = np.zeros((N, K))

    for i in range(N):
        for j in range(K):
            if done_counts[0, j] < split_units[0, j]:
                actions[i] = j
                values[i, j] = abs_value if done_counts[0, j] < split_units[0, j] / 2 else - abs_value
                done_counts[0, j] += 1
                break
            elif done_counts[1, j] < split_units[1, j]:
                done_counts[1, j] += 1
                break

    # Add offset
    offset = int((N * ( 1 - K*time) / K ) / 2)
    actions = np.roll(actions, offset)
    values = np.roll(values, offset, axis=0)

    return actions, values