import numpy as np

"""
Returns a specific hard coded action plan
/!\ the relevancy of the action plan will depend on the ground truth model /!\
Options:
    - mid_single_swipe__90__1/3: will perform a single swipe at 90 on one variable for 1/3 of the time in the middle of a trial
    - double_swipe__90__1/6: single swipe at 1/3 and another at 2/3
"""
def fetch_action_plan(use_action_plan, N, K=3):
    if use_action_plan.split('__')[0] == 'mid_single_swipe':
        value = float(use_action_plan.split('__')[1])
        time = float(use_action_plan.split('__')[2].split('/')[0]) / float(use_action_plan.split('__')[2].split('/')[1])
        actions = np.empty(N)
        actions[:] = np.nan
        actions[int(N/2):int(N/2+time*N)] = 0

        values = np.zeros((N, K))
        for k in range(K):
            a = np.where(actions == k)[0]
            values[a, k] = value

    elif use_action_plan.split('__')[0] == 'double_swipe':
        value = float(use_action_plan.split('__')[1])
        time = float(use_action_plan.split('__')[2].split('/')[0]) / float(use_action_plan.split('__')[2].split('/')[1])

        time_1_start = int(N/3)
        time_1_end = time_1_start + int(time*N)

        time_2_start = int(N*2/3)
        time_2_end = time_2_start + int(time*N)
        
        actions = np.empty(N)
        actions[:] = np.nan
        actions[time_1_start:time_1_end] = 0
        actions[time_2_start:time_2_end] = 1

        values = np.zeros((N, K))
        for k in range(K):
            a = np.where(actions == k)[0]
            values[a, k] = value
        
    
    return actions, values


"""
Generates a standardised action plan
abs_value defines the target value of the interventions, default is {-90, 90}
time defines the proportion of time intervened

It will always split the intervened time between all variable and will do the same with the time spend in the positive or negative range
"""

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