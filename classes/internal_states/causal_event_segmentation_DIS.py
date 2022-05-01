from classes.internal_states.internal_state import Discrete_IS
from scipy import stats
import numpy as np


# Causal event segmentation model

class causal_event_segmentation_DIS(Discrete_IS):
    def __init__(self, N, K, links, dt, abs_bounds, ces_type, ce_threshold=0.5, time_threshold = 15, guess=0.1, generate_sample_space=True, sample_params=False, prior_param=None, smoothing=False):
        super().__init__(N, K, links, dt, self._update_rule, generate_sample_space=generate_sample_space, sample_params=sample_params, prior_param=prior_param, smoothing=smoothing)

        self._bounds = abs_bounds

        # Threshold for event detection
        # Given by a percentage of the full range
        self._causal_event_threshold = ce_threshold * abs_bounds[1]

        # Time aspect
        self._time_threshold = time_threshold
            
        self._type_model = ces_type
        

        # Probability mass to be split between non causal event segmentation predictions
        self._guess = guess

        self._last_action = None
        self._last_action_len = None
        self._last_instant_action = None
        self._last_real_action = None
        self._last_obs = np.zeros(self._K)
        self._last_action_idx = 0




    def _update_rule(self, sensory_state, action_state):
        # Get current action from action state
        ## Beware: if not a_fit but a_real, this can lead to divergence because the model will interpret an intervention as normal data
        ## Maybe use a_real rather that a_fit
        intervention = action_state.a
        
        obs = sensory_state.s
        obs_alt = sensory_state.s_alt 
        

        # Action started but not learnable action
        # If fitting, check between fit and real action
        if action_state.realised:
            # If fitting, check between fit and real action
            if (not self._last_action and not action_state.a_real) or self._n == 0:
                self._last_obs = obs
                self._last_instant_action = action_state.a_real
                return self._posterior_params

            elif not self._last_action and action_state.a_real:
                # First action
                self._last_action_len = action_state.a_len_real      
                # Reset last action index
                self._last_action_idx = 0

                self._last_action = action_state.a_real

                self._last_action_idx += 1
                self._last_obs = obs
                self._last_instant_action = action_state.a_real
                

            elif self._last_action and action_state.a_real:
                if not self._last_instant_action:
                    # CHANGE OF ACTION
                    # Action length is
                    self._last_action_len = action_state.a_len_real      
                    # Reset last action index
                    self._last_action_idx = 0

                    self._last_action = action_state.a_real

                    self._last_action_idx += 1
                    self._last_obs = obs
                    self._last_instant_action = action_state.a_real
                        
                else:
                    self._last_action_idx += 1
                    self._last_obs = obs
                    self._last_instant_action = action_state.a_real
                        
        else:
            # If generating data
            if (not self._last_action and not intervention) or self._n == 0:
                self._last_obs = obs
                self._last_instant_action = intervention
                return self._posterior_params
            elif not self._last_action and intervention:
                # Action length is
                self._last_action_len = action_state.a_len     
                # Reset last action index
                self._last_action_idx = 0

                self._last_action = intervention
                self._last_instant_action = intervention
            elif self._last_action and intervention:
                if not self._last_instant_action:
                    # Action length is
                    self._last_action_len = action_state.a_len     
                    # Reset last action index
                    self._last_action_idx = 0

                    self._last_action = intervention
                    self._last_instant_action = intervention
    

        ## Get change
        ## Update evidence or values (posterior value)
        one_step_evidence = np.zeros(self._posterior_params.shape)
        idx = 0
        for i in range(self._K):
            for j in range(self._K):
                if i != j:
                    if j == self._last_action[0] or i != self._last_action[0]:
                        # If looking a link going into cause i or if looking at link that are not acted upon
                        idx += 1
                        continue

                    
                    if np.abs(obs[j]) > self._causal_event_threshold:
                        if self._type_model == 'strength_sensitive':
                            if self._last_action_idx < self._time_threshold:
                                # Accumulate strong link evidence 
                                if obs[j] * obs[i] > 1:
                                    # Positive link
                                    link_value = 1
                                    one_step_evidence[idx, :] = (self._L == link_value).astype(int)
                                else:
                                    # Negative link
                                    link_value = -1
                                    one_step_evidence[idx, :] = (self._L == link_value).astype(int)
                            else:
                                # Accumulate weak link evidence
                                if obs[j] * obs[i] > 1:
                                    # Positive link
                                    link_value = 1/2
                                    one_step_evidence[idx, :] = (self._L == link_value).astype(int)
                                else:
                                    # Negative link
                                    link_value = -1/2
                                    one_step_evidence[idx, :] = (self._L == link_value).astype(int)

                        elif self._type_model == 'strength_insensitive':
                            if obs[j] * obs[i] > 1:
                                one_step_evidence[idx, :] = (self._L == 1/2).astype(int)
                                one_step_evidence[idx, :] = (self._L == 1).astype(int)
                            else: 
                                one_step_evidence[idx, :] = (self._L == -1).astype(int)
                                one_step_evidence[idx, :] = (self._L == -1/2).astype(int)
                    
                    idx += 1
                    
        
        # Update evidence collected
        self._evidence_collected[self._n, :, :] = one_step_evidence
        # posterior_params: sum over evidence collected
        posterior_params = np.nanmean(self._evidence_collected, axis=0)
        posterior_params = posterior_params / posterior_params.sum(axis=1, keepdims=1)

        # update mus
        self._last_action_idx += 1
        self._last_obs = obs

        if action_state.realised:
            self._last_instant_action = action_state.a_real
        else:
            self._last_instant_action = intervention

        return posterior_params
        

    
    # Background methods
    ## Prior initialisation specific to model:
    def _local_prior_init(self):  
        self._evidence_collected = np.zeros((self._N, self._prior_params.shape[0], self._prior_params.shape[1]))

        self._evidence_collected[0, :, : ] = self._prior_params

    @property
    def posterior(self):
        if self._smoothing_temp:
            smoothed_posterior = self._smooth_softmax(self._posterior_params)
            return smoothed_posterior
        else:
            guess_posterior = self._likelihood(self._posterior_params)
            return guess_posterior
    

    def _likelihood(self, posterior_params):
        mask = (np.amax(posterior_params, axis=1, keepdims=1) == posterior_params).astype(bool)

        lh = np.zeros(posterior_params.shape)
        prob = 1 - self._guess

        if mask.sum() == mask.shape[0]:
            lh[mask] = prob
            lh[~mask] = self._guess / (lh.shape[1] - 1)
        elif mask.sum() < mask.size:
            mask2 = mask.sum(axis=1) == mask.shape[1]

            prob_mass = prob / mask.sum(axis=1, keepdims=1)
            lh[~mask2] += mask[~mask2] * prob_mass[~mask2]
            # If not true in one row in ~mask, then problems
            lh[~mask2] += ~mask[~mask2] * (self._guess / (~mask[~mask2]).sum(axis=1, keepdims=1))

            lh[mask2] = 1

            lh = lh / lh.sum(axis=1, keepdims=1)
        else:
            lh += mask / mask.sum(axis=1, keepdims=1)

        return lh

    def _generate_prior_from_judgement(self, prior_judgement, parameter):

        if type(prior_judgement) == np.ndarray and parameter > 0:
            prior_j = prior_judgement  
            param = parameter 
        else:
            #prior_j = np.zeros((self._K**2 - self._K, self._L.size))

            #return prior_j
            prior_j = np.zeros((self._K**2 - self._K))
            param = 1

        # Create mask
        prior = np.zeros((self._K**2 - self._K, self._L.size))

        for i in np.arange(prior_j.size):
            v = prior_j[i]
            prior[i, :] = param * (self._L == v).astype(int)

        
        return prior / prior.sum(axis=1, keepdims=1)


        





