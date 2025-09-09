""""""  		  	   		 	 	 			  		 			     			  	 
"""  		  	   		 	 	 			  		 			     			  	 
Template for implementing QLearner  (c) 2015 Tucker Balch  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	 	 			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		 	 	 			  		 			     			  	 
All Rights Reserved  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Template code for CS 4646/7646  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	 	 			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		 	 	 			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		 	 	 			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		 	 	 			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	 	 			  		 			     			  	 
or edited.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		 	 	 			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		 	 	 			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	 	 			  		 			     			  	 
GT honor code violation.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
-----do not edit anything above this line---  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Student Name: Yulian Lee Ying Hern		  	   		 	 	 			  		 			     			  	 
GT User ID: yhern3		  	   		 	 	 			  		 			     			  	 
GT ID: 903870865		  	   		 	 	 			  		 			     			  	 
"""  		  	   		 	 	 			  		 			     			  	 

import numpy as np
import random

class QLearner:
    def __init__(self,
                 num_states=100,
                 num_actions=4,
                 alpha=0.2,
                 gamma=0.9,
                 rar=0.5,
                 radr=0.99,
                 dyna=0,
                 verbose=False):
        self.verbose = verbose
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna

        self.q_table = np.zeros((num_states, num_actions))

        self.t_counts = np.full((num_states, num_actions, num_states), 1e-5) # intialize transitions table
        self.r_table = np.zeros((num_states, num_actions)) # initialize reward table

        self.s = 0
        self.a = 0

    def querysetstate(self, s):
        self.s = s
        self.a = self._get_action(s)
        return self.a

    def query(self, s_prime, r):
        best_q_next = np.max(self.q_table[s_prime])
        td_error = r + self.gamma * best_q_next - self.q_table[self.s, self.a]
        self.q_table[self.s, self.a] += self.alpha * td_error
        
        if self.dyna > 0:
            self.t_counts[self.s, self.a, s_prime] += 1
            self.r_table[self.s, self.a] += self.alpha * (r - self.r_table[self.s, self.a])
            
            self._run_dyna_q()
        
        action = self._get_action(s_prime)

        self.s = s_prime
        self.a = action
        self.rar *= self.radr

        return action
    
    def _get_action(self, state):
        if random.random() < self.rar:
            return random.randint(0, self.num_actions - 1)
        return np.argmax(self.q_table[state])
    
    def _run_dyna_q(self):
        if self.dyna <= 0:
            return

        """
        Imagine the t_counts table being initialized as
        t_counts = [
                    [[1e-5, 1e-5, 1e-5],  # s=0, a=0
                    [[1e-5, 1e-5, 1e-5]], # s=0, a=1
                    [[1e-5, 1e-5, 1e-5],  # s=1, a=0
                    [[1e-5, 1e-5, 1e-5]], # s=1, a=1
                    [[1e-5, 1e-5, 1e-5],  # s=2, a=0
                    [[1e-5, 1e-5, 1e-5]], # s=2, a=1
                    ]
        
        After some interactions, suppose t_counts[0, 0] = [1, 2, 3] 
        (i.e., from state 0, action 0, the agent transitioned to state 0 once, state 1 twice, and state 2 three times)

        The transition probabilities for (s=0, a=0) are calculated as:
        transitions[0, 0] = [1, 2, 3] / (1 + 2 + 3) = [1/6, 2/6, 3/6]
        """

        transitions = self.t_counts / np.sum(self.t_counts, axis=2, keepdims=True)

        for _ in range(self.dyna):
            s = random.randint(0, self.num_states - 1) # sample random state
            a = random.randint(0, self.num_actions - 1) # sample random action
            s_prime = np.argmax(transitions[s, a, :])
            r = self.r_table[s, a] 

            best_q_next = np.max(self.q_table[s_prime])
            td_error = r + self.gamma * best_q_next - self.q_table[s, a]
            self.q_table[s, a] += self.alpha * td_error


    def author(self):
        return 'yhern3' 
    
    def study_group(self):
        return 'yhern3'

  		  	   		 	 	 			  		 			     			  	 
if __name__ == "__main__":  		  	   		 	 	 			  		 			     			  	 
    print("Remember Q from Star Trek? Well, this isn't him")  		  	   		 	 	 			  		 			     			  	 


