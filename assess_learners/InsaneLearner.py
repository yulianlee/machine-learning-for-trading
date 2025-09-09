import numpy as np  		  	   		 	 	 			  		 			     			  	 
import BagLearner as bl
import LinRegLearner as rl  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.learners = [bl.BagLearner(learner=rl.LinRegLearner, kwargs={}, bags=20, boost=False, verbose=False) for _ in range(20)]

    def author(self):
        return 'yhern3'

    def add_evidence(self, Xtrain, Ytrain):
        
        [learner.add_evidence(Xtrain, Ytrain) for learner in self.learners]

    def query(self, Xtest):
        predictions = np.concatenate([learner.query(Xtest).reshape(-1, 1) for learner in self.learners], axis=1)
        return np.mean(predictions, axis=1) 