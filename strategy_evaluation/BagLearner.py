import numpy as np

class BagLearner:
    def __init__(self, learner, kwargs={}, bags=20, boost=False, verbose=False):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.learners = [learner(**kwargs) for _ in range(bags)]
    
    def author(self):
        return "yhern3"

    def study_group(self):
        return "yhern3"  	   
       
    def add_evidence(self, data_x, data_y):
        n = data_x.shape[0]
        for learner in self.learners:
            sample_indices = np.random.choice(n, n, replace=True)
            learner.add_evidence(data_x[sample_indices], data_y[sample_indices])
    
    def query(self, points):
        # Get predictions from all learners
        predictions = np.array([learner.query(points) for learner in self.learners])
        
        # For classification, take the mode (most common value) for each point
        result = np.zeros(points.shape[0])
        
        for i in range(points.shape[0]):
            values, counts = np.unique(predictions[:, i].astype(int), return_counts=True)
            result[i] = values[np.argmax(counts)]
        
        return result