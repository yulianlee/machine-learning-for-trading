import numpy as np

class DTLearner:
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose

    def author(self):
        return "yhern3"

    def study_group(self):
        return "yhern3"

    def add_evidence(self, data_x, data_y):
        self.tree = self.build_tree(data_x, data_y)

    def build_tree(self, data_x, data_y):
        if data_x.shape[0] <= self.leaf_size or np.all(data_y == data_y[0]):
            return np.array([[-1, np.mean(data_y), np.nan, np.nan]])

        best_feature, best_split_val = self.select_best_split(data_x, data_y)
        left_mask = data_x[:, best_feature] <= best_split_val
        right_mask = ~left_mask

        if np.all(left_mask) or np.all(right_mask):
            return np.array([[-1, np.mean(data_y), np.nan, np.nan]])

        left_tree = self.build_tree(data_x[left_mask], data_y[left_mask])
        right_tree = self.build_tree(data_x[right_mask], data_y[right_mask])

        root = np.array([[best_feature, best_split_val, 1, left_tree.shape[0] + 1]])
        return np.vstack((root, left_tree, right_tree))

    def select_best_split(self, data_x, data_y):
        correlation_matrix = np.corrcoef(np.column_stack((data_x, data_y)), rowvar=False)
        best_feature = np.argmax(np.abs(correlation_matrix[:-1, -1]))
        split_val = np.median(data_x[:, best_feature])
        return best_feature, split_val

    def query(self, points):
        predictions = np.apply_along_axis(self.predict_single, axis=1, arr=points)
        return predictions

    def predict_single(self, point):
        index = 0
        while self.tree[index, 0] != -1: 
            feature_index = int(self.tree[index, 0])
            split_value = self.tree[index, 1]
            index += 1 if point[feature_index] <= split_value else int(self.tree[index, 3])
        return self.tree[index, 1]