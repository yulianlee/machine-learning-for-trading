import numpy as np

class RTLearner(object):

    def __init__(self, leaf_size=5, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None  

    def author(self):
        return "yhern3"

    def study_group(self):
        return "yhern3"

    def add_evidence(self, data_x, data_y):
        self.tree = self.build_tree(data_x, data_y)

    def build_tree(self, data_x, data_y):
        # If number of rows is less than or equal to leaf size, return a leaf node
        if data_x.shape[0] <= self.leaf_size:
            # For classification with -1, 0, 1 values, we need to handle this specially
            # Count occurrences of each class manually
            unique_vals, counts = np.unique(data_y, return_counts=True)
            mode_val = unique_vals[np.argmax(counts)]
            return np.array([[-1, mode_val, np.nan, np.nan]])

        if np.all(data_y == data_y[0]):
            return np.array([[-1, data_y[0], np.nan, np.nan]])

        best_split_feature, best_split_val = self.calculate_best_split_value_and_feature(data_x)

        left_mask = data_x[:, best_split_feature] <= best_split_val
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            # If the split does not divide data, return a leaf node with mode
            unique_vals, counts = np.unique(data_y, return_counts=True)
            mode_val = unique_vals[np.argmax(counts)]
            return np.array([[-1, mode_val, np.nan, np.nan]])

        # Recursively build left and right subtrees
        left_tree = self.build_tree(data_x[left_mask], data_y[left_mask])
        right_tree = self.build_tree(data_x[right_mask], data_y[right_mask])

        root = np.array([[best_split_feature, best_split_val, 1, left_tree.shape[0] + 1]])
        return np.vstack([root, left_tree, right_tree])

    def calculate_best_split_value_and_feature(self, data_x):
        best_split_feature = np.random.randint(0, data_x.shape[1])
        split_val = np.median(data_x[:, best_split_feature])
        return best_split_feature, split_val

    def query(self, points):
        predictions = np.zeros(points.shape[0])

        for i, point in enumerate(points):
            node_index = 0
            while self.tree[node_index, 0] != -1:
                feature_index = int(self.tree[node_index, 0])
                split_value = self.tree[node_index, 1]

                if point[feature_index] <= split_value:
                    node_index += 1
                else:
                    node_index += int(self.tree[node_index, 3])

            predictions[i] = self.tree[node_index, 1]

        return predictions