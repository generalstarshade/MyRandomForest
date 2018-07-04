class Node:
    def __init__(self, feature, threshold, label):
        self.feature = feature
        self.threshold = threshold
        self.label = label
        self.left = None
        self.right = None

class DecisionTree:
    def __init__(self):
        self.root = None

    def train_decision_tree(self, bagged_training_set, bagged_features, min_sample_leaf):
        pass

class RandomForest:
    def __init__(self, max_features, n_estimators, min_sample_leaf, n_bag):
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.min_sample_leaf = min_sample_leaf
        self.n_bag = n_bag
        self.forest = []

    def train_random_forest(self, training_data):
        # create n_estimators decision trees
        for n in range(0, n_estimators):
            # obtain a random n_bag sample from training set
            bagged_training_set = bag_training_set(training_data, n_bag)

            # now obtain a random max_features sample from the feature space 
            bagged_features = bag_features(training_data, max_features)

            # train a decision tree on the bagged_training_set and with the bagged_features
            tree = DecisionTree()
            tree.train_decision_tree(bagged_training_set, bagged_features, min_sample_leaf)

            # append the trained tree to the forest
            forest.append(tree)
