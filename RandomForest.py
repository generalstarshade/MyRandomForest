import random
import math
from numpy import sign

EPSILON = 1e-7

class Node:
    def __init__(self, feature, threshold, label, label_0_count, label_1_count):
        self.feature = feature
        self.threshold = threshold
        self.label = label
        self.left = None
        self.right = None
        self.label_0_count = label_0_count
        self.label_1_count = label_1_count

class DecisionTree:
    def __init__(self):
        self.root = None

    def all_labels_are_same(self, vectors):
        if len(vectors) <= 1:
            return True
        comparison_variable = vectors[0][-1]
        for vector in vectors:
            if vector[-1] != comparison_variable:
                return False
        return True

    def count_labels(self, feature_idx, threshold, data):
        # given a feature and a threshold, split the data and count the labels
        # that end up on both sides
        left_vectors = []
        right_vectors = []
        for vector in data:
            if vector[feature_idx] < threshold:
                left_vectors.append(vector)
            else:
                right_vectors.append(vector)

        # now count the labels in the left and right vector lists
        left_label_0_count = 0
        left_label_1_count = 0
        right_label_0_count = 0
        right_label_1_count = 0

        for vector in left_vectors:
            if vector[-1] == 0:
                left_label_0_count += 1
            else:
                left_label_1_count += 1

        for vector in right_vectors:
            if vector[-1] == 0:
                right_label_0_count += 1
            else:
                right_label_1_count += 1

        total_left_labels = left_label_0_count + left_label_1_count
        total_right_labels = right_label_0_count + right_label_1_count

        # prevent division by 0
        if total_left_labels == 0:
            total_left_labels = 1
        if total_right_labels == 0:
            total_right_labels = 1

        total_labels = total_left_labels + total_right_labels
        
        return (float(total_left_labels), float(left_label_0_count), float(left_label_1_count), 
                float(total_right_labels), float(right_label_0_count), float(right_label_1_count), 
                float(total_labels))

    def get_threshold_candidates(self, data, bagged_features):
        sorted_thresholds = [] # of the form [(feature_idx, [threshes...]), ...]
        for feature_idx in bagged_features:
            working = []
            for vector_idx in range(0, len(data)):
                working.append(data[vector_idx][feature_idx])
            sorted_thresholds.append([feature_idx, working])

        for vector_idx in range(0, len(sorted_thresholds)):
            sorted_thresholds[vector_idx][1] = sorted(set(sorted_thresholds[vector_idx][1]))

        real_sorted_thresh = []
        next_thresh_idx = 0
        for feature_idx in range(0, len(sorted_thresholds)):
            working = []
            the_feature = sorted_thresholds[feature_idx][0]
            thresh_value = sorted_thresholds[feature_idx][1][0]
            thresholds_length = len(sorted_thresholds[feature_idx][1])
            for thresh_idx in range(0, thresholds_length - 1):
                thresh_value = (sorted_thresholds[feature_idx][1][thresh_idx] +
                               sorted_thresholds[feature_idx][1][thresh_idx + 1]) / 2.0
                working.append(thresh_value)

            if thresholds_length == 1:
                working.append(thresh_value)

            real_sorted_thresh.append([the_feature, working])

        return real_sorted_thresh

    def split_data(self, working_data, bagged_features):
        num_elements = len(working_data)
        #print num_elements
        entropy_list = [] # a list of the form: [((feature, threshold), entropy), ...]

        entropy_count = 0

        threshold_candidates = self.get_threshold_candidates(working_data, bagged_features)
        #print len(threshold_candidates)
        
        for feature_idx in range(0, len(threshold_candidates)):
            thresh_length = len(threshold_candidates[feature_idx][1])
            the_feature = threshold_candidates[feature_idx][0]
            for thresh_idx in range(0, thresh_length):
                # split the data and get label info
                threshold = threshold_candidates[feature_idx][1][thresh_idx]
                (total_left_labels, left_label_0_count, left_label_1_count, total_right_labels, right_label_0_count,
                 right_label_1_count, total_labels) = self.count_labels(the_feature, threshold, working_data)

                # calculate the entropy of this particular feature/threshold pair
                probability_left_label_is_0_given_left = left_label_0_count / total_left_labels
                probability_left_label_is_1_given_left = left_label_1_count / total_left_labels
                if abs(probability_left_label_is_0_given_left) < EPSILON:
                    left_expression = 0
                else:
                    left_expression = - probability_left_label_is_0_given_left * math.log(probability_left_label_is_0_given_left, math.e)

                if abs(probability_left_label_is_1_given_left) < EPSILON:
                    right_expression = 0
                else:
                    right_expression = - probability_left_label_is_1_given_left * math.log(probability_left_label_is_1_given_left, math.e)

                entropy_left = left_expression + right_expression

                probability_right_label_is_0_given_right = right_label_0_count / total_right_labels
                probability_right_label_is_1_given_right = right_label_1_count / total_right_labels

                if abs(probability_right_label_is_0_given_right) < EPSILON:
                    left_expression = 0
                else:
                    left_expression = - probability_right_label_is_0_given_right * math.log(probability_right_label_is_0_given_right, math.e)

                if abs(probability_right_label_is_1_given_right) < EPSILON:
                    right_expression = 0
                else:
                    right_expression = - probability_right_label_is_1_given_right * math.log(probability_right_label_is_1_given_right, math.e)

                entropy_right = left_expression + right_expression

                probability_label_is_left = total_left_labels / total_labels
                probability_label_is_right = total_right_labels / total_labels

                cond_entropy = probability_label_is_left * entropy_left + \
                               probability_label_is_right * entropy_right

                entropy_x = -probability_label_is_left * math.log(probability_label_is_left, math.e) + \
                            -probability_label_is_right * math.log(probability_label_is_right, math.e)

                info_gain = entropy_x - cond_entropy

                entropy_list.append(((the_feature, threshold), info_gain))

                entropy_count += 1
                if entropy_count % 5000 == 0:
                    print "Completed %d entropy calculations" % entropy_count

        # now that we have the full list of info gains, find the largest one
        sorted_entropies = sorted(entropy_list, key = lambda x : x[1])
        #print sorted_entropies[-1]
        feature = sorted_entropies[-1][0][0]
        threshold = sorted_entropies[-1][0][1]
        label_0s_at_node = left_label_0_count + right_label_0_count
        label_1s_at_node = left_label_1_count + right_label_1_count

        # create the left and right vectors
        left_vectors = []
        right_vectors = []
        for vector in working_data:
            if vector[feature] < threshold:
                left_vectors.append(vector)
            else:
                right_vectors.append(vector)

        return (feature, threshold, left_vectors, right_vectors, label_0s_at_node, label_1s_at_node)

    def train_decision_tree(self, root, bagged_training_set, bagged_features, min_sample_leaf):
        (feature, threshold, left_vectors, right_vectors, label_0_count, label_1_count) = self.split_data(bagged_training_set, bagged_features)

        if root == None:
            # base case, start of tree
            self.root = Node(feature, threshold, None, label_0_count, label_1_count)
            root = self.root

            root.left = Node(0, 0, None, 0, 0)
            root.right = Node(0, 0, None, 0, 0)

            # recurse on both ends of the tree
            self.train_decision_tree(root.left, left_vectors, bagged_features, min_sample_leaf)
            self.train_decision_tree(root.right, right_vectors, bagged_features, min_sample_leaf)

        else:

            root.feature = feature
            root.threshold = threshold
            root.label = None
            root.label_0_count = label_0_count
            root.label_1_count = label_1_count

            if len(left_vectors) == 0 or len(right_vectors) == 0:
                if label_0_count > label_1_count:
                    root.label = 0
                else:
                    root.label = 1
                return

            if self.all_labels_are_same(left_vectors):
                root.left = Node(feature, threshold, left_vectors[0][-1], label_0_count, label_1_count)
            else:
                root.left = Node(0, 0, 0, 0, 0)
                self.train_decision_tree(root.left, left_vectors, bagged_features, min_sample_leaf)

            if self.all_labels_are_same(right_vectors):
                root.right = Node(feature, threshold, right_vectors[0][-1], label_0_count, label_1_count)
            else:
                root.right = Node(0, 0, 0, 0, 0)
                self.train_decision_tree(root.right, right_vectors, bagged_features, min_sample_leaf)

        
    def classify(self, test_vector):
        working_node = self.root
        while working_node.label == None:
            if test_vector[working_node.feature] < working_node.threshold:
                working_node = working_node.left
            else:
                working_node = working_node.right

        label = working_node.label
        return label

class RandomForest:
    def __init__(self, max_features, n_estimators, min_sample_leaf, n_bag):
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.min_sample_leaf = min_sample_leaf
        self.n_bag = n_bag
        self.forest = []

    def train_random_forest(self, training_data):
        # create n_estimators decision trees
        for n in range(0, self.n_estimators):
            # obtain a random n_bag sample from training set
            bagged_training_set = random.sample(training_data, self.n_bag)

            # now obtain a random max_features sample from the feature space 
            bagged_features = random.sample(xrange(len(training_data[0]) - 1), self.max_features)
            print "Features: " + str(bagged_features)

            # train a decision tree on the bagged_training_set and with the bagged_features
            tree = DecisionTree()
            tree.train_decision_tree(None, bagged_training_set, bagged_features, self.min_sample_leaf)

            # append the trained tree to the forest
            self.forest.append(tree)

    def classify(self, test_data):
        error_count = 0
        i = 0
        for test_vector in test_data:
            label_0_count = 0
            label_1_count = 0
            predicted_label = 0
            for tree in self.forest:
                label = tree.classify(test_vector)
                if label == 0:
                    label_0_count += 1
                else:
                    label_1_count += 1

            if label_0_count > label_1_count:
                predicted_label = 0
            elif label_0_count < label_1_count:
                predicted_label = 1
            else:
                predicted_label = random.randint(0, 1)
                
            if predicted_label != test_vector[-1]:
                error_count += 1

            i += 1
            if i % 50 == 0:
                print "Classified %d vectors." % i

        error = error_count / float(len(test_data))
        return error
