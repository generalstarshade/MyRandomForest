import RandomForest
import numpy
import math

training_data = numpy.loadtxt("training_data.txt")
test_data = numpy.loadtxt("test_data.txt")

forest = RandomForest.RandomForest(int(math.sqrt(22)), 30, 0, int(len(training_data) * (2/3.)))


forest.train_random_forest(training_data)

error = forest.classify(test_data)
print "Error: %f" % error

"""
tree = RandomForest.DecisionTree()
tree.train_decision_tree(None, training_data, range(0, 22), 0)

error_count = 0
for vector in test_data:
    res = tree.classify(vector)
    if res == -1:
        res = 0
    if res != vector[-1]:
        error_count += 1
print "Error: %f" % (error_count / float(len(test_data)))
"""
