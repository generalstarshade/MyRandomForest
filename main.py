import RandomForest
import numpy
import math

training_data = numpy.loadtxt("training_data.txt")
test_data = numpy.loadtxt("test_data.txt")
forest = RandomForest.RandomForest(int(math.sqrt(22)), 10, 0, len(training_data) / 3)


forest.train_random_forest(training_data)

error = forest.classify(test_data)

print "Error: %f" % error
