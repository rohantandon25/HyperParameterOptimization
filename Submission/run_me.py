# run_me.py module
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

import kaggle
from sklearn.neighbors import KNeighborsClassifier
import sklearn.tree
import time
# Assuming you are running run_me.py from the Submission/Code directory, otherwise the path variable will be different for you
import numpy as np
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

# Load the Email spam data
path = '../../Data/Email_spam/'
train = np.load(path + 'train.npy')
test = np.load(path + 'test_distribute.npy')
train_x = train[:, 1:]
train_y = train[:, 0]
test_x = test[:, 1:]
# this is dummy vector of labels
#test_y = test[:, 0]

#knn classifier
knn = KNeighborsClassifier(n_neighbors=5)
start_knn = time.time()
knn.fit(train_x, train_y)
end_knn = time.time()
email_knn_time_training = end_knn - start_knn
print "knn email training time = ", email_knn_time_training

start_knn_predict = time.time()
test_y = knn.predict(test_x)
end_knn_predict = time.time()
email_knn_time_predicting = end_knn_predict - start_knn_predict
print "Email spam:", train_x.shape, train_y.shape, test_x.shape, test_y.shape 

#Save prediction file in Kaggle format
predictions = knn.predict(test_x)
kaggle.kaggleize(predictions, "../Predictions/Email_spam/testKNN.csv")
email_knn_accuracy = .77

#decision tree
dt = sklearn.tree.DecisionTreeClassifier()
dt_start = time.time()
dt.fit(train_x, train_y)
dt_end = time.time()
email_dt_time_training = dt_end - dt_start
x = time.time()
dt_predict = dt.predict(test_x)
y = time.time()
email_dt_time_predict = y - x
kaggle.kaggleize(dt_predict, "../Predictions/Email_spam/testDT.csv")
email_dt_accuracy = .88

#neural networks classifier

nn = MLPClassifier()
start = time.time()
nn.fit(train_x, train_y)
end = time.time()
email_nn_time_train = end - start
st = time.time()
nn_predict = nn.predict(test_x)
ed = time.time()
email_nn_time_predict = ed - st
kaggle.kaggleize(nn_predict, "../Predictions/Email_spam/testnn.csv")
email_nn_accuracy = .913

# Load the Occupancy detection data
path = '../../Data/Occupancy_detection/'
train = np.load(path + 'train.npy')
test = np.load(path + 'test_distribute.npy')
train_x = train[:, 1:]
train_y = train[:, 0]
test_x = test[:, 1:]
# this is dummy vector of labels
test_y = test[:, 0]
print "Occupancy detection:", train_x.shape, train_y.shape, test_x.shape, test_y.shape

#knn for occupancy data
knn = KNeighborsClassifier()
start_knn = time.time()
knn.fit(train_x, train_y)
end_knn = time.time()
occupancy_knn_time_training = end_knn - start_knn
print "knn email training time = ", email_knn_time_training

start_knn_predict = time.time()
test_y = knn.predict(test_x)
end_knn_predict = time.time()
occupancy_knn_time_predicting = end_knn_predict - start_knn_predict
predictions = knn.predict(test_x)
#predictions = np.zeros(test_y.shape)
kaggle.kaggleize(predictions, "../Predictions/Occupancy_detection/testKNN.csv")
occupancy_knn_accuracy = .989

#decision trees for occupancy data
dt = sklearn.tree.DecisionTreeClassifier()
dt_start = time.time()
dt.fit(train_x, train_y)
dt_end = time.time()
occupancy_dt_time_training = dt_end - dt_start
x = time.time()
dt_predict = dt.predict(test_x)
y = time.time()
occupancy_dt_time_predict = y - x
kaggle.kaggleize(dt_predict, "../Predictions/Occupancy_detection/testDT.csv")
occupancy_dt_accuracy = .987

#nn for occupancy data
nn = MLPClassifier()
start = time.time()
nn.fit(train_x, train_y)
end = time.time()
occupancy_nn_time_train = end - start
st = time.time()
nn_predict = nn.predict(test_x)
ed = time.time()
occupancy_nn_time_predict = ed - st
kaggle.kaggleize(nn_predict, "../Predictions/Occupancy_detection/testnn.csv")
occupancy_nn_accuracy = .981

#bagging for q6
bagging = BaggingClassifier()
timex = time.time()
bagging.fit(train_x, train_y)
time_end = time.time()
time_bagging_fit = time_end - timex
time2 = time.time()
bagging.predict(test_x)
time_end2 = time.time()
time_bagging_predict = time_end2 - time2
bagging_accuracy = .992


# Load the USPS digits data
path = '../../Data/USPS_digits/'
train = np.load(path + 'train.npy')
test = np.load(path + 'test_distribute.npy')
train_x = train[:, 1:]
train_y = train[:, 0]
test_x = test[:, 1:]
# this is dummy vector of labels
#test_y = test[:, 0]
print "USPS digits:", train_x.shape, train_y.shape, test_x.shape, test_y.shape 

#Save prediction file in Kaggle format
#predictions = np.zeros(test_y.shape)

#knn for usps data
knn = KNeighborsClassifier()
start_knn = time.time()
knn.fit(train_x, train_y)
end_knn = time.time()
usps_knn_time_training = end_knn - start_knn

start_knn_predict = time.time()
test_y = knn.predict(test_x)
end_knn_predict = time.time()
usps_knn_time_predicting = end_knn_predict - start_knn_predict
predictions = knn.predict(test_x)
kaggle.kaggleize(predictions, "../Predictions/USPS_digits/testKNN.csv")
usps_knn_accuracy = .968

#decision trees for usps data
dt = sklearn.tree.DecisionTreeClassifier()
dt_start = time.time()
dt.fit(train_x, train_y)
dt_end = time.time()
usps_dt_time_training = dt_end - dt_start
x = time.time()
dt_predict = dt.predict(test_x)
y = time.time()
usps_dt_time_predict = y - x
kaggle.kaggleize(dt_predict, "../Predictions/USPS_digits/testDT.csv")
usps_dt_accuracy = .827

#nn for occupancy data
nn = MLPClassifier()
start = time.time()
nn.fit(train_x, train_y)
end = time.time()
usps_nn_time_train = end - start
st = time.time()
nn_predict = nn.predict(test_x)
ed = time.time()
usps_nn_time_predict = ed - st
kaggle.kaggleize(nn_predict, "../Predictions/USPS_digits/testnn.csv")
usps_nn_accuracy = .96

#generic method for hyperparameter selection
#done on usps data
train_data = train_x
train_data_2 = train_y
path = '../../Data/USPS_digits/'
train = np.load(path + 'train.npy')

classifiers = (KNeighborsClassifier(), DecisionTreeClassifier(), MLPClassifier())
datasets = (train_data, train_data_2)
hyperParameterVals = range(1,101)

vals = {"classifier": KNeighborsClassifier(), "dataSets": train, "hypVals": 5}

# def hyperParameterSelection(listHyp, classifier, dataSets, hypVals):
#     labels = dataSets[:, 0]
#     data = dataSets[:, 1:]
#     seedVal = 0
#     for hyper in listHyp:
#         np.random.seed(seedVal)
#         np.random.shuffle(labels)
#         np.random.shuffle(data)
#         seedVal +=1
#         listDataChunks = ()
#         listLabelChunks = ()
#         for i in range(0, len(dataSets), 10):
#             listDataChunks.__add__(dataSets[i:i + 10])
#             listLabelChunks.__add__(labels[i:i+10])
#         while i < 10:
#             classifier.fit(listDataChunks(i), listLabelChunks(i))
#             [elem for elem in listDataChunks if listDataChunks.index(elem) != i]
#
#             i = i +1
#
# hyperParameterSelection(hyperParameterVals, **vals)
#
#
#
#




#email bar graph
p1 = plt.Rectangle((0, 0), 0.1, 0.1, fc="crimson")
p2 = plt.Rectangle((0, 0), 0.1, 0.1, fc="burlywood")
p3 = plt.Rectangle((0, 0), 0.1, 0.1, fc="chartreuse")

plt.legend((p1, p2, p3), ('Training Time','Prediction Time','Accuracy'), loc='upper left')

labels = ['        ', '      KNN', '        ', '        ', '      DecisionTrees', '        ', '        ', '      NeuralNetworks', '        ']

x = np.array([0,1,2,5,6,7,10,11,12]) + 1

y = np.array([email_knn_time_training, email_knn_time_predicting, email_knn_accuracy,
              email_dt_time_training, email_dt_time_predict, email_dt_accuracy,
              email_nn_time_train, email_nn_time_predict, email_nn_accuracy])

plt.xticks(x, labels)

# Creates the bar chart
plt.bar(left = x, height=y, color=['crimson', 'burlywood', 'chartreuse'])

plt.grid(which='both')
plt.ylabel('Scores')
plt.xlabel('Classifiers')
plt.title("Email Stats")

plt.show()
#plt.savefig("../Figures/emailStats.pdf")


# #occupancy bar graph
p1 = plt.Rectangle((0, 0), 0.1, 0.1, fc="crimson")
p2 = plt.Rectangle((0, 0), 0.1, 0.1, fc="burlywood")
p3 = plt.Rectangle((0, 0), 0.1, 0.1, fc="chartreuse")

plt.legend((p1, p2, p3), ('Training Time','Prediction Time','Accuracy'), loc='upper left')

labels = ['        ', '      KNN', '        ', '        ', '      DecisionTrees', '        ', '        ', '      NeuralNetworks', '        ']

x = np.array([0,1,2,5,6,7,10,11,12]) + 1

y = np.array([occupancy_knn_time_training, occupancy_knn_time_predicting, occupancy_knn_accuracy,
              occupancy_dt_time_training, occupancy_dt_time_predict, occupancy_dt_accuracy,
              occupancy_nn_time_train, occupancy_nn_time_predict, occupancy_nn_accuracy])

plt.xticks(x, labels)

# Creates the bar chart
plt.bar(left = x, height=y, color=['crimson', 'burlywood', 'chartreuse'])

plt.grid(which='both')
plt.ylabel('Scores')
plt.xlabel('Classifiers')
plt.title("Occupancy Stats")

plt.show()
#
# #usps bar graph
p1 = plt.Rectangle((0, 0), 0.1, 0.1, fc="crimson")
p2 = plt.Rectangle((0, 0), 0.1, 0.1, fc="burlywood")
p3 = plt.Rectangle((0, 0), 0.1, 0.1, fc="chartreuse")

plt.legend((p1, p2, p3), ('Training Time','Prediction Time','Accuracy'), loc='upper left')

labels = ['        ', '      KNN', '        ', '        ', '      DecisionTrees', '        ', '        ', '      NeuralNetworks', '        ']

x = np.array([0,1,2,5,6,7,10,11,12]) + 1

y = np.array([usps_knn_time_training, usps_knn_time_predicting, usps_knn_accuracy,
              usps_dt_time_training, usps_dt_time_predict, usps_dt_accuracy,
              usps_nn_time_train, usps_nn_time_predict, usps_nn_accuracy])

plt.xticks(x, labels)
plt.bar(left = x, height=y, color=['crimson', 'burlywood', 'chartreuse'])

plt.grid(which='both')
plt.ylabel('Scores')
plt.xlabel('Classifiers')
plt.title("USPS Stats")

plt.show()

#comparing knn vs dt vs neuralNetworks vs bagging for occupational data

p1 = plt.Rectangle((0, 0), 0.1, 0.1, fc="crimson")
p2 = plt.Rectangle((0, 0), 0.1, 0.1, fc="burlywood")
p3 = plt.Rectangle((0, 0), 0.1, 0.1, fc="chartreuse")


labels = ['        ', '      KNN', '        ', '        ', '      DecisionTrees', '        ', '        ', '      NeuralNetworks', '          ', '              ', '        Bagging']

x = np.array([0,1,2,5,6,7,10,11,12,13,15,16]) + 1

y = np.array([occupancy_knn_time_training, occupancy_knn_time_predicting, occupancy_knn_accuracy,
              occupancy_dt_time_training, occupancy_dt_time_predict, occupancy_dt_accuracy,
              occupancy_nn_time_train, occupancy_nn_time_predict, occupancy_nn_accuracy,
                time_bagging_fit, time_bagging_predict, bagging_accuracy ])

plt.legend((p1, p2, p3), ('Training Time','Prediction Time','Accuracy'), loc='upper left')
plt.xticks(x, labels)
plt.bar(left = x, height=y, color=['crimson', 'burlywood', 'chartreuse'])
plt.grid(which='both')
plt.ylabel("Scores")
plt.xlabel("Classifiers")
plt.title("Classifier Comparison")
plt.show()