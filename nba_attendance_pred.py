import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn


from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# CSV that I manually scraped the data for
attendance = read_csv('C:/Users/drewm/Desktop/attendance/all_data.csv')

# Predictive variables are: previous season's winning %, previous season's home
# winning %, how many All-NBA players the team had in the previous season, how well
# the team did in the playoffs in the previous season, and previous season's
# attendance

# Target variable is this season's attendance

print(attendance.shape)
#print(attendance.head(5))

#attendance.hist()
#pyplot.show()

array = attendance.values
X = array[:,8:]
y = array[:,7]
#print(X)

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# trying a bunch of different prediction models
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

results = []
names = []

# check the accuracy of each
for name, model in models:
 kfold = StratifiedKFold(n_splits=8, random_state=1, shuffle=True)
 cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
 results.append(cv_results)
 names.append(name)
 print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

 # Logistic regression is the most accurate at 44.5% accuracy (comparing
 # to the base rate of 16.7%, this is pretty good)

