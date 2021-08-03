from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from PreProcessing import *
import pickle
import time

from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

def randomForestClassifer(dataset):
    cols = ['artists', 'id', 'name', 'release_date', 'popularity_level']
    dataset = preProcessingByLabelEncoder(dataset, cols)
    dataset1 = dataset.iloc[:, 0:10]
    dataset2 = preProcessingByOneHotEncoder(dataset['key'])
    dataset3 = dataset.iloc[:, 11:19]
    dataset = dataset1.join(dataset2)
    dataset = dataset.join(dataset3)
    dataset = dataset.fillna(0)
    # getCorrelationClassification(spotify_data)
    dataset.drop(['acousticness', 'artists', 'id', 'duration_ms', 'explicit', 'instrumentalness', 'liveness', 'mode','release_date'], axis=1, inplace=True)

    X = dataset.iloc[:, :-1]
    Y = dataset.iloc[:, -1];

    # split data to train and test
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1,shuffle=True);

    # load the model from disk

    RFC_Model = RandomForestClassifier()
    start = time.time()
    RFC_Model.fit(X_train, Y_train)
    RFC_Predict = RFC_Model.predict(X_test)
    RFC_Accuracy = accuracy_score(Y_test, RFC_Predict)
    print("Accuracy of Random Forest: " + str(RFC_Accuracy * 100) + "%")
    end = time.time()
    print('Training Time :', end - start, 'Sec')
    # save the model to disk
    filename = 'RandomForestModel.sav'
    pickle.dump(RFC_Model, open(filename, 'wb'))
'''  filename = 'RandomForestModel.sav'
    # Load from file
    with open(filename, 'rb') as file:
        pickle_model = pickle.load(file)

    start = time.time()
    result = pickle_model.score(X, Y)
    print(f"Accuracy Score of Random Forest : {result * 100:.5f}%")
    end = time.time()
    print('Testing Time :', end - start, 'Sec')
'''