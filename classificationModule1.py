from sklearn.linear_model import LogisticRegression
from pandas import read_csv
from PreProcessing import *
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
import  numpy as np
from sklearn.metrics import r2_score
import time
import pickle

def logisticRegression(dataset):

    cols = ['artists', 'id', 'name', 'release_date','popularity_level']
    dataset = preProcessingByLabelEncoder(dataset, cols)
    dataset1 = dataset.iloc[:, 0:10]
    dataset2 = preProcessingByOneHotEncoder(dataset['key'])
    dataset3 = dataset.iloc[:, 11:19]
    dataset = dataset1.join(dataset2)
    dataset = dataset.join(dataset3)
    dataset = dataset.fillna(0)
    #getCorrelationClassification(spotify_data)
    dataset.drop(['acousticness','artists','id','duration_ms','explicit','instrumentalness','liveness','mode','release_date'], axis=1, inplace=True)

    X = dataset.iloc[:,:-1]
    Y = dataset.iloc[:, -1];

    #split data to train and test
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1,shuffle=True);
    #X_validation = np.c_[X_train,Y_train]
    #X_validation_train,X_validation_test,Y_validation_train,Y_validation_test = train_test_split(X_train,Y_train,test_size=0.2,random_state=1);

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    logreg= LogisticRegression()
    start=time.time()
    logreg.fit(X_train,Y_train)
    end1=time.time()
    print("Training Time:", end1-start,"Sec")

    # save the model to disk
    filename = 'LogisticRegressionModel.sav'
    pickle.dump(logreg, open(filename, 'wb'))

    prediction=logreg.predict(X_test)
    end2=time.time()
    print('Mean Square Error of Logistic Regression :', metrics.mean_squared_error(Y_test, prediction))
    print(f"Accuracy Score of Logistic  Regression : {logreg.score(X_test,Y_test)*100:.5f}%")
    print("Testing Time:", end2-end1,"Sec")


'''
    # load the model from disk
    filename = 'LogisticRegressionModel.sav'
    # Load from file
    with open(filename, 'rb') as file:
        pickle_model = pickle.load(file)
    ypred = pickle_model.predict(X)

    start=time.time()
    result = pickle_model.score(X, Y)
    print('Mean Square Error of Logistic Regression :', metrics.mean_squared_error(Y,ypred))
    print(f"Accuracy Score of Logistic Regression : {result * 100:.5f}%")
    end=time.time()
    print("Testing Time:", end - start, "Sec")
'''