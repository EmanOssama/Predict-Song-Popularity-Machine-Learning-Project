from sklearn.svm import SVC
from pandas import read_csv
from PreProcessing import *
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
import pickle

def polyKernel(dataset):

    cols = ['artists', 'id', 'name', 'release_date','popularity_level']
    dataset = preProcessingByLabelEncoder(dataset, cols)
    dataset1 = dataset.iloc[:, 0:10]
    dataset2 = preProcessingByOneHotEncoder(dataset['key'])
    dataset3 = dataset.iloc[:, 11:19]
    dataset = dataset1.join(dataset2)
    dataset = dataset.join(dataset3)
    dataset = dataset.fillna(0)
    #getCorrelationClassification(spotify_data)
    dataset.drop(['mode','explicit','valence','acousticness','duration_ms','artists','speechiness','instrumentalness',0,1,2,3,4,5,6,7,8,9,10,11], axis=1, inplace=True)

    X = dataset.iloc[:,:-1]
    Y = dataset.iloc[:, -1];

    #split data to train and test
    #X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1,shuffle=True);
    #X_validation = np.c_[X_train,Y_train]
    #X_validation_train,X_validation_test,Y_validation_train,Y_validation_test = train_test_split(X_train,Y_train,test_size=0.2,random_state=1);

    scaler = StandardScaler()
    scaler.fit(X)

    X = scaler.transform(X)

    # load the model from disk
    filename = 'PolynomialKernelModel.sav'
    # Load from file
    with open(filename, 'rb') as file:
        pickle_model = pickle.load(file)
    ypred = pickle_model.predict(X)

    start=time.time()
    result = pickle_model.score(X, Y)
    print('Mean Square Error of Polynomial SVM :', metrics.mean_squared_error(Y,ypred))
    print(f"Accuracy Score of Polynomial SVM : {result * 100:.5f}%")
    end=time.time()
    print("Testing Time:", end - start, "Sec")
'''
    svm = SVC(kernel='poly',C=5,degree=5)
    start=time.time()
    svm.fit(X_train,Y_train)
    end1=time.time()
    print("Training Time:", end1-start,"Sec")

    # save the model to disk
    filename = 'PolynomialKernelModel.sav'
    pickle.dump(svm, open(filename, 'wb'))

    prediction=svm.predict(X_test)
    end2=time.time()
    print('Mean Square Error of Polynomial SVM :', metrics.mean_squared_error(Y_test, prediction))
    print(f"Accuracy Score of Polynomial SVM : {svm.score(X_test,Y_test)*100:.5f}%")
    print("Testing Time:", end2-end1,"Sec")
'''