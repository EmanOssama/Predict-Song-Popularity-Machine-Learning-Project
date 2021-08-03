from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
import numpy as np
import time
import pickle

def elasticNetRegression(X , Y):
    # evaluate an elastic net model on the dataset
    # define model
    model = ElasticNet(alpha=0.00001, l1_ratio=1)

   # X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.32,random_state=1);

    #X_validation = np.c_[X_train,Y_train]
    #X_validation_train,X_validation_test,Y_validation_train,Y_validation_test = train_test_split(X_train,Y_train,test_size=0.2,random_state=1);
    # define model evaluation method
    #model.fit(X_validation_train,Y_validation_train)
    #prediction = model.predict(X_validation_test)
    #print('Mean Square Error of Validation Elastic Net :', metrics.mean_squared_error(Y_validation_test, prediction))
    #print(f"Accuracy Score of Validation Elastic Net Regression : {r2_score(Y_validation_test,prediction)*100:.1f}%")
    # load the model from disk
    filename = 'ElasticNetRegression.sav'
    # Load from file
    with open(filename, 'rb') as file:
        pickle_model = pickle.load(file)
    ypred = pickle_model.predict(X)

    start = time.time()
    result = pickle_model.score(X, Y)
    print('Mean Square Error of Elastic Net Regression :', metrics.mean_squared_error(Y, ypred))
    print('R2Score of Elastic Net Regression :', r2_score(Y, ypred))
    print(f"Accuracy Score of Elastic Net Regression : {result * 100:.5f}%")
    end = time.time()
    print("Testing Time :", end - start, "Sec")
'''
    # define model evaluation method
    model.fit(X_train,Y_train)
    prediction = model.predict(X_test)
    print('Mean Square Error of Test Elastic Net :', metrics.mean_squared_error(Y_test, prediction))
    print(f"Accuracy Score of Test Elastic Net Regression : {r2_score(Y_test,prediction)*100:.1f}%")

    # save the model to disk
    filename = 'ElasticNetRegression.sav'
    pickle.dump(model, open(filename, 'wb'))
'''
