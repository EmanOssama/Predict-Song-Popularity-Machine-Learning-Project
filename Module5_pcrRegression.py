from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import  numpy as np
import time
import pickle

def pcrRegression(X , Y):
    #split data to train and test
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.32,random_state=1);

    #X_validation = np.c_[X_train,Y_train]
    #X_validation_train,X_validation_test,Y_validation_train,Y_validation_test = train_test_split(X_train,Y_train,test_size=0.2,random_state=1);

    pcr = make_pipeline(StandardScaler(), PCA(n_components=17), LinearRegression())
    #pcr.fit(X_validation_train, Y_validation_train)
    #pca = pcr.named_steps['pca']  # retrieve the PCA step of the pipeline
    #prediction = pcr.predict(X_validation_test)
    #print('Mean Square Error of Validation PCR:', metrics.mean_squared_error(Y_validation_test, prediction))
    #print(f"Accuracy Score of Validation PCR Regression : {r2_score(Y_validation_test,prediction)*100:.1f}%")

    pcr.fit(X_train, Y_train)
    pca = pcr.named_steps['pca']  # retrieve the PCA step of the pipeline
    prediction = pcr.predict(X_test)
    print('Mean Square Error of Test PCR:', metrics.mean_squared_error(Y_test, prediction))
    print(f"Accuracy Score of Test PCR Regression : {r2_score(Y_test,prediction)*100:.1f}%")

    # save the model to disk
    filename = 'PCRRegression.sav'
    pickle.dump(pcr, open(filename, 'wb'))
'''
    # load the model from disk
    filename = 'PCRRegression.sav'
    # Load from file
    with open(filename, 'rb') as file:
        pickle_model = pickle.load(file)
    ypred = pickle_model.predict(X)

    start=time.time()
    result = pickle_model.score(X, Y)
    print('Mean Square Error of PCR Regression :',metrics.mean_squared_error(Y,ypred))
    print('R2Score of PCR Regression :', r2_score(Y, ypred))
    print(f"Accuracy Score of PCR Regression : {result * 100:.5f}%")
    end=time.time()
    print("Testing Time :", end - start, "Sec")
'''