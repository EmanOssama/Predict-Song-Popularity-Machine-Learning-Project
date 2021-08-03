from Module1_multiVariableRegression import *
from Module2_polynomialRegression import *
from Module3_elasticNetRegression import *
from Module4_plsRegression import *
from Module5_pcrRegression import *
from PreProcessing import *
from Feature_Analysis import *
from pandas import read_csv
from collections import Counter

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # read data from excel
    dataset = read_csv('spotify_testing.csv', low_memory=False);
    #dataset['artists'] = dataset['artists'].fillna('UNKNOWN')
    #dataset['id'] = dataset['id'].fillna('UNKNOWN')
    # drop nulls
    #tempDataset = pd.DataFrame(dataset)

    dataset.dropna(how='any', inplace=True);
    spotify_data = dataset.iloc[:, :];
    '''mostCommonList = []

    def Most_Common(lst):
        data = Counter(lst)
        return data.most_common(1)[0][0]

    for i in dataset:
        mostCommonList.append(Most_Common(dataset[i]))
    dataset = pd.DataFrame(tempDataset)
    j = 0
    for i in dataset:
        dataset[i] = dataset[i].fillna(mostCommonList[j])
        j += 1'''
    dataset['artists'] = dataset['artists'].fillna('0')
    dataset['id'] = dataset['id'].fillna('0')
    dataset['name'] = dataset['name'].fillna('0')
    dataset['release_date'] = dataset['release_date'].fillna('0')
    dataset = dataset.fillna(0)
    #############################
    cols = ['artists', 'id', 'name', 'release_date']
    dataset = preProcessingByLabelEncoder(dataset, cols)
    dataset1 = dataset.iloc[:, 0:10]
    dataset2 = preProcessingByOneHotEncoder(dataset['key'])
    dataset3 = dataset.iloc[:, 11:19]
    dataset = dataset1.join(dataset2)
    dataset = dataset.join(dataset3)

    getCorrelationRegression(spotify_data)
    dataset.drop([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], axis=1, inplace=True)

    X = dataset.iloc[:,:-1]
    Y = dataset.iloc[:, -1];

    #multiVariableRegression(X,Y)
    #print()
    #polynomialRegression(X,Y)
    #print()
    elasticNetRegression(X,Y)
    print()
    #plsRegression(X,Y)
    #print()
    #pcrRegression(X,Y)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/