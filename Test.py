from pandas import read_csv
from classificationModule1 import *
from classificationModule2 import *
from classificationModule3 import *
from classificationModule4 import *
from classificationModule5 import *
from classificationModule6 import *
from classificationModule7 import *

dataset = read_csv('spotify_testing_classification.csv', low_memory=False);

# drop nulls
dataset['valence'] = dataset['valence'].fillna(0)
dataset['year'] = dataset['year'].fillna(0)
dataset['acousticness'] = dataset['acousticness'].fillna(0)
dataset['danceability'] = dataset['danceability'].fillna(0)
dataset['energy'] = dataset['energy'].fillna(0)
dataset['explicit'] = dataset['explicit'].fillna(0)
dataset['instrumentalness'] = dataset['instrumentalness'].fillna(0)
dataset['key'] = dataset['key'].fillna(0)
dataset['liveness'] = dataset['liveness'].fillna(0)
dataset['loudness'] = dataset['loudness'].fillna(0)
dataset['mode'] = dataset['mode'].fillna(0)
dataset['tempo'] = dataset['tempo'].fillna(0)
dataset['speechiness'] = dataset['speechiness'].fillna(0)
tempDataset = pd.DataFrame(dataset)

dataset.dropna(how='any', inplace=True);
spotify_data = dataset.iloc[:, :];
mostCommonList = []

def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

for i in dataset:
    mostCommonList.append(Most_Common(dataset[i]))
dataset = pd.DataFrame(tempDataset)
j = 0
for i in dataset:
    dataset[i] = dataset[i].fillna(mostCommonList[j])
    j += 1
'''cols = ['artists', 'id', 'name', 'release_date','popularity_level']
dataset = preProcessingByLabelEncoder(dataset, cols)
dataset1 = dataset.iloc[:, 0:10]
dataset2 = preProcessingByOneHotEncoder(dataset['key'])
dataset3 = dataset.iloc[:, 11:19]
dataset = dataset1.join(dataset2)
dataset = dataset.join(dataset3)
dataset = dataset.fillna(0)'''
logisticRegression(dataset)
print()
linearKernel(dataset)
print()
#polyKernel(dataset)
print()
#kernelRBF(dataset)
print()
randomForestClassifer(dataset)
print()
KnnClassifer(dataset)
PCA_(dataset)
print()