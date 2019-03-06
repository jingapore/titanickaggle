import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

'''
LOADING OF CSV

df columns are: ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

Round column for 'Age', as some are in decimals.

Remove examples with null vals from 'Age' and 'Embarked'.
- Age is likely important. If so, replacing it with a mean will distort analysis.
- Only 1 Embarked example has null val.
'''

traindf = pd.read_csv('./TitanicData/all/train.csv').round({'Age':0}).drop(columns=['PassengerId', 'Ticket'])
traindf.dropna(axis=0, subset=['Age'], inplace=True)
# print('Length and na count in traindf')
# print(len(traindf))
# print(traindf.isna().sum())

testdf = pd.read_csv('./TitanicData/all/test.csv').round({'Age':0}).drop(columns=['PassengerId', 'Ticket'])
# print('Length and na count in testdf')
# print(len(testdf))
# print(testdf.isna().sum())
testdf.dropna(axis=0, subset=['Age'], inplace=True) #331 rows

train_survival = traindf['Survived']

traindf.drop(labels='Survived', axis=1, inplace=True)
combineddf = pd.concat([traindf, testdf], ignore_index=True) #1043 rows, of which 712 are training data and 331 are test data.

#samplesub = pd.read_csv('./TitanicData/all/gender_submission.csv')

'''
FEATURE ENGINEERING

Work to be done:
- Examine cabin, since it has the most number of 'na's.
- Include categories in OneHotEnc, in case not all categories are represented.

'''

#Examine 'Cabin' feature.
has_cabin_mask = combineddf['Cabin'].notna()
no_cabin_mask = combineddf['Cabin'].isna()

combineddf_hascabin = combineddf[has_cabin_mask]
combineddf_nocabin = combineddf[no_cabin_mask]

#Function with input of 'Cabin' values for passenger, and return number of cabins for passenger.
def countCabin(x):
    if isinstance(x, str):
        return x.count(' ')+1
    else:
        return 0

combineddf['CabinCount'] = combineddf['Cabin'].apply(countCabin)

Pclass_CabinCount_group = combineddf.groupby(['Pclass', 'CabinCount']).size().unstack('CabinCount').fillna(0)

sns.plot(Pclass_CabinCount_group)
ax_CabinCountvsPclass = Pclass_CabinCount_group.plot(kind='bar', stacked=True)

# cabinseries = pd.DataFrame(combineddf['Cabin'])
# cabinseries_multilevels = combineddf.loc[[88, 97, 118, 311],:]

#Examine 'Age' feature.

fig_age, ax_age = plt.subplots(1, 1, tight_layout=True)
sns.distplot(combineddf['Age'], bins=15, kde=True, ax=ax_age)
#ax.hist(combineddf['Age'], bins=10)
plt.show()

ct = ColumnTransformer([('categorical_normaliser', OneHotEncoder(), ['Pclass', 'Sex', 'Embarked']), ('continuous_normaliser', StandardScaler(), ['Age', 'SibSp', 'Parch', 'Fare'])]) #may not use 'pclass' for OneHotEnc, since values could be ordinal.

ytrain = train_survival
XTrain_t = ct.fit_transform(traindf)
XTest_t = ct.fit_transform(testdf)

'''
MODEL: DECISION TREE
'''

'''
MODEL: SVM
'''