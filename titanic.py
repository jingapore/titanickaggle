import pandas as pd
import re, os
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

testdf = pd.read_csv('./TitanicData/all/test.csv').round({'Age':0}).drop(columns=['PassengerId', 'Ticket'])
testdf.dropna(axis=0, subset=['Age'], inplace=True) #331 rows
testdf['Train'] = 0 #To identify whether example is from training set or test set, after concat.

train_survival = traindf['Survived']

traindf.drop(labels='Survived', axis=1, inplace=True)
traindf['Train'] = 1 #To identify whether example is from training set or test set, after concat.

combineddf = pd.concat([traindf, testdf], ignore_index=True) #1043 rows, of which 712 are training data and 331 are test data.
combineddf['Sex'] = combineddf['Sex'].apply(lambda x: 1 if x=='male' else 0) #1 for male, 2 for female.

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
        # To count 'F GXX' as 1, instead of 2--which "x.count(' ')+1" would.
        return len(re.findall(pattern=r'\d+\s+', string=x))
    else:
        return 0

combineddf['CabinCount'] = combineddf['Cabin'].apply(countCabin)

#To plot stacked graph of 'CabinCount' vs 'Pclass'.
Pclass_CabinCount_group = combineddf.groupby(['Pclass', 'CabinCount']).size().unstack('CabinCount').fillna(0)
ax_CabinCountvsPclass = Pclass_CabinCount_group.plot(kind='bar', stacked=True, colormap='Blues', edgecolor='black')

#To observe correlation strength.
combineddf_training_survived = combineddf[combineddf['Train']==1].loc[:, combineddf.columns != 'Train']
combineddf_training_survived['Survived'] = train_survival
combineddf_training_survived.corr().to_csv(path_or_buf='Output/correlation.csv', columns=['Survived'])

#Examine 'Age' feature.

fig_age, ax_age = plt.subplots(1, 1, tight_layout=True)
sns.distplot(combineddf['Age'], bins=15, kde=True, ax=ax_age)

# ct = ColumnTransformer([('categorical_normaliser', OneHotEncoder(), ['Pclass', 'Sex', 'Embarked']), ('continuous_normaliser', StandardScaler(), ['Age', 'SibSp', 'Parch', 'Fare'])]) #may not use 'pclass' for OneHotEnc, since values could be ordinal.
#
# ytrain = train_survival
# XTrain_t = ct.fit_transform(traindf)
# XTest_t = ct.fit_transform(testdf)

'''
MODEL: DECISION TREE
'''

'''
MODEL: SVM
'''