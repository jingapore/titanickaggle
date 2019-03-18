import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from titanic_script import countCabin, corrTable

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
combineddf['Embarked'].fillna(value=combineddf['Embarked'].mode(), inplace=True)

#samplesub = pd.read_csv('./TitanicData/all/gender_submission.csv')

'''
FEATURE ENGINEERING

Work to be done:
- Include categories in OneHotEnc, in case not all categories are represented.

'''

#Examine 'Cabin' feature.
has_cabin_mask = combineddf['Cabin'].notna()
no_cabin_mask = combineddf['Cabin'].isna()

combineddf_hascabin = combineddf[has_cabin_mask]
combineddf_nocabin = combineddf[no_cabin_mask]

combineddf['CabinCount'] = combineddf['Cabin'].apply(countCabin)

#To plot stacked graph of 'CabinCount' vs 'Pclass'.
Pclass_CabinCount_group = combineddf.groupby(['Pclass', 'CabinCount']).size().unstack('CabinCount').fillna(0)
ax_CabinCountvsPclass = Pclass_CabinCount_group.plot(kind='bar', stacked=True, colormap='Blues', edgecolor='black')

#To observe correlation strength.
corrTable(combineddf, train_survival, 'Output/correlation1.csv')

#To examine 'Age' feature.
# print(combineddf['Age'].describe())
fig_age, ax_age = plt.subplots(1, 1, tight_layout=True)
sns.distplot(combineddf['Age'], bins=26, kde=True, ax=ax_age)

#To bin 'Age' into 26 bins.
combineddf['Age_Bins'] = pd.cut(x=combineddf['Age'], bins=26, labels=np.linspace(1, 10, num=26)).astype(float) #Type changed to float, so as to perform correlation analysis.

corrTable(combineddf, train_survival, 'Output/correlation2.csv')
combineddf.to_csv('Output/combineddf.csv')

'''
Normalising and OneHot-ting features.

Categorial:
'Sex': [1 (male), 0 (feamle)]
'Embarked': [C,Q,S]


Continuous:
'Patch'
'Fare'
'CabinCount'
'Age_Bin': [1 to 26]
'Pclass': [1-3]

'''

#numeric features
numeric_features_no_scale = ['CabinCount', 'Age_Bin', 'Pclass']
numeric_features_scale =  ['Patch', 'Fare']
categorical_features = ['Sex', 'Embarked']

numeric_transformer_scale = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

numeric_transformer_no_scale = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='contant', fill_value='missing')),
    ('onehot'), OneHotEncoder(handule_unknown='ignore')
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num_no_scale', numeric_transformer_no_scale, numeric_features_no_scale),
        ('num_scale', numeric_transformer_scale, numeric_features_scale),
        ('cat', categorical_transformer, categorical_features)
    ]
)

clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classfier', RandomForestClassifier())
])
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