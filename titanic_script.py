import re

#Function with input of 'Cabin' values for passenger, and return number of cabins for passenger.
def countCabin(x):
    if isinstance(x, str):
        # To count 'F GXX' as 1, instead of 2--which "x.count(' ')+1" would.
        return len(re.findall(pattern=r'\d+\s+', string=x))
    else:
        return 0

def corrTable(featuresdf, outcomedf, path):
    featuresdf_train = featuresdf[featuresdf['Train']==1].loc[:, featuresdf.columns != 'Train']
    featuresdf_train['Survived'] = outcomedf
    return featuresdf_train.corr().to_csv(path_or_buf=path)