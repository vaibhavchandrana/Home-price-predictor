import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None 

    
    

def house_price(location,area,bathroom,bhk):
    dataframe1 = pd.read_csv("bengaluru_house_prices.csv")
    df2 = dataframe1.drop(['area_type','society','balcony','availability'],axis='columns')
    df3 = df2.dropna()
    df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
    df4 = df3.copy()
    df4.total_sqft = df4.total_sqft.apply(convert_sqft_to_num)
    df4 = df4[df4.total_sqft.notnull()]
    df5 = df4.copy()
    df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']
    df5.location = df5.location.apply(lambda x: x.strip())
    location_stats = df5['location'].value_counts(ascending=False)
    location_stats_less_than_10 = location_stats[location_stats<=10]
    df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
    df6 = df5[~(df5.total_sqft/df5.bhk<300)]
    df_out = pd.DataFrame()
    for key, subdf in df6.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    df7= df_out
    exclude_indices = np.array([])
    for location, location_df in df7.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    df8= df7.drop(exclude_indices,axis='index')
    df9 = df8[df8.bath<df8.bhk+2]
    df10 = df9.drop(['size','price_per_sqft'],axis='columns')
    dummies = pd.get_dummies(df10.location)
    df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
    df12 = df11.drop('location',axis='columns')
    X = df12.drop(['price'],axis='columns')
    y = df12.price
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)
    lr_clf = LinearRegression()
    lr_clf.fit(X_train,y_train)
    lr_clf.score(X_test,y_test)
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    cross_val_score(LinearRegression(), X, y, cv=cv)
    print(X.head(20))
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = area
    x[1] = bathroom
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]
    

house_price("1st Block Jayanagar",1200,3.0,5)