# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 18:23:33 2024

@author: neall
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r"listings.csv")
df.columns

#%%
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

listings = train_set[['price', 'host_is_superhost','host_listings_count',
                      'accommodates', 'bathrooms_text', 'beds', 'minimum_nights', 'maximum_nights', 
                      'number_of_reviews', 'review_scores_rating', 'property_type']]
# does not include longitude and latitude

#%%

def clean_price(price_column):
    #cleans and converts price to strings  
    price_column = price_column.str.strip('$').str.replace(',', '').reset_index()['price']
    price_column = price_column.str[:-3]
    price_column = pd.to_numeric(price_column, errors='raise')
    return price_column

listings['price'] = clean_price(listings['price'])
listings = listings[(listings['price'] < listings['price'].quantile(.99)) & (listings['price'] > listings['price'].quantile(.01))]

'''
def superhost_numeric(host_is_superhost):
    #converts this column to numeric
    host_is_superhost = host_is_superhost.map({'f':0, 't':1})
    return host_is_superhost

listings['host_is_superhost'] = superhost_numeric(listings['host_is_superhost'])
'''

def clean_bathrooms(bathrooms_text):
    #cleans the bathrooms... har har har... no actually it cleans the bathrooms_text column
    pattern = r'(\d.?\d?)\s'
    pattern2 = r'(Half)'
    bathrooms_text[bathrooms_text.str.contains(pattern2, na=False)] = .5
    bathrooms_text = bathrooms_text.str.extract(pattern)
    return bathrooms_text.astype(float)

listings['bathrooms'] = clean_bathrooms(listings['bathrooms_text'])
listings.drop(columns='bathrooms_text', inplace=True)


def clean_property_type(property_type_col):
    #consolidate the property_type var to a few common categories   
    property_type_col[property_type_col.str.contains(r'Entire')] = 'Entire Unit'
    property_type_col[property_type_col.str.contains(r'Tiny home')] = 'Entire Unit'
    property_type_col[property_type_col.str.contains(r'[Rr]oom')] = 'Private Room'
    property_type_col[property_type_col.str.contains(r'Camp')] = 'Camping'
    
    bad_values = ['Shipping container', 'Bus', 'Casa particular', 'Shared room in rental unit', 'Tent', 'Farm stay', 'Treehouse', 'Yurt', 'Shared room in hostel']
    property_type_col = property_type_col[~property_type_col.isin(bad_values)]
    return property_type_col

listings['property_type'] = clean_property_type(listings['property_type'])

#%%

listings['price_log'] = listings['price'].apply(np.log)

listings['price_log'].hist(bins=50, figsize=(12,8))

#%%

y = listings[['price_log']]
listings = listings.drop(columns=['price', 'price_log'])
#%%

#dealing with missings in numeric columns

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')

listings_num = listings.drop(columns=['property_type', 'host_is_superhost'])
imputer.fit(listings_num)
listings_num = imputer.transform(listings_num)

#%%

#dealing with categorical columns

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(drop='if_binary', sparse_output=False)
cat_imputer = SimpleImputer(strategy='most_frequent')

listings_cat = listings[['property_type', 'host_is_superhost']]
listings_cat = cat_imputer.fit_transform(listings_cat)
listings_cat = encoder.fit_transform(listings_cat)

#%%

listings.info()
# %%

encoder.get_feature_names_out()

# %%

listings['property_type'].value_counts()
# %%
