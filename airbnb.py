# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 18:23:33 2024

@author: neall
"""
#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

listings = pd.read_csv(r"listings.csv")
listings.columns

listings = listings[['host_is_superhost', 'host_listings_count', 'host_total_listings_count', 'latitude', 'longitude', 'property_type', 
          'accommodates', 'beds', 'price', 'minimum_nights', 'maximum_nights', 'number_of_reviews', 'review_scores_rating', 'review_scores_cleanliness', 
          'review_scores_communication', 'review_scores_value']]

#%%

listings = listings.dropna(subset='price')

def clean_price(price_column):
    #cleans and converts price to strings  
    price_column = price_column.str.strip('$').str.replace(',', '').reset_index()['price']
    price_column = price_column.str[:-3]
    price_column = price_column.astype(int)
    return price_column

listings['price'] = clean_price(listings['price'])

listings = listings[listings['price'] < listings['price'].quantile(.95)]

def superhost_numeric(host_is_superhost):
    host_is_superhost = host_is_superhost.map({'f':0, 't':1})
    return host_is_superhost

listings['host_is_superhost'] = superhost_numeric(listings['host_is_superhost'])

def clean_property_type(property_type_col):
    #consolidate the property_type var to a few common categories   
    property_type_col[property_type_col.str.contains(r'Entire')] = 'Entire Unit'
    property_type_col[property_type_col.str.contains(r'Tiny home')] = 'Entire Unit'
    property_type_col[property_type_col.str.contains(r'[Rr]oom')] = 'Private Room'
    property_type_col[property_type_col.str.contains(r'Camp')] = 'Camping'
    
    bad_values = ['Shipping container', 'Shared room in rental unit', 'Tent', 'Farm stay', 'Treehouse', 'Yurt', 'Shared room in hostel']
    property_type_col = property_type_col[~property_type_col.isin(bad_values)]
    return property_type_col

listings['property_type'] = clean_property_type(listings['property_type'])


#%%

listings.hist(bins=50, figsize=(15,10))



#%%

corr = listings.corr(numeric_only=True)

corr['price'].sort_values(ascending=False)

#%%


listings.plot.scatter(x = 'longitude', y = 'latitude', grid=True, 
                       c='price', cmap='jet', colorbar=True,
                      legend=True, figsize=(10,7))


