from sklearn.base import BaseEstimator, TransformerMixin

class PropertyTypeCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, col_name='property_type'):
        self.col_name = col_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X.loc[X[self.col_name].str.contains(r'Entire|Tiny home', case=False, na=False), self.col_name] = 'Entire Unit'
        X.loc[X[self.col_name].str.contains(r'Shared', case=False, na=False), self.col_name] = 'Shared Space'
        X.loc[X[self.col_name].str.contains(r'[Rr]oom', case=False, na=False), self.col_name] = 'Private Room'
        X.loc[X[self.col_name].str.contains(r'Camp', case=False, na=False), self.col_name] = 'Camping Space'

        good_labels = ['Entire Unit', 'Private Room', 'Shared Space', 'Camping Space']
        X.loc[~X[self.col_name].isin(good_labels), self.col_name] = 'Other'
        return X
