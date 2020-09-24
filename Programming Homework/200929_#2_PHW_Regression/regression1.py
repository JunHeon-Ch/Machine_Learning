import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV


# Scaling
def scaling(df):
    scaled_df = pd.DataFrame(MinMaxScaler().fit_transform(df))
    scaled_df.columns = df.columns.values
    scaled_df.index = df.index.values

    return scaled_df


# Encoding
def encoder(df):
    cleaned_df = df.copy()
    features = ['sex', 'smoker', 'region']
    for feature in features:
        cleaned_df[feature] = LabelEncoder().fit_transform(cleaned_df[feature])

    return cleaned_df


def linear_regression(x, y):
    # Build model
    lr = LinearRegression()

    # Do k-fold cross validation
    mse = cross_val_score(estimator=lr, X=x, y=y, scoring='neg_mean_squared_error', cv=10)

    print('Linear Regression')
    print('Mean squared error: ', np.mean(mse))
    print('=======================================\n')


def ridge_regression(x, y, params):
    # Build model
    ridge = Ridge()

    # Do k-fold cross validation and tune parameter
    ridge_gscv = GridSearchCV(estimator=ridge, param_grid=params, scoring='neg_mean_squared_error', cv=10)

    ridge_gscv.fit(x, y)

    print('Ridge Regression')
    print('Best parameter: ', ridge_gscv.best_params_)
    print('Mean squared error: ', ridge_gscv.best_score_)
    print('=======================================\n')


def lasso_regression(x, y, params):
    # Build model
    lasso = Lasso()

    # Do k-fold cross validation and tune parameter
    lasso_gscv = GridSearchCV(estimator=lasso, param_grid=params, scoring='neg_mean_squared_error', cv=10)

    lasso_gscv.fit(x, y)

    print('Lasso Regression')
    print('Best parameter: ', lasso_gscv.best_params_)
    print('Mean squared error: ', lasso_gscv.best_score_)
    print('=======================================\n')


# Import dataset
df = pd.read_csv('insurance.csv')

# Encode dataset
encoded_df = encoder(df)
# Scale dataset
scaled_df = scaling(encoded_df)

# Separate predict columns and target column
x = scaled_df.drop(['charges'], axis=1)
y = scaled_df['charges']

# Hyper-parameter
params = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}

linear_regression(x, y)
ridge_regression(x, y, params)
lasso_regression(x, y, params)
