import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


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


def elastic_net_regression(x, y):
    # Build model
    en = ElasticNet()
    # Hyper-parameter
    params = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20],
              'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}

    # Do k-fold cross validation and tune parameter
    en_gscv = GridSearchCV(estimator=en, param_grid=params, scoring='neg_mean_squared_error', cv=10)

    en_gscv.fit(x, y)

    print('Elastic Net Regression')
    print('Best parameter: ', en_gscv.best_params_)
    print('Mean squared error: ', en_gscv.best_score_)
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

elastic_net_regression(x, y)
