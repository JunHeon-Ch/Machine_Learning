import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


# Read dataset
def read_dataset():
    dataset = pd.read_csv("mushrooms.csv")

    return dataset


# Standard Scaler
def standard_scaling(x):
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    scaled_x = pd.DataFrame(sc.fit_transform(x))
    scaled_x.columns = x.columns.values
    scaled_x.index = x.index.values

    return scaled_x


# Minmax Scaler
def minmax_scaling(x):
    from sklearn.preprocessing import MinMaxScaler

    sc = MinMaxScaler()
    scaled_x = pd.DataFrame(sc.fit_transform(x))
    scaled_x.columns = x.columns.values
    scaled_x.index = x.index.values

    return scaled_x


# Normalizer
def normalize_scaling(x):
    from sklearn.preprocessing import Normalizer

    sc = Normalizer()
    scaled_x = pd.DataFrame(sc.fit_transform(x))
    scaled_x.columns = x.columns.values
    scaled_x.index = x.index.values

    return scaled_x


# Label Encoder
def label_encoding(dataset):
    from sklearn.preprocessing import LabelEncoder

    encoded_data = dataset.copy()
    for col in encoded_data.columns:
        encoded_data[col] = LabelEncoder().fit_transform(encoded_data[col])

    return encoded_data


# Ordinal Encoder
def ordinal_encoding(dataset):
    from sklearn.preprocessing import OrdinalEncoder

    encoded_data = pd.DataFrame(OrdinalEncoder().fit_transform(dataset))
    encoded_data.columns = dataset.columns.values
    encoded_data.index = dataset.index.values

    return encoded_data


# Purity of a given clustering result.
def calculate_purity(dataset, cluster):
    elements = np.unique(cluster)

    # If length of cluster list is 1 and cluster is 1, all data was not clustered.
    # Return -1.
    if len(elements) == 1 and elements[0] == -1:
        return -1

    cluster = pd.DataFrame(cluster)
    cluster.columns = ['cluster']
    c_dataset = pd.concat([dataset, cluster], axis=1)

    # Calculate purity
    sum_purity = 0
    for element in elements:
        # If cluster is 1, this means outliers.
        if element == -1:
            continue
        sum_purity += c_dataset[c_dataset['cluster'] == element]['class'].value_counts().max()

    # Return calculated purity
    return sum_purity / len(c_dataset)


# Draw a scatter plot to visualize the clustering result.
# To show high-dimensional data in two dimensions, use PCA.
def c_scatter(data, c, title):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data.drop(columns='class'))
    principal_df = pd.DataFrame(data=principal_components, columns=['principal component1', 'principal component2'])

    plt.scatter(x=principal_df['principal component1'], y=principal_df['principal component2'], c=c)
    plt.title(title)
    plt.savefig(title + '.png')


def dbscan(dataset, best_purity, max_purity):
    # hyper-parameter
    params = {
        'eps': [0.05, 0.1],
        'min_samples': [5, 10, 20, 50, 100],
        'algorithm': ['ball_tree', 'kd_tree', 'brute'],
        'metric': ['euclidean', 'hamming']
        # Use 'minkowski'
        # 'metric': ['minkowski'],
        # 'p': [1, 2, 3]
    }

    # Compute and compare the purity for various combinations of hyper-parameters.
    for eps in params['eps']:
        for min_sample in params['min_samples']:
            for algo in params['algorithm']:
                for metric in params['metric']:
                    # Does not work when algorithm is kd_tree and the metric is hamming.
                    if algo == 'kd_tree' and metric == 'hamming':
                        continue

                    # Create model, fit and predict using hyper-parameter.
                    dbs = DBSCAN(eps=eps, min_samples=min_sample, algorithm=algo, metric=metric)
                    x = dataset.drop(columns='class')
                    cluster = dbs.fit_predict(x)

                    # Draw a scatter plot to visualize the clustering result.
                    title = 'DBSCAN (eps={}, min_samples={}, algo={}, metric={})'\
                        .format(eps, min_sample, algo, metric)

                    # Purity of a given clustering result.
                    purity_rate = calculate_purity(dataset, cluster)
                    # Find the best purity.
                    if max_purity < purity_rate:
                        max_purity = purity_rate
                        del best_purity[:]
                        best_purity.append((title, purity_rate))
                    elif max_purity == purity_rate:
                        best_purity.append((title, purity_rate))

                    # if purity is -1, all data was not clustered.
                    if purity_rate != -1:
                        print(title)
                        print('purity: %2f' % purity_rate)
                        c_scatter(dataset, cluster, title)

    return max_purity


ds = read_dataset()

best_purity = []
max_purity = 0
# Label encoding
processed_ds = label_encoding(ds)
print('========== Label encoding ==========')
max_purity = dbscan(processed_ds, best_purity, max_purity)

# Label encoding and standard scaling
processed_ds = label_encoding(ds)
x = processed_ds.drop(columns='class')
x = standard_scaling(x)
y = processed_ds['class']
processed_ds = pd.concat([x, y], axis=1)
print('========== Label encoding and standard scaling ==========')
max_purity = dbscan(processed_ds, best_purity, max_purity)

# Label encoding and minmax scaling
processed_ds = label_encoding(ds)
x = processed_ds.drop(columns='class')
x = minmax_scaling(x)
y = processed_ds['class']
processed_ds = pd.concat([x, y], axis=1)
print('========== Label encoding and minmax scaling ==========')
max_purity = dbscan(processed_ds, best_purity, max_purity)

# Label encoding and normalize scaling
processed_ds = label_encoding(ds)
x = processed_ds.drop(columns='class')
x = normalize_scaling(x)
y = processed_ds['class']
processed_ds = pd.concat([x, y], axis=1)
print('========== Label encoding and normalize scaling ==========')
max_purity = dbscan(processed_ds, best_purity, max_purity)

# Ordinal encoding
processed_ds = ordinal_encoding(ds)
print('========== Ordinal encoding ==========')
max_purity = dbscan(processed_ds, best_purity, max_purity)

# Ordinal encoding and standard scaling
processed_ds = ordinal_encoding(ds)
x = processed_ds.drop(columns='class')
x = standard_scaling(x)
y = processed_ds['class']
processed_ds = pd.concat([x, y], axis=1)
print('========== Ordinal encoding and standard scaling ==========')
max_purity = dbscan(processed_ds, best_purity, max_purity)

# Ordinal encoding and minmax scaling
processed_ds = ordinal_encoding(ds)
x = processed_ds.drop(columns='class')
x = minmax_scaling(x)
y = processed_ds['class']
processed_ds = pd.concat([x, y], axis=1)
print('========== Ordinal encoding and minmax scaling ==========')
max_purity = dbscan(processed_ds, best_purity, max_purity)

# Ordinal encoding and normalize scaling
processed_ds = ordinal_encoding(ds)
x = processed_ds.drop(columns='class')
x = normalize_scaling(x)
y = processed_ds['class']
processed_ds = pd.concat([x, y], axis=1)
print('========== Ordinal encoding and normalize scaling ==========')
max_purity = dbscan(processed_ds, best_purity, max_purity)

print('========== Best purity set ==========')
for purity in best_purity:
    print(purity[0])
    print('purity: %2f\n' % purity[1])
