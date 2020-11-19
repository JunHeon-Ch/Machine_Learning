import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


# Read dataset
def read_dataset():
    dataset = pd.read_csv("flu.csv")

    return dataset


# Label encoding
def label_encoding(dataset):
    from sklearn.preprocessing import LabelEncoder

    encoded_data = dataset.copy()
    for col in encoded_data.columns:
        encoded_data[col] = LabelEncoder().fit_transform(encoded_data[col])

    return encoded_data


# Minmax scaling
def minmax_scaling(x):
    from sklearn.preprocessing import MinMaxScaler

    sc = MinMaxScaler()
    scaled_x = pd.DataFrame(sc.fit_transform(x))
    scaled_x.columns = x.columns.values
    scaled_x.index = x.index.values

    return scaled_x


dataset = read_dataset()
encoded_dataset = label_encoding(dataset)
x = encoded_dataset.drop(columns='flu')
y = encoded_dataset['flu']
x = minmax_scaling(x)

# In the 40 data, 30 data are used for training and 10 data are used for testing.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
# Create a Gaussian classifier
nb = GaussianNB()
# Train the model using the training sets.
nb.fit(x_train, y_train)
# Predict output.
predicted = nb.predict(x_test)
print('===== Accuracy =====')
print(nb.score(predicted.reshape(-1, 1), y_test))
print('\n===== Actual and Predicted value =====')
for a, p in zip(y_test, predicted):
    print('Actual value:', a, ', Predicted value:', p)