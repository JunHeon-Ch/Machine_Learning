import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from mpl_toolkits.mplot3d import axes3d


def dataset_preprocessing(df):
    cleaned_df = df.copy()

    # Data cleaning
    for i in range(len(cleaned_df)):
        if cleaned_df['car'][i] != 'unacc':
            cleaned_df['car'][i] = 'acc'

    # Encoding
    features = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'car']
    for feature in features:
        cleaned_df[feature] = LabelEncoder().fit_transform(cleaned_df[feature])

    # Scaling
    sc = MinMaxScaler()
    scaled_df = pd.DataFrame(sc.fit_transform(cleaned_df))
    scaled_df.columns = cleaned_df.columns.values
    scaled_df.index = cleaned_df.index.values

    return scaled_df


count = 0


# Print a confusion matrix using Seaborn
def confusionMatrix(y_test, y_pred, title):
    global count

    conf_matrix = confusion_matrix(y_test, y_pred)

    f, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", linewidths=10, ax=ax)

    plt.title(title, fontsize=20)
    ax.set_xticklabels(['acc', 'unacc'], fontsize=16)
    ax.set_yticklabels(['acc', 'unacc'], fontsize=16, rotation=360)

    count += 1
    plt.savefig(str(count) + '.png')


# Do k-fold cross validation
def K_fold(classifier, df, title):
    accuracies = []

    # Do k-fold cross validation
    cv = KFold(n_splits=10, shuffle=True, random_state=0)
    for i, (idx_train, idx_test) in enumerate(cv.split(df)):
        df_train = df.iloc[idx_train]
        df_test = df.iloc[idx_test]

        # Split independent columns and target column
        X_train = df_train.drop(columns='car')
        y_train = df_train['car']
        X_test = df_test.drop(columns='car')
        y_test = df_test['car']

        # fit and predict
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        # Compute the accuracy score
        accuracies.append(accuracy_score(y_test, y_pred))

        # Show confusion matrix
        new_title = str(i + 1) + ". " + title
        confusionMatrix(y_test, y_pred, new_title)

    return accuracies


# Display the accuracy score in the form of 3D bar chart
def bar_chart_3D(X, Y, Z, x_label, y_label, title):
    style.use('ggplot')

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')

    x3 = X
    y3 = Y
    z3 = np.zeros(Z.size)

    dx = np.ones(X.size)
    dy = np.ones(Y.size)
    dz = Z

    ax1.bar3d(x3, y3, z3, dx, dy, dz)

    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.set_zlabel('accuracy')

    plt.title(title)
    plt.savefig(title + '.png')


def trainRandomForest(df):
    # Parameter values
    param_grid = {'criterion': ['gini', 'entropy'],
                  'n_estimators': [1, 10, 100],
                  'max_depth': [1, 5, 8, 10]}

    best_score = 0
    best_parameter = {}
    x = []
    y = []
    z = []
    # For each combination of parameter values,
    # Do k-fold cross validation
    # Print a confusion matrix using Seaborn
    # Display the accuracy score in the form of 3D bar chart
    for criterion in param_grid['criterion']:
        for n_estimators in param_grid['n_estimators']:
            for max_depth in param_grid['max_depth']:
                # Build classifier model
                rfc = RandomForestClassifier(criterion=criterion, n_estimators=n_estimators, max_depth=max_depth)

                # K-fold cross validation
                title = "'criterion': " + str(criterion) + ", 'n_estimators': " + str(
                    n_estimators) + ", 'max_depth': " + str(max_depth)
                accuracies = K_fold(rfc, df, title + "\nRandom Forest Confusion Matrix")

                # Compute average score
                score = np.mean(accuracies)
                print('Random Forest Average Accuracy')
                print('criterion:', criterion, 'n_estimators:', n_estimators, 'max_depth:', max_depth)
                print('Mean accuracy: ', score)
                print('=============================\n')

                # Find the maximum score and the corresponding parameter values
                if score > best_score:
                    best_score = score
                    best_parameter = {'criterion': criterion, 'n_estimators': n_estimators, 'max_depth': max_depth}

                x.append(n_estimators)
                y.append(max_depth)
                z.append(score)

        # Display the accuracy score in the form of 3D bar chart
        title = 'Random Forest(' + criterion + ')'
        bar_chart_3D(np.array(x), np.array(y), np.array(z), 'n_estimators', 'max_depth', title)
        del x[:]
        del y[:]
        del z[:]

    # Print the maximum score and the corresponding parameter values
    print('Random Forest')
    print('Best score')
    print(best_score)
    print('Best parameter')
    print(best_parameter)
    print('=============================\n')


def trainLogisticRegression(df):
    # Parameter values
    param_grid = {'C': [0.1, 1.0, 10.0],
                  'solver': ['liblinear', 'lbfgs', 'sag'],
                  'max_iter': [50, 100, 200]}

    best_score = 0
    best_parameter = {}
    x = []
    y = []
    z = []
    # For each combination of parameter values,
    # Do k-fold cross validation
    # Print a confusion matrix using Seaborn
    # Display the accuracy score in the form of 3D bar chart
    for solver in param_grid['solver']:
        for C in param_grid['C']:
            for max_iter in param_grid['max_iter']:
                # Build classifier model
                lr = LogisticRegression(solver=solver, C=C, max_iter=max_iter)

                # K-fold cross validation
                title = "'solver': " + str(solver) + ", 'C': " + str(C) + ", 'max_iter': " + str(max_iter)
                accuracies = K_fold(lr, df, title + "Logistic Regression Confusion Matrix")

                # Compute average score
                score = np.mean(accuracies)
                print('Logistic Regression Average Accuracy')
                print('solver:', solver, 'C:', C, 'max_iter:', max_iter)
                print('Mean accuracy: ', score)
                print('=============================\n')

                # Find the maximum score and the corresponding parameter values
                if score > best_score:
                    best_score = score
                    best_parameter = {'solver': solver, 'C': C, 'max_iter': max_iter}

                x.append(C)
                y.append(max_iter)
                z.append(score)

        # Display the accuracy score in the form of 3D bar chart
        title = 'Logistic Regresion(' + solver + ')'
        bar_chart_3D(np.array(x), np.array(y), np.array(z), 'C', 'max_iter', title)
        del x[:]
        del y[:]
        del z[:]

    # Print the maximum score and the corresponding parameter values
    print('Logistic Regression')
    print('Best score')
    print(best_score)
    print('Best parameter')
    print(best_parameter)
    print('=============================\n')


def trainSVM(df):
    # Parameter values
    param_grid = {'C': [0.1, 1.0, 10.0],
                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                  'gamma': [0.01, 0.1, 1.0, 10.0]}

    best_score = 0
    best_parameter = {}
    x = []
    y = []
    z = []
    # For each combination of parameter values,
    # Do k-fold cross validation
    # Print a confusion matrix using Seaborn
    # Display the accuracy score in the form of 3D bar chart
    for kernel in param_grid['kernel']:
        for C in param_grid['C']:
            for gamma in param_grid['gamma']:
                # Build classifier model
                svc = SVC(kernel=kernel, C=C, gamma=gamma)

                # K-fold cross validation
                title = "'kernel': " + str(kernel) + ", 'C': " + str(C) + ", 'gamma': " + str(gamma)
                accuracies = K_fold(svc, df, title + "Support Vector Machine Confusion Matrix")

                # Compute average score
                score = np.mean(accuracies)
                print('Support Vector Machine Average Accuracy')
                print('kernel:', kernel, 'C:', C, 'gamma:', gamma)
                print('Mean accuracy: ', score)
                print('=============================\n')

                # Find the maximum score and the corresponding parameter values
                if score > best_score:
                    best_score = score
                    best_parameter = {'kernel': kernel, 'C': C, 'gamma': gamma}

                x.append(C)
                y.append(gamma)
                z.append(score)

        # Display the accuracy score in the form of 3D bar chart
        title = 'Support Vector Machine(' + kernel + ')'
        bar_chart_3D(np.array(x), np.array(y), np.array(z), 'C', 'gamma', title)
        del x[:]
        del y[:]
        del z[:]

    # Print the maximum score and the corresponding parameter values
    print('Support Vector Machine')
    print('Best score')
    print(best_score)
    print('Best parameter')
    print(best_parameter)
    print('=============================\n')


# Import dataset
df = pd.read_csv('car.csv', header=None)
df.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'car']

# Data preprocessing
# Encoding, Scaling, Data cleaning
processed_df = dataset_preprocessing(df)

# Random forest
trainRandomForest(processed_df)
# Logistic Regression
trainLogisticRegression(processed_df)
# SVM
trainSVM(processed_df)
