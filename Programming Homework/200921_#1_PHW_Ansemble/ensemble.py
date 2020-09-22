import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import style
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

count = 0


# Scaling
def scaling(df):
    scaled_df = pd.DataFrame(MinMaxScaler().fit_transform(df))
    scaled_df.columns = df.columns.values
    scaled_df.index = df.index.values

    return scaled_df


# Print a confusion matrix using Seaborn
def make_confusion_matrix(y_test, y_pred, title):
    global count

    conf_matrix = confusion_matrix(y_test, y_pred)

    f, ax = plt.subplots(figsize=(14, 16))
    sns.heatmap(conf_matrix, annot=True, fmt="d", linewidths=10, ax=ax)

    plt.title(title, fontsize=20)
    ax.set_xticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], fontsize=12)
    ax.set_yticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], fontsize=12, rotation=360)
    ax.set_xlabel('Predicted', fontsize=16)
    ax.set_ylabel('Actual', fontsize=16)

    count += 1
    plt.savefig(str(count) + '.png')


# Display the accuracy score in the form of 3D bar chart
def bar_chart_3d(x, y, z, x_label, y_label, title):
    style.use('ggplot')

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')

    x3 = x
    y3 = y
    z3 = np.zeros(z.size)

    dx = np.ones(x.size)
    dy = np.ones(y.size)
    dz = z

    ax1.bar3d(x3, y3, z3, dx, dy, dz)

    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.set_zlabel('accuracy')

    plt.title(title)
    plt.savefig(title + '.png')


# Do k-fold cross validation
def k_fold(classifier, df):
    accuracies = []
    y_tests = []
    y_preds = []

    # Do k-fold cross validation
    cv = KFold(n_splits=10, shuffle=True, random_state=0)
    for i, (idx_train, idx_test) in enumerate(cv.split(df)):
        df_train = df.iloc[idx_train]
        df_test = df.iloc[idx_test]

        # Split independent columns and target column
        x_train = df_train.drop(columns='label')
        y_train = df_train['label']
        x_test = df_test.drop(columns='label')
        y_test = df_test['label']

        # fit and predict
        classifier.fit(x_train.values, y_train)
        y_pred = classifier.predict(x_test.values)

        # Compute the accuracy score
        accuracies.append(accuracy_score(y_test, y_pred))
        y_tests.append(y_test)
        y_preds.append(y_pred)

    return accuracies, y_tests, y_preds


def train_random_forest(df):
    # Parameter values
    param_grid = {'criterion': ['gini', 'entropy'],
                  'n_estimators': [200, 300, 500],
                  'max_depth': [8, 10, 12]}

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
                accuracies, y_tests, y_preds = k_fold(rfc, df)

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

                # For 3D bar chart, store parameters and accuracies
                x.append(n_estimators)
                y.append(max_depth)
                z.append(score)

                # For each combination of parameter values, show confusion matrix about best score
                cm_title = "criterion: {0}, n_estimators: {1}, max_depth: {2}\nRandom Forest Confusion Matrix" \
                    .format(criterion, n_estimators, max_depth)
                make_confusion_matrix(y_tests[accuracies.index(max(accuracies))],
                                      y_preds[accuracies.index(max(accuracies))], cm_title)

        # Display the accuracy score in the form of 3D bar chart
        td_title = 'Random Forest(' + criterion + ')'
        bar_chart_3d(np.array(x), np.array(y), np.array(z), 'n_estimators', 'max_depth', td_title)
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

    # Build best classifier model
    x_train = df.drop(columns='label')
    y_train = df['label']
    best_rfc = RandomForestClassifier(criterion=best_parameter['criterion'],
                                      n_estimators=best_parameter['n_estimators'],
                                      max_depth=best_parameter['max_depth']).fit(x_train, y_train)

    return best_rfc


def train_logistic_regression(df):
    # Parameter values
    param_grid = {'C': [0.1, 1.0, 10.0],
                  'solver': ['liblinear', 'lbfgs'],
                  'max_iter': [600, 800]}
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
                accuracies, y_tests, y_preds = k_fold(lr, df)

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

                # For each combination of parameter values, show confusion matrix about best score
                cm_title = "solver: {0}, C: {1}, max_iter: {2}\nLogistic Regression Confusion Matrix" \
                    .format(solver, C, max_iter)
                make_confusion_matrix(y_tests[accuracies.index(max(accuracies))],
                                      y_preds[accuracies.index(max(accuracies))], cm_title)

        # Display the accuracy score in the form of 3D bar chart
        td_title = 'Logistic Regression(' + solver + ')'
        bar_chart_3d(np.array(x), np.array(y), np.array(z), 'C', 'max_iter', td_title)
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

    # Build best classifier model
    x_train = df.drop(columns='label')
    y_train = df['label']
    best_lr = LogisticRegression(solver=best_parameter['solver'], C=best_parameter['C'],
                                 max_iter=best_parameter['max_iter']).fit(x_train, y_train)

    return best_lr


def train_SVM(df):
    # Parameter values
    param_grid = {'C': [0.1, 1.0, 10.0],
                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                  'gamma': [0.1, 1.0, 10.0]}

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
                accuracies, y_tests, y_preds = k_fold(svc, df)

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

                # For each combination of parameter values, show confusion matrix about best score
                cm_title = "kernel: {0}, C: {1}, gamma: {2}\nSupport Vector Machine Confusion Matrix" \
                    .format(kernel, C, gamma)
                make_confusion_matrix(y_tests[accuracies.index(max(accuracies))],
                                      y_preds[accuracies.index(max(accuracies))], cm_title)

        # Display the accuracy score in the form of 3D bar chart
        td_title = 'Support Vector Machine(' + kernel + ')'
        bar_chart_3d(np.array(x), np.array(y), np.array(z), 'C', 'gamma', td_title)
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

    # Build best classifier model
    x_train = df.drop(columns='label')
    y_train = df['label']
    best_svm = SVC(kernel=best_parameter['kernel'], C=best_parameter['C'], gamma=best_parameter['gamma']) \
        .fit(x_train, y_train)

    return best_svm


# Find the most frequent element
def majority_count(column):
    elements, count = np.unique(column, return_counts=True)

    # If all classifiers have different predictions, choose the one returned by the classifier with the highest score
    if len(elements) == len(column):
        return np.max(elements)
    else:
        return elements[count.argmax()]


# Approach without sklearn.ensemble.VotingClassifier
def voting_without_classifier(classifiers, df):
    x_test = df.drop(columns='label')
    y_test = df['label']

    # Data frame for storing predicted results using classifier models
    results = pd.DataFrame(index=np.arange(len(classifiers)), columns=np.arange(len(df)))

    # predict the labels with ensemble dataset
    for i in range(len(classifiers)):
        pred = classifiers[i].predict(x_test)

        # Store result into DataFrame
        for j in range(len(df)):
            results.iat[i, j] = pred[j]

    # The majority vote determines the final forecast
    final_result = []
    for i in range(len(df)):
        final_result.append(majority_count(results.iloc[:, i]))

    # Display a confusion matrix
    make_confusion_matrix(y_test, final_result, "Confusion Matrix without classifier")

    # Display a accuracy
    print('Ensemble without classifier')
    print(accuracy_score(y_test, final_result))
    print('=============================\n')


# Approach with sklearn.ensemble.VotingClassifier
def voting_with_classifier(classifiers, df):
    x_test = df.drop(columns='label')
    y_test = df['label']

    # Fit and predict the labels using VotingClassifier
    vc = VotingClassifier(estimators=[('0', classifiers[0]), ('1', classifiers[1]), ('2', classifiers[2])],
                          voting='hard').fit(x_test, y_test)
    y_pred = vc.predict(x_test)

    # Display a confusion matrix
    make_confusion_matrix(y_test, y_pred, "Confusion Matrix with classifier")

    # Display a accuracy
    print('Ensemble with classifier')
    print(accuracy_score(y_test, y_pred))
    print('=============================\n')


# Import dataset
df = pd.read_csv('mnist.csv')

# Scale the numerical columns
x = df.drop(columns='label')
y = df['label']
x = scaling(x)
scaled_df = pd.concat([x, y], axis=1)

original, test = train_test_split(df, test_size=0.2, random_state=42)

# Split the dataset into two sub-datasets (10% of dataset)
training, ensemble = train_test_split(test, test_size=0.5, random_state=42)

classifiers = []

# Random forest
rf = train_random_forest(training)
classifiers.append(rf)

# Logistic Regression
lr = train_logistic_regression(training)
classifiers.append(lr)

# SVM
svm = train_SVM(training)
classifiers.append(svm)

x_ensemble = ensemble.drop(columns='label')
y_ensemble = ensemble['label']

# Test the ensemble classifier model
voting_without_classifier(classifiers, ensemble)
voting_with_classifier(classifiers, ensemble)

# Make confusion matrix using each of the three classifiers
rf_pred = rf.predict(x_ensemble)
make_confusion_matrix(y_ensemble, rf_pred, "Random Forest with Best parameter")
print('Random Forest with Best parameter')
print(accuracy_score(y_ensemble, rf_pred))
print('=============================\n')

lr_pred = lr.predict(x_ensemble)
make_confusion_matrix(y_ensemble, lr_pred, "Logistic Regression with best parameter")
print('Random Forest with Best parameter')
print(accuracy_score(y_ensemble, lr_pred))
print('=============================\n')

svm_pred = svm.predict(x_ensemble)
make_confusion_matrix(y_ensemble, svm_pred, "Support Vector Machine with Best parameter")
print('Random Forest with Best parameter')
print(accuracy_score(y_ensemble, svm_pred))
print('=============================\n')
