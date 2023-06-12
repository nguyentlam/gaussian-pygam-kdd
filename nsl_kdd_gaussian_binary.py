from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from pygam import LogisticGAM, s, f

def parseNumber(s):
    try:
        return float(s)
    except ValueError:
        return s

data_train = np.loadtxt('./KDDTrain+.txt', dtype =object, delimiter=',', encoding='latin1', converters=parseNumber)
data_test = np.loadtxt('./KDDTest+.txt', dtype =object, delimiter=',', encoding='latin1', converters=parseNumber)
print('len(data_train)', len(data_train))
print('len(data_test)', len(data_test))

N = 100000
X_train_raw = data_train[:, 0:41]
y_train_raw = data_train[:, [41]]
print('X_train_raw[0:3]===========', X_train_raw[0:3])
print('y_train_raw[0:5]===========', y_train_raw[0:5])
print('=================')

X_test_raw = data_test[:, 0:41]
y_test_raw = data_test[:, [41]]
print('X_test_raw[0:3]===========', X_test_raw[0:3])
print('y_test_raw[0:3]===========', y_test_raw[0:3])
print('=================')

x_columns = np.array(list(range(41)))
print('x_columns', x_columns)
categorical_x_columns = np.array([1, 2, 3])
numberic_x_columns = np.delete(x_columns, categorical_x_columns)
print('numberic_x_columns', numberic_x_columns)
x_ct = ColumnTransformer(transformers = [("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_x_columns),
                                         ('normalize', Normalizer(norm='max'), numberic_x_columns)], remainder = 'passthrough')

x_ct.fit(X_train_raw)
X_train = x_ct.transform(X_train_raw)
X_train = X_train.astype('float')
print('X_train[0:3]', X_train[0:3])
print('len(X_train[0])', len(X_train[0]))

X_test = x_ct.transform(X_test_raw)
X_test = X_test.astype('float')
print('X_train[0:3]', X_train[0:3])
print('len(X_train[0])', len(X_train[0]))


def preprocess_label(y_data):
    print(y_data[0])
    return np.array(list(map(lambda x : 0 if x[0] == 'normal' else 1, y_data)))

y_train = preprocess_label(y_train_raw)
y_train = y_train.astype('float')
print('y_train[0:2]===', y_train[0:2])

y_test = preprocess_label(y_test_raw)
y_test = y_test.astype('float')
print('y_test[0:2]===', y_test[0:2])

# Split the dataset into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_transformed, test_size=0.2, random_state=42)

# Train a logistic regression classifier on the training set
# clf = LogisticRegression(penalty=None, C=1e-6, solver='saga', multi_class='ovr', max_iter = 100)
# clf = DecisionTreeClassifier()
# clf = MLPClassifier(
#     hidden_layer_sizes=(40, 20, 40),
#     solver="adam",
#     activation='logistic',
#     alpha=1e-5,
#     max_iter=200,
#     learning_rate='constant',
#     learning_rate_init=0.2,
#     random_state=1,
#     verbose=True,
# )
clf = GaussianMixture(
    n_components=2, covariance_type="tied", max_iter=100, random_state=0, init_params='random_from_data'
    , reg_covar=1e-6
)

# clf = KMeans(
#     n_clusters=2
# )

clf.fit(X_train)

# Use the trained classifier to predict the classes of the test set
y_pred = clf.predict(X_test)
y_val = clf.predict(X_train)

y_2 = np.array(clf.predict_proba(X_test))

y_filterd = []
for y in y_2:
    if (y[0] > 0.2 and y[1] < 0.8) or (y[0] < 0.8 and y[1] > 0.2):
        y_filterd.append(y)
# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
validate = accuracy_score(y_train, y_val)
print("Validate:", validate)

# print("y_2", y_2[0:100])
print("len(y_filterd)", len(y_filterd))
print("y_filterd", y_filterd[0:100])
print("AIC:", clf.aic(X_test))
print("BIC:", clf.bic(X_test))
# print("means_", clf.means_)
# print("weights_", clf.weights_)
# print("covariances_", clf.covariances_)
# print("precisions_", clf.precisions_)
# print("precisions_cholesky_", clf.precisions_cholesky_)
# print("converged_", clf.converged_)
# print("n_iter_", clf.n_iter_)
# print("n_features_in_", clf.n_features_in_)
# print("lower_bound_", clf.lower_bound_)