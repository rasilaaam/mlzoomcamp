import pandas as pd
import numpy as np

df = pd.read_csv("car_fuel_efficiency_2026.csv")

# features, where target feature is going to be 'fuel_efficiency_mpg
features = ['model_year', 'engine_displacement', 'horsepower',
            'vehicle_weight', 'fuel_efficiency_mpg']

df = df[features]

# solution 1: horsepower column has 877 null values
print(df.isnull().sum())

# solution 2: median for variable horsepower is 254.0
print(df['horsepower'].median())

# preparing the datasets: train, val, and test

def prepare_dataset(df, target_var, seed=42):

    n = len(df)
    n_val = int(n * 0.2)
    n_test = int(n * 0.2)
    n_train = n - n_test - n_val

    np.random.seed(seed)
    idx = np.arange(n)
    np.random.shuffle(idx)

    df_train = df.iloc[idx[:n_train]]
    df_val = df.iloc[idx[n_train:n_train+n_val]]
    df_test = df.iloc[idx[n_train+n_val:]]

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = df_train[target_var].values
    y_val = df_val[target_var].values
    y_test = df_test[target_var].values

    del df_train[target_var]
    del df_val[target_var]
    del df_test[target_var]

    datasets = [df_train, df_val, df_test]
    target_sets = [y_train, y_val, y_test]

    return datasets, target_sets

df_train, df_val, df_test = prepare_dataset(df, 'fuel_efficiency_mpg')[0]
y_train, y_val, y_test = prepare_dataset(df, 'fuel_efficiency_mpg')[1]

# solution 3: rmse is lower with mean as substitute for na

def prepareX(df):
    # filling na in horsepower with mean
    # df = df.fillna(df['horsepower'].mean())

    #filling na in horsepower with 0s
    df = df.fillna(0)

    X = df.values

    return X

def train_linear_regression(X, y, r=0):
    ones = np.ones(np.shape(X)[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)

    XTX_inv = np.linalg.inv(XTX + r * np.eye(np.shape(XTX)[0]))

    w = XTX_inv.dot(X.T).dot(y)

    return w[0], w[1:]

def linear_regression(X, w):

    return w[0] + X.dot(w[1:])

def root_mean_square_error(y_pred, y_train):
    se = (y_pred - y_train) ** 2
    mse = se.mean()

    return np.sqrt(mse)

X_train = prepareX(df_train)

w0, w = train_linear_regression(X_train, y_train)

X_val = prepareX(df_val)

y_pred = linear_regression(X_val, np.insert(w, 0, w0))

rmse1 = root_mean_square_error(y_pred, y_val)
print(round(rmse1, 3))

# solution 4: 0 gives the best rmse
for r in [0, 0.01, 0.1, 1, 5, 10, 100]:
    w0, w = train_linear_regression(X_train, y_train, r)

    y_pred = linear_regression(X_val, np.insert(w, 0, w0))

    rmse2 = root_mean_square_error(y_pred, y_val)
    print(r, round(rmse2, 4))

# solution 5: 
rmse_scores = np.zeros(10)
for s in range(10):
    df_train, df_val, df_test = prepare_dataset(df, 'fuel_efficiency_mpg', s)[0]
    y_train, y_val, y_test = prepare_dataset(df, 'fuel_efficiency_mpg', s)[1]

    X_train = prepareX(df_train)

    w0, w = train_linear_regression(X_train, y_train)

    X_val = prepareX(df_val)

    y_pred = linear_regression(X_val, np.insert(w, 0, w0))

    rmse = root_mean_square_error(y_pred, y_val)

    rmse_scores[s] = rmse

std_deviation = np.std(rmse_scores)
print(round(std_deviation, 3))

# solution 6
df_train, df_val, df_test = prepare_dataset(df, 'fuel_efficiency_mpg', 9)[0]
y_train, y_val, y_test = prepare_dataset(df, 'fuel_efficiency_mpg', 9)[1]

df_full_train = pd.concat([df_train, df_val])
y_full_train = np.concatenate([y_train, y_val])

df_full_train = df_full_train.reset_index(drop=True)

X_full_train = prepareX(df_full_train)
w0, w = train_linear_regression(X_full_train, y_full_train, 0.001)

X_test = prepareX(df_test)

y_test_pred = linear_regression(X_test, np.insert(w, 0, w0))

rmse = root_mean_square_error(y_test_pred, y_test)

print(round(rmse, 3))

