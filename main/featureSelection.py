import os
import numpy as np
import pandas as pd
import pingouin as pg
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import MinMaxScaler


def icc(X_train, X_test, y_train, y_test, param_prefix):
    # Read the feature data for the training set only
    data_train_1 = pd.read_csv(param_prefix + "A150feature_train.csv").filter(regex='^(?!diagnostics)')
    data_train_2 = pd.read_csv(param_prefix + "feature-other_train.csv").filter(regex='^(?!diagnostics)')

    # Insert reader column to distinguish between the two datasets
    data_train_1.insert(0, "reader", np.ones(data_train_1.shape[0]))
    data_train_2.insert(0, "reader", np.ones(data_train_2.shape[0]) * 2)

    # Insert target column representing row indices
    data_train_1.insert(0, "target", range(data_train_1.shape[0]))
    data_train_2.insert(0, "target", range(data_train_2.shape[0]))

    # Combine both datasets
    combined_data = pd.concat([data_train_1, data_train_2])

    # List to store columns to drop
    drop_columns = []

    # Iterate through columns to calculate ICC
    for i, column in enumerate(combined_data.columns):
        if i < 3:
            continue
        icc_result = pg.intraclass_corr(data=combined_data, targets="target", raters="reader", ratings=column)

        # Drop columns with ICC less than 0.80
        if icc_result.iloc[4, 2] < 0.80:
            drop_columns.append(column)

    # Print the number of dropped columns
    print(f"Number of dropped columns: {len(drop_columns)}")

    # Drop the identified columns from training and test datasets
    X_train = X_train.drop(drop_columns, axis=1)
    X_test = X_test.drop(drop_columns, axis=1)

    return X_train, X_test, y_train, y_test


def normalize(X_train, X_test, y_train, y_test, scaler):
    scaler.fit(X_train)  # Fit the scaler on the training data only
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Return normalized dataframes
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    return X_train_scaled_df, X_test_scaled_df, y_train, y_test


def apply_smote(X_train, X_test, y_train, y_test):
    X_resampled, y_resampled = SMOTE(random_state=0).fit_resample(X_train, y_train)
    return X_resampled, X_test, y_resampled, y_test


def perform_lasso(X_train, X_test, y_train, y_test):
    lambda_values = np.logspace(-5, 2, 1000)  # Lambda values from 10^-5 to 10^2
    lasso_coefficients = []

    # Train Lasso models with different lambda values
    for lambda_val in lambda_values:
        lasso = Lasso(alpha=lambda_val, random_state=0)
        lasso.fit(X_train, y_train)
        lasso_coefficients.append(lasso.coef_)

    # Use LassoCV for cross-validation
    lasso_cv = LassoCV(alphas=lambda_values, cv=5, random_state=0)
    lasso_cv.fit(X_train, y_train)

    # Print the best alpha value
    print("Best alpha value:", lasso_cv.alpha_)

    # Select features with non-zero coefficients
    selected_features = X_train.columns[lasso_cv.coef_ != 0]
    print('Selected features:', len(selected_features))

    # Filter training and test datasets to retain selected features
    X_train_selected = X_train.loc[:, selected_features]
    X_test_selected = X_test.loc[:, selected_features]

    return X_train_selected, X_test_selected, y_train, y_test


def radiomics(X_train, X_test, y_train, y_test, param_prefix):
    # Perform ICC feature selection
    X_train_filtered, X_test_filtered, y_train_filtered, y_test_filtered = icc(X_train, X_test, y_train, y_test,
                                                                               param_prefix)

    # Apply SMOTE for resampling
    X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled = apply_smote(X_train_filtered,
                                                                                           X_test_filtered,
                                                                                           y_train_filtered,
                                                                                           y_test_filtered)

    # Normalize the data
    scaler = MinMaxScaler()
    X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized = normalize(X_train_resampled,
                                                                                             X_test_resampled,
                                                                                             y_train_resampled,
                                                                                             y_test_resampled, scaler)

    # Perform Lasso feature selection
    X_train_lasso, X_test_lasso, y_train_lasso, y_test_lasso = perform_lasso(X_train_normalized, X_test_normalized,
                                                                             y_train_normalized, y_test_normalized)

    # Reset index for the test set
    y_test_lasso = y_test_lasso.reset_index(drop=True)

    # Save processed data
    train_file = pd.concat([X_train_lasso, y_train_lasso], axis=1)
    test_file = pd.concat([X_test_lasso, y_test_lasso], axis=1)

    save_path = os.path.join("saveFeature", param_prefix)
    os.makedirs(save_path, exist_ok=True)
    train_file.to_csv(os.path.join(save_path, param_prefix + "_train.csv"), index=False)
    test_file.to_csv(os.path.join(save_path, param_prefix + "_test.csv"), index=False)


def data_preprocess():
    param_prefix = "A150"

    # Read training data
    data_train = pd.read_csv(param_prefix + "A100feature_train.csv").filter(regex='^(?!diagnostics)')

    # Read test data
    data_test = pd.read_csv(param_prefix + "A100feature_test.csv").filter(regex='^(?!diagnostics)')

    # Split features and labels for training and testing
    X_train = data_train.iloc[:, 1:-1]
    y_train = data_train.iloc[:, -1]
    X_test = data_test.iloc[:, 1:-1]
    y_test = data_test.iloc[:, -1]

    # Call radiomics function to process features
    radiomics(X_train, X_test, y_train, y_test, param_prefix)

    return


if __name__ == '__main__':
    data_preprocess()
