import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc


def load_data(train_path, test_path):
    # Load training and test data from CSV files
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Split into features (X) and target labels (y)
    X_train = train_data.drop(columns=["target"])
    y_train = train_data["target"]
    X_test = test_data.drop(columns=["target"])
    y_test = test_data["target"]

    return X_train, X_test, y_train, y_test


def evaluate_model(y_true, y_pred, y_prob):
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Sensitivity (Recall or True Positive Rate)
    sensitivity = tp / (tp + fn)

    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp)

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Positive Predictive Value (PPV or Precision)
    ppv = precision_score(y_true, y_pred)

    # Negative Predictive Value (NPV)
    npv = tn / (tn + fn)

    # AUC (Area Under ROC Curve)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc_score = auc(fpr, tpr)

    # Print metrics
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"PPV: {ppv:.4f}")
    print(f"NPV: {npv:.4f}")
    print(f"AUC: {auc_score:.4f}")

    return sensitivity, specificity, accuracy, ppv, npv, auc_score


def train_and_evaluate(X_train, X_test, y_train, y_test):
    # Initialize Random Forest classifier with default parameters
    rf_classifier = RandomForestClassifier(random_state=0)

    # Train the model
    rf_classifier.fit(X_train, y_train)

    # Predict on the test set
    y_pred = rf_classifier.predict(X_test)

    # Predict probabilities for AUC
    y_prob = rf_classifier.predict_proba(X_test)[:, 1]

    # Evaluate the model
    evaluate_model(y_test, y_pred, y_prob)


def main():
    # Paths to the saved training and test data
    train_path = "saveFeature/A150_train.csv"
    test_path = "saveFeature/A150_test.csv"

    # Load data
    X_train, X_test, y_train, y_test = load_data(train_path, test_path)

    # Train and evaluate the model
    train_and_evaluate(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()
