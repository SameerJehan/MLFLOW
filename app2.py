from flask import Flask, render_template
import mlflow.sklearn
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load your data and features
data = pd.read_csv("group_WOE_11.csv")  # Replace with your actual data file
data2 = pd.read_csv("group_WOE_12.csv")  # Replace with your actual test data file
features = data.columns[5:]  # Adjust based on your feature selection logic

mlflow.set_tracking_uri("http://localhost:5000")  # Set the MLflow tracking URI to your server

# Specify the model URI (adjust the path if needed)
model_uri = "runs:/9cfe49c083d7493ab7c0fa68027d5264/logistic_regression_model"  # Replace <run_id> with the actual run ID

X_test = pd.read_csv("X_test_scaled.csv")  # Load the artifact into a DataFrame


# Load the model
loaded_model = mlflow.sklearn.load_model(model_uri)

# Use the loaded model for predictions
y_pred_proba = loaded_model.predict_proba(X_test)

# Function to get top customers by class
def log_top_customers_by_class(y_pred_proba, data, class_labels, top_n=10):

    top_customers_by_class = {}
    for class_idx, class_label in enumerate(class_labels):
        # Get probabilities for the current class
        class_probabilities = y_pred_proba[:, class_idx]

        # Get the indices of the top N probabilities
        top_indices = np.argsort(class_probabilities)[-top_n:][::-1]

        # Extract UNIQUE_CONSUMER_KEY and probabilities
        top_customers = data.iloc[top_indices].copy()
        top_customers["Probability"] = class_probabilities[top_indices]

        # Keep only UNIQUE_CONSUMER_KEY and Probability columns
        if "UNIQUE_CONSUMER_KEY" in top_customers.columns:
            top_customers = top_customers[["UNIQUE_CONSUMER_KEY", "Probability"]]
        else:
            raise KeyError("The column 'UNIQUE_CONSUMER_KEY' is not found in the dataset.")

        # Store the results in the dictionary
        top_customers_by_class[class_label] = top_customers

    return top_customers_by_class

@app.route("/")
def display_top_customers():
    # Define class labels (adjust based on your model)
    class_labels = ["Class 0", "Class 1", "Class 2"]

    class_descriptions = {
        "Class 0": "Didn't open account",
        "Class 1": "Opened and Defaulted",
        "Class 2": "Opened and Didn't Default",
    }

    # Get the top customers by class
    top_customers_by_class = log_top_customers_by_class(y_pred_proba, data2, class_labels, top_n=10)

    # Render the results in an HTML template
    return render_template("top_customers_group.html", 
                           top_customers_by_class=top_customers_by_class,
                            class_descriptions=class_descriptions)   

if __name__ == "__main__":
    app.run(debug=True, port = 5001)