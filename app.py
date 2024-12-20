import joblib
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

# Load the trained model and the scaler
model = joblib.load('random_forest_model_knnimputed.pkl')
scaler = joblib.load('scaler.pkl')

# Define the encoding mappings as provided
sex_mapping = {'female': 0, 'male': 1}
housing_mapping = {'free': 0, 'own': 1, 'rent': 2}
saving_accounts_mapping = {'little': 0, 'moderate': 1, 'quite rich': 2, 'rich': 3}
checking_account_mapping = {'little': 0, 'moderate': 1, 'rich': 2}
purpose_mapping = {
    'business': 0, 'car': 1, 'domestic appliances': 2, 'education': 3, 'furniture/equipment': 4,
    'radio/TV': 5, 'repairs': 6, 'vacation/others': 7
}
job_mapping = {0: 0, 1: 1, 2: 2, 3: 3}  # Mapping for the Job feature (0, 1, 2, 3)

# Function to preprocess the new input data point
def preprocess_input(input_data):
    # Convert categorical features using the mappings
    input_data['Sex'] = sex_mapping.get(input_data['Sex'], -1)  # Default to -1 if invalid input
    input_data['Housing'] = housing_mapping.get(input_data['Housing'], -1)
    input_data['Saving accounts'] = saving_accounts_mapping.get(input_data['Saving accounts'], -1)
    input_data['Checking account'] = checking_account_mapping.get(input_data['Checking account'], -1)
    input_data['Purpose'] = purpose_mapping.get(input_data['Purpose'], -1)
    input_data['Job'] = job_mapping.get(input_data['Job'], -1)  # Handle the new Job feature
    
    # Make sure all values are valid
    if -1 in input_data.values():
        print("Error: Invalid categorical input value")
        return None
    
    # Normalize the continuous features using the previously fitted scaler
    continuous_cols = ['Age', 'Credit amount', 'Duration']
    
    # Only select the continuous features and normalize them
    continuous_data = np.array([input_data[col] for col in continuous_cols]).reshape(1, -1)
    normalized_values = scaler.transform(continuous_data)[0]  # Normalize continuous features
    
    # Update the input_data with the normalized values for continuous features
    for i, col in enumerate(continuous_cols):
        input_data[col] = normalized_values[i]
    
    # Return the processed input as a numpy array
    return np.array([list(input_data.values())])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        input_data = {
            'Age': float(request.form['Age']),
            'Sex': request.form['Sex'],
            'Housing': request.form['Housing'],
            'Saving accounts': request.form['Saving accounts'],
            'Checking account': request.form['Checking account'],
            'Credit amount': float(request.form['Credit amount']),
            'Duration': float(request.form['Duration']),
            'Purpose': request.form['Purpose'],
            'Job': int(request.form['Job'])
        }

        processed_input = preprocess_input(input_data)
        if processed_input is None:
            return render_template('form.html', error="Invalid data. Please try again.")

        prediction = model.predict(processed_input)
        result = "Good" if prediction == 1 else "Bad"
        return render_template('result.html', result=result)

    return render_template('form.html')

if __name__ == "__main__":
    app.run(debug=True)
