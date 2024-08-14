from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model from the pickle file
with open('D:/Lung Disease Prediction System/liver_1.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/input', methods=['GET', 'POST'])
def input():
    if request.method == 'POST':
        try:
            # Retrieve form data
            data = [
                request.form.get('age'),
                request.form.get('gender'),
                request.form.get('total_bilirubin'),
                request.form.get('direct_bilirubin'),
                request.form.get('alkaline_phosphotase'),
                request.form.get('alamine_aminotransferase'),
                request.form.get('aspartate_aminotransferase'),
                request.form.get('total_proteins'),
                request.form.get('albumin'),
                request.form.get('albumin_globulin_ratio'),
            ]

            # Convert the data to the appropriate format
            data = np.array(data).reshape(1, -1).astype(float)

            # Make the prediction
            prediction = model.predict(data)
            probability = model.predict_proba(data)

            return render_template('result.html', prediction=prediction[0], probability=probability[0][1])

        except Exception as e:
            print(e)
            return render_template('input.html', error='Invalid input. Please enter valid values.')
    else:
        return render_template('input.html')

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
