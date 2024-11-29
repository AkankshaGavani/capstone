# save this as app.py
from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__, template_folder='templates')
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            gender = request.form.get('gender')
            married = request.form.get('married')
            dependents = request.form.get('dependents')
            education = request.form.get('education')
            employed = request.form.get('employed')
            credit = request.form.get('credit')
            area = request.form.get('area')
            applicant_income = request.form.get('ApplicantIncome')
            coapplicant_income = request.form.get('CoapplicantIncome')
            loan_amount = request.form.get('LoanAmount')
            loan_amount_term = request.form.get('Loan_Amount_Term')

            # Convert categorical values to numerical format
            gender_map = {'Male': 1, 'Female': 0}
            married_map = {'Yes': 1, 'No': 0}
            education_map = {'Graduate': 1, 'Not Graduate': 0}
            employed_map = {'Yes': 1, 'No': 0}
            area_map = {'Urban': 2, 'Semiurban': 1, 'Rural': 0}
            dependents_map = {'0': 0, '1': 1, '2': 2, '3+': 3}

            # Perform mappings
            gender = gender_map.get(gender, -1)
            married = married_map.get(married, -1)
            education = education_map.get(education, -1)
            employed = employed_map.get(employed, -1)
            area = area_map.get(area, -1)
            dependents = dependents_map.get(dependents, -1)
            credit = float(credit) if credit else 0.0
            applicant_income = float(applicant_income)
            coapplicant_income = float(coapplicant_income)
            loan_amount = float(loan_amount)
            loan_amount_term = float(loan_amount_term)

            # Add default values for missing features (update based on model requirements)
            additional_feature_1 = 0  # Replace with actual default
            additional_feature_2 = 0  # Replace with actual default
            additional_feature_3 = 0  # Replace with actual default

            # Validate inputs
            if -1 in [gender, married, education, employed, area, dependents]:
                return render_template(
                    'prediction.html', prediction_text="Error: Please fill all fields correctly."
                )

            # Prepare input for the model (add missing features)
            input_data = np.array([
                gender, married, dependents, education, employed,
                credit, area, applicant_income, coapplicant_income,
                loan_amount, loan_amount_term,
                additional_feature_1, additional_feature_2, additional_feature_3
            ]).reshape(1, -1)

            # Get prediction
            prediction = model.predict(input_data)
            prediction_text = "Loan Denied" if prediction[0] == 1 else "Loan Approved"

        except ValueError as e:
            prediction_text = f"Error in input: {str(e)}"
        except Exception as e:
            prediction_text = f"Error in prediction: {str(e)}"

        # Render the result page with prediction
        return render_template('prediction.html', prediction_text=prediction_text)

    else:
        return render_template('predict.html')  # Form page

@app.route('/about')
def about():
    return render_template('about.html')

# Contact page
@app.route('/contact')
def contact():
    return render_template('contact.html')

# Services page
@app.route('/faq')
def services():
    return render_template('faq.html')


if __name__ == "__main__":
    app.run(debug=True)