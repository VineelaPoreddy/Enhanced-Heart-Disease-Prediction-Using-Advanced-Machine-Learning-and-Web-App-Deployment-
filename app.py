from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import joblib
import numpy as np
import secrets
import pandas as pd
import re  # Importing regular expression module for email validation

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)  # Secure random secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///yourdatabase.db'  # Adjust the URI as needed
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# Database models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    features = db.Column(db.String(500), nullable=False)
    result = db.Column(db.String(100), nullable=False)

# Load classifiers
classifiers = {
    'LogisticRegression': joblib.load('LogisticRegression_model.pkl'),
    'RandomForest': joblib.load('RandomForest_model.pkl'),
    'GradientBoosting': joblib.load('GradientBoosting_model.pkl'),
    'DecisionTree': joblib.load('DecisionTree_model.pkl'),
    'NaiveBayes': joblib.load('NaiveBayes_model.pkl'),
    'KNN': joblib.load('KNN_model.pkl'),
    'AdaBoost': joblib.load('Adaboost_model.pkl'),
    'VotingClassifier': joblib.load('VotingClassifier_model.pkl')
}

# Load classifier accuracies from saved file
classifier_accuracies = joblib.load('classifier_accuracies.pkl')

@app.route('/')
def home():
    return render_template('index.html')


from flask import flash, redirect, url_for

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        # Validate email format using regular expression
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            flash("Invalid email format. Please enter a valid email address.", 'danger')
            return redirect(url_for('register'))

        # Check if the email already exists in the database
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email address already exists. Please use a different one.', 'danger')
            return redirect(url_for('register'))

        # If email doesn't exist, create the new user
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(name=name, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Account created successfully!', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Validate email format using regular expression
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            flash("Invalid email format. Please enter a valid email address.", 'danger')
            return redirect(url_for('login'))

        user = User.query.filter_by(email=email).first()

        if user:
            if bcrypt.check_password_hash(user.password, password):
                session['user_id'] = user.id
                session['name'] = user.name
                flash("Login successful!", "success")
                return redirect(url_for('dashboard'))  # Redirect to dashboard
            else:
                flash("Incorrect password. Please try again.", "danger")
        else:
            flash("No account found with that email address. Please check your email or register.", "danger")

    return render_template('login.html')


@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash("Please log in to access the dashboard.", "danger")
        return redirect(url_for('login'))

    predictions = Prediction.query.filter_by(user_id=session['user_id']).all()
    return render_template('dashboard.html', name=session['name'], predictions=predictions)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Ensure user is logged in
    if 'user_id' not in session:
        flash("Please log in to access predictions.", "danger")
        return redirect(url_for('login'))
    # Debugging session
    print("Session user_id:", session.get('user_id'))

    if request.method == 'POST':
        print("POST request received")

        try:
            # Collect and convert input data
            features = [
                float(request.form['Age']),                # Age
                float(request.form['Diet']),              # Diet
                float(request.form['Smoking_Status']),    # Smoking_Status
                float(request.form['BMI']),               # BMI
                float(request.form['Diabetes']),          # Diabetes
                float(request.form['Stroke']),            # Stroke
                float(request.form['HighBP']),           # HighBP
                float(request.form['HighChol']),         # HighChol
                float(request.form['CholCheck']),        # CholCheck
                float(request.form['PhysActivity']), # PhysActivity
                float(request.form['HvyAlcoholConsump']),     # HvyAlcoholConsump
                float(request.form['AnyHealthcare']),        # AnyHealthcare
                float(request.form['Sex']),               # Sex
                float(request.form['Stress_Level'])       # Stress_Level
            ]

            # Define feature names
            feature_names = [
                'Age', 'Diet', 'Smoking_Status', 'BMI', 'Diabetes', 
                'Stroke', 'HighBP', 'HighChol', 'CholCheck', 
                'PhysActivity', 'HvyAlcoholConsump', 'AnyHealthcare', 
                'Sex', 'Stress_Level'
            ]

            # Convert to DataFrame
            input_data = pd.DataFrame([features], columns=feature_names)

            # Load the scaler and transform input data
            scaler = joblib.load('scaler.pkl')
            input_data_scaled = scaler.transform(input_data)

            # Convert scaled data back to a DataFrame with feature names
            input_data_scaled = pd.DataFrame(input_data_scaled, columns=feature_names)

            # Store results for all classifiers
            classifier_predictions = []
            results = []
            for classifier_name, clf in classifiers.items():
                # Get the prediction (0 or 1)
                prediction = clf.predict(input_data_scaled)[0]
                
                # Get the accuracy from the pre-calculated accuracies
                accuracy = classifier_accuracies.get(classifier_name, "N/A")  # Default to "N/A" if not found

                results.append({
                    'classifier': classifier_name,
                    'prediction': prediction,
                    'accuracy': accuracy  # Pass accuracy to the results
                })

            # Perform majority voting
            majority_prediction = int(sum([r['prediction'] for r in results]) > len(results) / 2)
            unified_output = "Heart Disease Detected" if majority_prediction == 1 else "No Heart Disease"

            features_dict = dict(zip(feature_names, features))  # Create a dictionary with feature names as keys
            import json
            # Store the features as a JSON string (as a dictionary)
            past_prediction = Prediction(
                user_id=session['user_id'],
                features=json.dumps(features_dict),  # Use json.dumps to store as JSON
                result=f"Unified Prediction: {unified_output}"
            )

            db.session.add(past_prediction)
            db.session.commit()

            # Render the results page
            return render_template('result.html', results=results, unified_output=unified_output)

        except Exception as e:
            print(f"Error in prediction: {e}")  # Log the error for debugging
            flash(f"An error occurred: {e}", "danger")
            return redirect(url_for('predict'))

    return render_template('predict.html')


@app.route('/result')
def result():
    results = session.pop('results', None)  # Correctly get and remove from session
    if results is None:
        flash("No predictions available.", "danger")
        return redirect(url_for('predict'))
    return render_template('result.html', results=results)


import json

@app.route('/history')
def history():
    if 'user_id' not in session:
        flash("Please log in to access history.", "danger")
        return redirect(url_for('login'))

    predictions = Prediction.query.filter_by(user_id=session['user_id']).all()

    # Convert features from JSON string back to a dictionary
    for prediction in predictions:
        prediction.features = json.loads(prediction.features)
        print("Prediction features:", prediction.features)  # Debugging print
        print("Type of features:", type(prediction.features))  # Debugging print to check the type
        # Deserialize from JSON

    return render_template('history.html', predictions=predictions)


if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # Create all tables
    app.run(debug=True)
