import os
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_cors import CORS
import pandas as pd
import csv
import random
import datetime
from collections import defaultdict
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy

# Load the spacy NLP model
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)

# Load data
symptoms_conditions_df = pd.read_csv('data/symptoms_conditions1.csv')
conditions_treatments_df = pd.read_csv('data/conditions_treatments.csv', encoding='ISO-8859-1')

symptoms_conditions_dict = symptoms_conditions_df.groupby('Symptom')['Condition'].apply(list).to_dict()
conditions_treatments_dict = conditions_treatments_df.groupby('Condition')['Treatment'].apply(list).to_dict()

# User state
user_state = defaultdict(lambda: {'name': None, 'conversation_stage': 'get_name', 'condition': None, 'duration': None, 'symptoms': []})

# Load doctors and appointments
def load_csv(filename):
    with open(filename, 'r') as f:
        return list(csv.DictReader(f))

doctors = load_csv('data/doctors.csv')
appointments = load_csv('data/appointments.csv')

# Save appointment to appointments.csv
def save_appointment(appointment):
    fieldnames = ['ID', 'Name', 'Time', 'Date', 'Illness', 'Doctor', 'Title', 'Description']
    with open('appointments.csv', 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if f.tell() == 0:  # If file is empty, write header
            writer.writeheader()
        writer.writerow(appointment)

def get_available_slots(date):
    booked_slots = [apt['Time'] for apt in appointments if apt['Date'] == date]
    all_slots = [f"{h:02d}:00" for h in range(8, 21)]  # Expanded to cover all hours from 8 AM to 8 PM
    return [slot for slot in all_slots if slot not in booked_slots]

def find_closest_slot(preferred_time, available_slots):
    preferred_minutes = int(preferred_time[:2]) * 60 + int(preferred_time[3:])
    closest_slot = min(available_slots, key=lambda x: abs(preferred_minutes - (int(x[:2]) * 60 + int(x[3:]))))
    return closest_slot

def get_greeting(name):
    greetings = [
        f"👋 Hello {name}! How can I assist you today?",
        f"Hi there, {name}! 😊 What brings you here?",
        f"Greetings, {name}! 🌟 How may I help you?",
        f"Welcome, {name}! 🤗 What would you like to know?",
        f"Hey {name}! 👨‍⚕️ How can I be of service today?"
    ]
    return random.choice(greetings)

def is_greeting(message):
    greetings = ['hi', 'hello', 'hey', 'greetings', 'hola']
    return any(greeting in message.lower() for greeting in greetings)

def preprocess_text(text):
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    return [token for token in tokens if token not in stop_words]

def match_symptoms(user_input):
    # Use spaCy NLP pipeline to process the text
    doc = nlp(user_input.lower())  # Lowercase for consistency
    matched_symptoms = []

    # Process each symptom and check if it's mentioned in the user input
    for symptom in symptoms_conditions_dict.keys():
        symptom_doc = nlp(symptom.lower())
        if any(token.text in [t.text for t in doc] for token in symptom_doc):
            matched_symptoms.append(symptom)

    app.logger.debug(f"Matched Symptoms: {matched_symptoms}")  # Log the matched symptoms
    return matched_symptoms

# @app.route('/')
# def index():
#     return render_template('index.html')
@app.route('/', methods=['GET', 'POST'])
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Debug: Print credentials for testing
        print(f"Username: {username}, Password: {password}")

        # Check if the user exists in session and validate the password
        if username in session.get('users', {}) and session['users'][username] == password:
            session['user'] = username  # Store the username in the session
            return redirect(url_for('index'))  # Redirect to the index page
        else:
            return "Invalid credentials, please try again.", 401

    return render_template('login.html')

# Route for the index page (after successful login)
@app.route('/index')
def index():
    if 'user' in session:  # Check if user is logged in
        return render_template('index.html', username=session['user'])
    return redirect(url_for('login'))  # Redirect to login if not logged in

# Route for the signup page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Ensure session has a place to store temporary users
        session['users'] = session.get('users', {})
        
        # Check if username already exists in the current session
        if username in session['users']:
            return "User already exists. Please login."

        # Save the username and password temporarily in the session
        session['users'][username] = password
        return redirect(url_for('login'))  # Redirect to login after successful signup

    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()  # Clear all session data
    return redirect(url_for('login'))


@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        data = request.json
        user_input = data.get('message', '').strip()
        session_id = data.get('session_id', 'default')

        state = user_state[session_id]
        user_name = state['name']

        if state['conversation_stage'] == 'get_name':
            if user_input.lower() == 'reset':
                user_state[session_id] = {'name': None, 'conversation_stage': 'get_name', 'condition': None, 'duration': None, 'symptoms': []}
                return jsonify({'response': "Let's start over. What should I call you?"})

            state['name'] = user_input
            state['conversation_stage'] = 'chat'
            return jsonify({'response': get_greeting(user_input)})

        if is_greeting(user_input):
            return jsonify({'response': get_greeting(user_name)})

        if user_input.lower() == 'ok':
            response = "Thank you for confirming. If you need further assistance, feel free to reach out. Have a great day ahead! 🌟\n"
            response += "Please Provide me your name."
            user_state[session_id] = {'name': None, 'conversation_stage': 'get_name', 'condition': None, 'duration': None, 'symptoms': []}
            return jsonify({'response': response})

        if state['conversation_stage'] == 'ask_duration':
            try:
                duration = int(user_input)
                state['duration'] = duration
                state['conversation_stage'] = 'chat'

                # If condition is found, ask about the appointment
                if state['condition']:
                    if duration >= 5:
                        response = f"{user_name}, since you've been experiencing these symptoms for {duration} days, which is 5 or more days, I recommend consulting a doctor. Would you like to book an appointment? (Yes/No)"
                        state['conversation_stage'] = 'ask_booking'  # Transition to booking stage
                    else:
                        response = f"I understand, {user_name}. Since it's been less than 5 days, please monitor your symptoms closely. If they persist or worsen, please consult a doctor. Is there anything else I can help you with?"
                        state['conversation_stage'] = 'end_conversation'#endconversation
                else:
                    response = f"I'm not sure about your condition based on the information provided, {user_name}. Could you tell me more about your symptoms?"

            except ValueError:
                response = f"I'm sorry, {user_name}, but I didn't understand that. Could you please enter the number of days you've been experiencing these symptoms?"

            return jsonify({'response': response})
        
        #end_conversation
        if state['conversation_stage'] == 'end_conversation' and user_input.lower() == 'no':
            response = f"Thank you for using our service, {user_name}. Take care and stay healthy! 😊"
    
        # Reset the session state to allow a new conversation
            user_state[session_id] = {'name': None, 'conversation_stage': 'get_name', 'condition': None, 'duration': None, 'symptoms': []}
            return jsonify({'response': response})

        
        if state['conversation_stage'] == 'ask_booking' and user_input.lower() == 'yes':
            # Handle the booking flow
            response = "Great! Let's proceed with booking your appointment. What time would you prefer for your appointment? (Please provide in HH:MM format)"
            state['conversation_stage'] = 'ask_time'

        elif state['conversation_stage'] == 'ask_booking' and user_input.lower() == 'no':
            response = "No worries, {user_name}. If you change your mind, feel free to ask for help anytime."

        else:
            # Symptom matching logic
            matched_symptoms = match_symptoms(user_input)
            state['symptoms'].extend(matched_symptoms)

            if matched_symptoms:
                all_conditions = [condition for symptom in state['symptoms'] for condition in symptoms_conditions_dict.get(symptom, [])]
                if all_conditions:
                    condition = max(set(all_conditions), key=all_conditions.count)
                    state['condition'] = condition
                    treatments = conditions_treatments_dict.get(condition, ["Consult a healthcare professional"])
                    response = f"{user_name}, based on your symptoms, you may have {condition}. Suggested precautions or treatments include: {', '.join(treatments)}."
                    response += f"\n\n{user_name}, how many days have you been experiencing these symptoms?"
                    state['conversation_stage'] = 'ask_duration'
                else:
                    response = f"I've noted your symptoms, {user_name}. Could you provide more details about how you're feeling?"
            else:
                response = f"I'm not sure about your condition based on the information provided, {user_name}. Could you tell me more about your symptoms?"

        return jsonify({'response': response})

    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        return jsonify({'response': "I apologize, but an error occurred. Please try again or contact support if the problem persists."})

@app.route('/book_appointment', methods=['POST'])
def handle_appointment():
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        name = user_state[session_id]['name']
        illness = user_state[session_id]['condition']
        preferred_time = data.get('preferred_time')

        if not all([name, illness, preferred_time]):
            return jsonify({'response': "I'm sorry, but I'm missing some information. Could you please provide all the necessary details for booking an appointment?"})

        # Use current date for the appointment
        current_date = datetime.datetime.now()
        appointment_date = current_date.strftime("%Y-%m-%d")

        # Check if the preferred time is in the future
        preferred_datetime = datetime.datetime.strptime(f"{appointment_date} {preferred_time}", "%Y-%m-%d %H:%M")
        if preferred_datetime <= current_date:
            return jsonify({'response': f"I'm sorry {name}, but the requested time has already passed. Please choose a future time for your appointment."})

        # Automatically book the appointment without checking for available slots
        assigned_doctor = doctors[0]['Name']  # Assign the first doctor for simplicity

        appointment_id = f"APPT-{random.randint(1000, 9999)}"
        appointment = {
            'ID': appointment_id,
            'Name': name,
            'Time': preferred_time,
            'Date': appointment_date,
            'Illness': illness,
            'Doctor': assigned_doctor,
            'Title': f"Appointment for {illness}",
            'Description': f"Consultation for {illness} symptoms"
        }

        # Save the appointment
        save_appointment(appointment)

        response = f"Great news, {name}! Your appointment has been booked successfully!\n"
        response += f"Appointment ID: {appointment_id}\n"
        response += f"Doctor: {assigned_doctor}\n"
        response += f"Date: {appointment_date}\n"
        response += f"Time: {preferred_time}\n"
        response += f"Please arrive at the GuniSter Health Hospital (opposite the shopping center) a few minutes before your appointment time."

        return jsonify({'response': response})

    except Exception as e:
        app.logger.error(f"An error occurred while booking the appointment: {str(e)}")
        return jsonify({'response': "I apologize, but an error occurred while booking your appointment. Please try again or contact our support team for assistance."})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
