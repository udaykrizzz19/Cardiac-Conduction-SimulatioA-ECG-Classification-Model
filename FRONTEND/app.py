from flask import Flask, url_for, redirect, render_template, request, session
import mysql.connector, os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib as plt
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D



app = Flask(__name__)
app.secret_key = 'admin'

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Uday_Krishna@1",
    port="3306",
    database='ecgsignal'
)

mycursor = mydb.cursor()

def executionquery(query,values):
    mycursor.execute(query,values)
    mydb.commit()
    return

def retrivequery1(query,values):
    mycursor.execute(query,values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['c_password']
        if password == c_password:
            query = "SELECT UPPER(email) FROM users"
            email_data = retrivequery2(query)
            email_data_list = []
            for i in email_data:
                email_data_list.append(i[0])
            if email.upper() not in email_data_list:
                query = "INSERT INTO users (email, password) VALUES (%s, %s)"
                values = (email, password)
                executionquery(query, values)
                return render_template('login.html', message="Successfully Registered!")
            return render_template('register.html', message="This email ID is already exists!")
        return render_template('register.html', message="Conform password is not match!")
    return render_template('register.html')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        
        query = "SELECT UPPER(email) FROM users"
        email_data = retrivequery2(query)
        email_data_list = []
        for i in email_data:
            email_data_list.append(i[0])

        if email.upper() in email_data_list:
            query = "SELECT UPPER(password) FROM users WHERE email = %s"
            values = (email,)
            password__data = retrivequery1(query, values)
            if password.upper() == password__data[0][0]:
                global user_email
                user_email = email

                return render_template('home.html')
            return render_template('login.html', message= "Invalid Password!!")
        return render_template('login.html', message= "This email ID does not exist!")
    return render_template('login.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

# Load the saved classification model

model = tf.keras.models.load_model('mobilenet_v2_classifier_model.h5')

# Reload the MobileNet feature extractor
base_model = MobileNet(weights='imagenet', include_top=False)
feature_extractor = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))

class_names = [
    'ECG Images of Myocardial Infarction Patients',  # Class 0
    'ECG Images of Patient that have History of MI',  # Class 1
    'ECG Images of Patient that have abnormal heartbeat',  # Class 2
    'Normal Person ECG Images'  # Class 3
]

# Relevant information or suggestions for each class
class_info = {
    'ECG Images of Myocardial Infarction Patients': "This ECG suggests a recent or past heart attack. Key features may include elevated ST segments, T-wave inversion, or Q-wave changes. Further examination by a cardiologist is advised for treatment planning.",
    'ECG Images of Patient that have History of MI': "This ECG shows signs of a previous heart attack. Look for abnormal Q waves or left ventricular hypertrophy. Close monitoring and lifestyle adjustments are recommended.",
    'ECG Images of Patient that have abnormal heartbeat': "This ECG indicates arrhythmia, possibly including atrial fibrillation, PVCs, or tachycardia. It is essential to monitor the heart's rhythm and consult a specialist for potential medication or procedures.",
    'Normal Person ECG Images': "This ECG is from a healthy individual with no signs of heart disease. Regular heart function is observed, but maintaining a healthy lifestyle and routine checkups are always recommended."
}

def make_prediction(model, image_path):
    """
    Preprocess the image, make a prediction using the trained model, and provide relevant information for the predicted class.
    """
    # Load the image and preprocess it
    img = load_img(image_path, target_size=(224, 224))  # Resize to MobileNet input size
    img_array = img_to_array(img)  # Convert to a NumPy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess for MobileNet

    # Extract features using the MobileNet feature extractor
    features = feature_extractor.predict(img_array)

    # Predict the class using the trained classification model
    predictions = model.predict(features)
    predicted_class_idx = np.argmax(predictions)  # Get index of the highest probability
    predicted_class = class_names[predicted_class_idx]  # Map index to class name
    confidence_score = predictions[0][predicted_class_idx]  # Get confidence score

    # Retrieve the relevant information based on the predicted class
    suggestion = class_info.get(predicted_class, "No relevant information available.")
    
    return predicted_class, confidence_score, suggestion

global suggestion

@app.route('/upload', methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        # Get the uploaded file
        myfile = request.files['file']
        fn = myfile.filename  # Extract filename
        mypath = os.path.join('static', 'img', fn)  # Save path
        myfile.save(mypath)  # Save the file to the server

        # Make prediction
        predicted_class, confidence_score, suggestion = make_prediction(model, mypath)

        # Only show confidence score if NOT "Normal Person ECG Images"
        if predicted_class != "Normal Person ECG Images":
            confidence_text = f"{confidence_score * 100:.2f}%"
        else:
            confidence_text = "0%"  # Don't show any confidence score

        # Return result to the template with suggestion
        return render_template('upload.html', 
                               path=mypath, 
                               prediction=predicted_class, 
                               confidence=confidence_text, 
                               suggestion=suggestion)

    return render_template('upload.html')



if __name__ == '__main__':
    app.run(debug = True)