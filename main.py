from flask import Flask, render_template, request, redirect, url_for, Response,jsonify,send_file
import mysql.connector
import os
import cv2
import re
import numpy as np
from keras.models import load_model
import joblib
import mysql.connector
from datetime import datetime
import tensorflow as keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import csv


app = Flask(__name__)
#database connection

def connect_to_database():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",  # Enter your database password here
        database="batch1"
    )
    #database connection ended
#Home page start

@app.route('/')
def index():
    return render_template('index.html')
#home page end 

#login page start

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        roll_number = request.form['roll_number']
        password = request.form['password']
        db_connection = connect_to_database()
        cursor = db_connection.cursor(dictionary=True)
        query = "SELECT * FROM registrations WHERE roll_number = %s AND password = %s"
        cursor.execute(query, (roll_number, password))
        user = cursor.fetchone()
        cursor.close()
        db_connection.close()
        if user:
            return redirect(url_for('user_details', roll_number=roll_number))
        else:
            return render_template('login.html', message="Invalid credentials. Please try again.")
    else:
        return render_template('login.html')
    #login page ended
    
    #user details start

@app.route('/user/<roll_number>')
def user_details(roll_number):
    db_connection = connect_to_database()
    cursor = db_connection.cursor(dictionary=True)
    # Fetch user details
    user_query = "SELECT * FROM registrations WHERE roll_number = %s"
    cursor.execute(user_query, (roll_number,))
    user = cursor.fetchone()
    
    # Fetch attendance details
    attendance_query = "SELECT date, total_days_attended FROM attendance WHERE roll_number = %s"
    cursor.execute(attendance_query, (roll_number,))
    attendance = cursor.fetchall()
    
    cursor.close()
    db_connection.close()
    
    if user:
        return render_template('user_details.html', user=user, attendance=attendance)
    else:
        return "User not found."
    
    #user details from ended
    
    
    #registration code
    
    
    
    # Function to validate password
def validate_password(password):
    # Check if password contains at least 8 characters, one lowercase, one uppercase, and one special character
    if len(password) < 8:
        return False
    if not re.search("[a-z]", password):
        return False
    if not re.search("[A-Z]", password):
        return False
    if not re.search("[!@#$%^&*()_+{}:<>?]", password):
        return False
    return True
'''
# Function to establish database connection
def connect_to_database():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",  # Enter your database password here
        database="batch1"
    )
    '''

# Function to perform face detection and save images
def take_images(roll_number):
    # Create a folder based on roll number if not exists
    folder_path = f"images/{roll_number}"
    os.makedirs(folder_path, exist_ok=True)
    
    # Initialize the face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    count = 0
    while count < 300:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Draw rectangle around detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # Save cropped face image
            cv2.imwrite(f"{folder_path}/{count}.jpg", frame[y:y+h, x:x+w])
            count += 1
        
        cv2.imshow('Face Detection', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    return folder_path

@app.route('/registration', methods=['GET','POST'])

def registration():
    if request.method == 'POST':
        full_name = request.form['full_name']
        roll_number = request.form['roll_number']
        phone_number = request.form['phone_number']
        email = request.form['email']
        gender = request.form['gender']
        department = request.form['department']
        year_of_study = request.form['year_of_study']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        # Check if passwords match
        if password != confirm_password:
            return "Passwords do not match"
        
        # Validate password
        if not validate_password(password):
            return "Password must contain at least 8 characters including one lowercase, one uppercase, and one special character."
        
        # Check if roll number or email already exists in the database
        conn = connect_to_database()
        cursor = conn.cursor()
        
        # Query to check if roll number exists
        sql_roll = "SELECT * FROM registrations WHERE roll_number = %s"
        val_roll = (roll_number,)
        cursor.execute(sql_roll, val_roll)
        roll_result = cursor.fetchone()
        
        # Consume all results from the cursor
        cursor.fetchall()
        
        # Query to check if email exists
        sql_email = "SELECT * FROM registrations WHERE email = %s"
        val_email = (email,)
        cursor.execute(sql_email, val_email)
        email_result = cursor.fetchone()
        
        # Consume all results from the cursor
        cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        if roll_result:
            return "Roll number already exists. Please try a different roll number."
        
        if email_result:
            return "Email already exists. Please try a different email."
        
        # Proceed with registration if roll number and email are unique
        folder_path = take_images(roll_number)
        
        # Connect to database
        conn = connect_to_database()
        cursor = conn.cursor()
        
        # Insert registration data into database
        sql_insert = "INSERT INTO registrations (full_name, roll_number, phone_number, email, gender, department, year_of_study, password, image_path) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
        val_insert = (full_name, roll_number, phone_number, email, gender, department, year_of_study, password, folder_path)
        cursor.execute(sql_insert, val_insert)
        conn.commit()
        
        # database connection close
        cursor.close()
        conn.close()
        print(f"Registration successful.")
        
        return f"Registration successful.............. :) " and  redirect(url_for('login'))
    
    return render_template('registration.html')

#registrationform ended

#techers side page start

teacher_credentials = {
    'HOD_SIR': 'ECE',
    'teacher2': 'password2'
}
@app.route('/teachers', methods=['GET', 'POST'])
def teacher_login():
    if request.method == 'POST':
        teacher_id = request.form['teacher_id']
        password = request.form['password']
        
        # Check if teacher ID exists and password matches
        if teacher_id in teacher_credentials and teacher_credentials[teacher_id] == password:
            # Redirect to teacher's dashboard
            return redirect(url_for('teachers_dash'))
        else:
            return render_template('teachers.html', message="Invalid credentials. Please try again.")
    else:
        return render_template('teachers.html')
@app.route('/teachers_dash')
def teachers_dash():
    return render_template('teachers_dash.html')


#take attendence using cam start

model = load_model('face_detection_model.h5')

# Load the label encoder
label_encoder = joblib.load('label_encoder.pkl')

# Load the pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Connect to MySQL database
db_connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",  # Enter your database password here
    database="batch1"
)
db_cursor = db_connection.cursor()

# Function to preprocess images
def preprocess_image(image):
    # Resize image to the size expected by the model
    resized_image = cv2.resize(image, (200, 200))
    # Normalize pixel values to be in the range [0, 1]
    normalized_image = resized_image / 255.0
    return normalized_image

# Function to detect faces in the frame
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    return faces

# Create a new table to store attendance records if it doesn't exist
def create_attendance_table():
    try:
        db_cursor.execute("CREATE TABLE IF NOT EXISTS attendance (id INT AUTO_INCREMENT PRIMARY KEY, roll_number VARCHAR(255), date DATE, time TIME, total_days_attended INT DEFAULT 0, UNIQUE(roll_number, date))")
        db_connection.commit()
    except mysql.connector.Error as err:
        print("Error:", err)

# Function to check if attendance has already been taken for today
def attendance_taken_today(roll_number):
    try:
        today = datetime.now().date()
        db_cursor.execute("SELECT COUNT(*) FROM attendance WHERE roll_number = %s AND date = %s", (roll_number, today))
        result = db_cursor.fetchone()
        return result[0] > 0
    except mysql.connector.Error as err:
        print("Error:", err)
        return False

# Function for live recognition
def live_recognition(camera):
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        # Detect faces in the frame
        faces = detect_faces(frame)

        # Loop over detected faces
        for (x, y, w, h) in faces:
            # Draw rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # Extract the face region from the frame
            face_region = frame[y:y+h, x:x+w]
            # Preprocess the face region
            processed_face = preprocess_image(face_region)
            # Make predictions using the model
            prediction = model.predict(np.expand_dims(processed_face, axis=0))
            # Convert prediction to label
            label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
            # Display the label on the frame
            cv2.putText(frame, str(label), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            
            # Check if attendance has already been taken for today
            if not attendance_taken_today(str(label)):
                # Mark attendance in the database
                try:
                    now = datetime.now()
                    sql = "INSERT INTO attendance (roll_number, date, time) VALUES (%s, %s, %s)"
                    val = (str(label), now.date(), now.time())
                    db_cursor.execute(sql, val)
                    db_connection.commit()
                    
                    # Increment total days attended
                    db_cursor.execute("UPDATE attendance SET total_days_attended = total_days_attended + 1 WHERE roll_number = %s", (str(label),))
                    db_connection.commit()
                    
                except mysql.connector.Error as err:
                    print("Error:", err)

        # Encode the frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Release the camera
    camera.release()

@app.route('/take_attendance')
def take_attendance():
    return render_template('take_attendance.html')

@app.route('/start_attendance')
def start_attendance():
    # Initialize the webcam
    camera = cv2.VideoCapture(0)
    create_attendance_table()
    return Response(live_recognition(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_attendance')
def stop_attendance():
    # Release the camera
    cv2.destroyAllWindows()
    return redirect(url_for('teachers_dash'))

#attendance will ended

#trained data method started

def load_images_from_database():
    conn = connect_to_database()
    cursor = conn.cursor()

    cursor.execute("SELECT image_path, roll_number FROM registrations")
    rows = cursor.fetchall()

    images = []
    labels = []

    for row in rows:
        image_folder, label = row
        # Normalize image_folder path to handle slashes
        image_folder_path = os.path.normpath(image_folder)
        for i in range(300):
            # Use os.path.join for generating image_path
            image_path = os.path.join(image_folder_path, f"{i}.jpg")
            if os.path.exists(image_path):  # Check if the image file exists
                image = cv2.imread(image_path)
                if image is not None:  # Check if the image was successfully read
                    image = cv2.resize(image, (200, 200))  # Resize images to a consistent size
                    images.append(image)
                    labels.append(label)
            else:
                print(f"Warning: Image not found at {image_path}")

    cursor.close()
    conn.close()

    # Convert images and labels to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    return images, labels



# Route to train the model
@app.route('/train_data')
def train_data():
    def generate():
        # Yield the initial HTML content
        yield "<h2>Training in progress...</h2>"
        yield "<p>Epochs:</p>"

        # Load images and labels from the database
        images, labels = load_images_from_database()

        # Check if the dataset has enough samples for splitting
        if len(images) == 0:
            yield "Error: Dataset is too small to train the model."

        # Preprocess the data
        images = images / 255.0  # Normalize pixel values to the range [0, 1]

        # Encode labels
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)

        # Split the dataset into training and testing sets
        if len(images) >= 5:
            train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
        else:
            train_images, test_images, train_labels, test_labels = images, images, labels, labels

        # Define the CNN model
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),  # Adjust input shape
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(len(np.unique(labels)), activation='softmax')  # Output layer with softmax activation for multiclass classification
        ])

        # Compile the model
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # Train the model
        history = model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels))

        # Save the trained model and label encoder
        model.save('face_detection_model.h5')

        # Save label encoder
        joblib.dump(label_encoder, 'label_encoder.pkl')

        # Yield training progress
        for epoch, (loss, accuracy) in enumerate(zip(history.history['loss'], history.history['accuracy']), 1):
            # Generate HTML content for each epoch
            yield f"<p>Epoch {epoch}: Loss {loss}, Accuracy {accuracy}</p>"
            # Flush the response to send the content immediately
            yield "\n"

        # Yield training status
        yield "<p>Training successfully completed.</p>"

    # Return the response with the generator
    return Response(generate(), content_type='text/html')

#download details

@app.route('/download_details')
def download_details():
    # Connect to the database
    conn = connect_to_database()
    cursor = conn.cursor()

    try:
        # Fetching data from the registrations table
        cursor.execute("SELECT full_name, roll_number, phone_number, email, gender, department, year_of_study FROM registrations")
        registrations_data = cursor.fetchall()

        # Fetching data from the attendance table
        cursor.execute("SELECT roll_number, MAX(total_days_attended) as total_days_attended FROM attendance GROUP BY roll_number")
        attendance_data = cursor.fetchall()

        # Combine data from both tables
        combined_data = []
        for registration_row in registrations_data:
            roll_number = registration_row[1]
            for attendance_row in attendance_data:
                if attendance_row[0] == roll_number:
                    combined_data.append((*registration_row, attendance_row[1]))
                    break

        # Prepare CSV data
        csv_data = []
        csv_data.append(['Full Name', 'Roll Number', 'Phone Number', 'Email', 'Gender', 'Department', 'Year of Study', 'Total Days Attended'])
        for row in combined_data:
            csv_data.append(row)

        # Generate CSV file
        with open('student_details.csv', 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerows(csv_data)

        # Close database connection
        cursor.close()
        conn.close()

        # Send the CSV file as a downloadable attachment
        return send_file('student_details.csv', as_attachment=True)

    except Exception as e:
        return str(e)
    
    #download details ended
    
    #finding student started
@app.route('/find_student', methods=['GET', 'POST'])
def find_student():
    if request.method == 'POST':
        roll_number = request.form['roll_number']
        connection = connect_to_database()
        cursor = connection.cursor(dictionary=True)

        # Fetching student details from registrations table
        cursor.execute("SELECT * FROM registrations WHERE roll_number = %s", (roll_number,))
        student_details = cursor.fetchone()

        # Fetching attendance details from attendance table
        cursor.execute("SELECT * FROM attendance WHERE roll_number = %s ORDER BY total_days_attended DESC LIMIT 1", (roll_number,))
        attendance_details = cursor.fetchone()

        connection.close()

        return render_template('student_details.html', student=student_details, attendance=attendance_details)
    else:
        return render_template('find_student.html')

if __name__ == '__main__':
    app.run(debug=True)
    
    
if __name__ == '__main__':
    app.run(debug=True)
