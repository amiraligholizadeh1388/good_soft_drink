from flask import Flask, Response, render_template, redirect, url_for, request, flash, session
from flask_sqlalchemy import SQLAlchemy
import cv2
import os
import numpy as np
from datetime import timedelta
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default_secret_key')
app.config['SESSION_PERMANENT'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    phone_number = db.Column(db.String(15), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)

@app.route('/')
def signing():
    return render_template('signing.html')

@app.route('/start_capture', methods=['POST'])
def start_capture():
    first_name = request.form['first_name']
    last_name = request.form['last_name']
    phone_number = request.form['phone_number']
    email = request.form['email']

    # Save user info to the database
    new_user = User(first_name=first_name, last_name=last_name, phone_number=phone_number, email=email)
    db.session.add(new_user)
    db.session.commit()
    session['user_id'] = new_user.id
    return redirect(url_for('video_feed', user_id=new_user.id))

@app.route('/video_feed/<int:user_id>')
def video_feed(user_id):
    user = User.query.get_or_404(user_id)
    return render_template('video_feed.html', user_id=user.id)

@app.route('/video_feed_source')
def video_feed_source():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def prepare_training_data(dataset_path):
    faces = []
    labels = []
    label_dict = {}

    for image_name in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, image_name)
        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Extract the person ID from the filename
        person_id = int(image_name.split('_')[0])
        label_dict[person_id] = person_id

        faces.append(gray_image)
        labels.append(person_id)

    return faces, np.array(labels), label_dict


def train():
    dataset_path = 'face_dataset'

    # Initialize the face recognizer

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces, labels, label_dict = prepare_training_data(dataset_path)
    recognizer.train(faces, labels)

    # Save the trained model
    recognizer.save('trained_face_model.yml')

def generate_frames():
    cap = cv2.VideoCapture(0)
    haar_cascade_path = 'cascades/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(haar_cascade_path)
    person_id = 1
    count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face = gray[y:y + h, x:x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_filename = os.path.join('face_dataset', f"{person_id}_{count}.jpg")
            cv2.imwrite(face_filename, face)
        # Encode the frame in JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        train()
        if count == 30:
            yield (b'--f\r\n'
                            b'Content-Type: text/plain\r\n\r\n'
                            b'REDIRECT\r\n\r\n')
            break
    cap.release()  # Release the camera at the end
    

@app.route('/account')
def account():
    user_id = session.get('user_id')  # Get user_id from session
    if user_id is None:
        return redirect(url_for('signing'))  # Redirect to signing if no user_id
    user = User.query.get(user_id)
    return render_template('account.html', first_name=user.first_name)

@app.route('/face_login')
def face_login():
    return render_template('face_login.html')

@app.route('/perform_face_login')
def perform_face_login():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x + x + w]

            # Predict the person
            label, confidence = recognizer.predict(face)

            # If confidence is good enough (you can define a threshold)
            if confidence < 50:  # lower value means better match
                # Log in the user by retrieving their user ID
                user = User.query.get(label)
                if user:
                    session['user_id'] = user.id
                    flash(f"Welcome back, {user.first_name}!", "success")
                    return redirect(url_for('account'))
            else:
                flash("Face not recognized. Try again.", "danger")

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        train()
    cap.release()
    cv2.destroyAllWindows()
    return "Face recognition login complete."


if __name__ == '__main__':
    # Create database tables within an application context
    with app.app_context():
        db.create_all()  # Create database tables
    app.run(debug=True)
