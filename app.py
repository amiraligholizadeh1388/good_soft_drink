from flask import Flask, Response, render_template, redirect, url_for, request, flash, session
from flask_sqlalchemy import SQLAlchemy
import cv2
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default_secret_key')
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

    return redirect(url_for('video_feed', user_id=new_user.id))

@app.route('/video_feed/<int:user_id>')
def video_feed(user_id):
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    cap = cv2.VideoCapture(0)
    haar_cascade_path = 'cascades/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(haar_cascade_path)
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Encode the frame in JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()  # Release the camera at the end

@app.route('/account')
def account():
    user_id = session.get('user_id')  # Get user_id from session
    if user_id is None:
        return redirect(url_for('signing'))  # Redirect to signing if no user_id
    user = User.query.get(user_id)
    return render_template('account.html', first_name=user.first_name)

if __name__ == '__main__':
    # Create database tables within an application context
    with app.app_context():
        db.create_all()  # Create database tables
    app.run(debug=True)
