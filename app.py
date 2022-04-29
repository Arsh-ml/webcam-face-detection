from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)
camera = cv2.VideoCapture(0)


# Defining cascades 
faceCascade = cv2.CascadeClassifier("haar\haarcascade_frontalface_default.xml")


def generate_frames():
    while True:
        success, frame = camera.read()

        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Converting frame to Black and white for faster processing

        faces = faceCascade.detectMultiScale(frameGray, 1.3, 5)

        # Making box around the face
        for face in faces:
            x, y, w, h = face
            top_left = (x, y)
            bottom_right = ((x + w), (y + h))
            cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 3)


        # End if no camera found
        if not success:
            break

        else:
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
        
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video")
def video():
    return Response(generate_frames(), mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug = True)