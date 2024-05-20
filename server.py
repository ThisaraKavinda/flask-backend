from flask import Flask, send_from_directory, request
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS

from functions.process_frames_copy import process_video
from functions.calibrate_lips import calibrate

app = Flask(__name__, static_folder='../client/build', static_url_path='/')
CORS(app)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@socketio.on('connect')
def handle_connect():
    room = request.args.get('room')
    join_room(room)
    print(f'Client connected to room: {room}')
    
@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')

@socketio.on('join_room')
def on_join(data):
    username = data['username']
    room = data['room']
    join_room(room)
    print(f'{username} has entered the room: {room}')

@socketio.on('leave_room')
def on_leave(data):
    username = data['username']
    room = data['room']
    leave_room(room)
    print(f'{username} has left the room: {room}')
    

@socketio.on('frame')
def handle_frame(data):
    print("Frame received")
    frame = data['frame']
    room = data['room']
    result = process_video(frame)
    if result:
        emit('frame', {'status': 'frame received', 'result': result}, room=room)
        
@socketio.on('calibrate_lips')
def calibrate_lips(data):
    print("Frame received - calibrate lips")
    frame = data['frame']
    room = data['room']
    result = calibrate(frame)
    if result:
        if result == "SUCCESS":
            emit('calibrate_lips', {'status': 'SUCCESS', 'result': "Calibrating finished"}, room=room)
        elif result == "COME_CLOSER":
            emit('calibrate_lips', {'status': 'COME_CLOSER', 'result': "Please come closer to the camera"}, room=room)
        elif result == "MOVE_AWAY":
            emit('calibrate_lips', {'status': 'MOVE_AWAY', 'result': "Please move away from the camera"}, room=room)
        else: 
            emit('calibrate_lips', {'status': 'CALIBRATING_LIPS', 'result': "Calibrating lips in progress"}, room=room)

if __name__ == '__main__':
    socketio.run(app, port=5006, debug=True)
