from flask import Flask, jsonify
import asyncio
import websockets

app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({"message": "WebSocket server"})

async def websocket_handler(websocket, path):
    async for message in websocket:
        print("Received message:", message)
        print("Path:", path)
        
    async for connect in websocket:
        print("Connected")
        
    async for disconnect in websocket:
        print("Disconnected")

async def main():
    # Define the WebSocket server
    ws_server = websockets.serve(websocket_handler, "localhost", 6789)
    await ws_server  # Start the WebSocket server

    # Run Flask app with Hypercorn in a subprocess
    from hypercorn.asyncio import serve
    from hypercorn.config import Config
    config = Config()
    config.bind = ["localhost:5005"]  # Flask app will run on port 5000
    await serve(app, config)

if __name__ == '__main__':
    asyncio.run(main())


