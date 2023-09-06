from flask import Flask
from flask_socketio import SocketIO

socketio = SocketIO()

def create_app():
    app = Flask(__name__)
    socketio.init_app(app)

    from . import routes
    app.register_blueprint(routes.bp)

    return app