from flask import Flask
app = Flask(__name__)

@app.route('/')
def index():
    return "Test message. The server is running"

@app.route('/hello')
def hello_func():
    return 'hello!'


if __name__ == '__main__':
    app.run('localhost', 5000)