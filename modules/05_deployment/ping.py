from flask import Flask


# Flask is a tool for building web services in Python
# Create the web app: 'ping' is just the app's name
app = Flask('ping')


# When someone visits the /ping endpoint, run the function below.
# GET means the client is only asking for data
@app.route('/ping', methods=['GET'])
def ping():
    return 'PONG'


if __name__ == '__main__':  # only when run directly, not when imported
    app.run(
        debug=True,      # auto-reload and show detailed errors
        host='0.0.0.0',  # accept connections from any machine
        port=9696        # port to listen on
    )
