from flask import Flask, request, jsonify
from multiprocessing import Process, Queue

def start_http_server(finnagi):
    app = Flask(__name__)
    queue = Queue()

    # Function to handle user input
    def user_input_process(queue):
        while True:
            user_input = input("Enter your input: ")
            queue.put(user_input)

    # Route to receive user input
    @app.route("/userinput", methods=["POST"])
    def receive_user_input():
        user_input = request.json["input"]
        queue.put(user_input)
        return "OK"

    # Route to get agent response
    @app.route("/agentresponse", methods=["GET"])
    def get_agent_response():
        if not queue.empty():
            user_input = queue.get()
            response = finnagi.process_input(user_input) # Replace "finnagi" with your agent object
            return jsonify({"response": response})
        else:
            return jsonify({"response": ""})

    if __name__ == "__main__":
        # Start the user input process
        user_input_proc = Process(target=user_input_process, args=(queue,))
        user_input_proc.start()

        # Start the HTTP server
        app.run()