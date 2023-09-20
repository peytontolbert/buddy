from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# In-memory storage for the document, to be replaced by database
document_content = "This is a sample document."

@app.route("/", methods=["GET", "POST"])
def index():
    global document_content
    if request.method == "POST":
        new_content = request.json.get("content")
        # Embed code here
        # new_content = embed_code(new_content)
        document_content = new_content
        return jsonify({"status": "success"})
    return render_template("index.html", content=document_content)

if __name__ == "__main__":
    app.run(debug=True)