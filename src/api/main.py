from flask import Flask, request, jsonify

app = Flask(__name__) 

@app.route("/")
def home():
    return jsonify(
        {"message" : "CLIP model API is running! upload a image in '/predict' page to predictions"} 
    ), 200

@app.route("/predict", methods=['POST'])
def predict():
    if "image" not in request.files:
        return jsonify({"Error": "No Image uploaded"}), 400
    try:
        pass
    except Exception as e:
        print(f"Internal Server Error: {e}")
        return jsonify({"Error": str(e)}), 500

if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 5000) 