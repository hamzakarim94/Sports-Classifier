from flask import Flask, request, jsonify
import util
app = Flask(__name__)

@app.route('/classify_image', methods=['GET', 'POST'])
def classify_image():
    className = ["Christiano Ronaldo", "Lionel Messi", "Paulo Dybala", "Sergio Aguero", "Sergio Romero"]
    image_data = request.form['image_data']

    response = jsonify(util.classify_image(image_data,className))

    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

if __name__ == "__main__":
    print("Starting Python Flask Server For Sports Celebrity Image Classification")
    className = ["Christiano Ronaldo", "Lionel Messi", "Paulo Dybala", "Sergio Aguero", "Sergio Romero"]
    util.load_model(className)
    app.run(host='0.0.0.0',port=5000)