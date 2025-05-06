from flask import Flask, render_template, request
from teach import predict_image  # Import the predict_image function from teach.py

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    if request.method == "POST":
        if 'image' not in request.files:
            return "No file part"
        
        image = request.files["image"]

        if image.filename == "":
            return "No selected file"

        if image:
            try:
                # Pass the uploaded image to the predict_image function
                result, confidence = predict_image(image)
            except Exception as e:
                return f"Error during prediction: {str(e)}"

    # Ensure that result and confidence are passed to the template
    return render_template("index.html", result=result, confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)
