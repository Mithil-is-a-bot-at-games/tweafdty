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
                result, confidence = predict_image(image)
            except Exception as e:
                return f"Error during prediction: {str(e)}"

    return render_template("index.html", result=result, confidence=confidence)
