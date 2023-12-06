from flask import Flask, request, jsonify
from app.torch_utils import transform_image, get_prediction

app = Flask(__name__) # Flask app instance

def allowed_file(filename):
    # xxx.png
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ['png', 'jpg', 'jpeg']

@app.route('/predict', methods=['POST'])  # Your API endpoint URL would consist /predict
def predict():
    if request.method == 'POST':
        # we will get the file from the request
        file = request.files['file']
        if file is None or file.filename == "":
            return jsonify({'result': 0, 'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'result': 0, 'error': 'format not supported'})
        
        try:
            img_bytes = file.read()
            tensor = transform_image(img_bytes)
            prediction = get_prediction(tensor)
            data = {'prediction': prediction.item(), 'class_name': str(prediction.item())}  
            return jsonify(data)
        except:
            return jsonify({'result': 0, 'error': 'error during prediction'})
