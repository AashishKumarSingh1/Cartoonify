from flask import Flask,request,jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
import os
app=Flask(__name__)
CORS(app)

def cartoonify_img(image_data):
    img_data=base64.b64decode(image_data)
    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

    img=cv2.resize(img,(800,600))
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray=cv2.medianBlur(gray,5)

    edges = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 9, 9)

    color = cv2.bilateralFilter(img, 9, 75, 75)

    cartoon = cv2.bitwise_and(color, color, mask=edges)

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    cartoon = cv2.filter2D(cartoon, -1, kernel)

    cartoon = cv2.convertScaleAbs(cartoon, alpha=1, beta=50)

    _, buffer = cv2.imencode('.jpg', cartoon)
    cartoon_base64 = base64.b64encode(buffer).decode('utf-8')

    return cartoon_base64

@app.route('/cartoonify', methods=['POST'])
def cartoonify_route():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400
    
    base64_image = data['image']
    cartoon_image = cartoonify_img(base64_image)

    return jsonify({'cartoon_image': cartoon_image})


if __name__=='__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))