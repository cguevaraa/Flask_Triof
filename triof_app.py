from flask import Flask, render_template, request
from src.utils import *
from base64 import b64encode
from base64 import b64decode
import cv2

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/start')
def insert():
    open_waste_slot()
    
    return render_template('insert.html')

@app.route('/waste/is_clean', methods=['POST'])
def is_clean():
    close_waste_slot()
    # Load the CNN
    model = load_model(vgg16=True)

    # Get the image from the POST request and convert into np.array (CV image)
    img, img_str = read_img_from_form(request)

    # Classify clean or dirty
    result = classify_dirty(img, model)

    return render_template('isclean.html', result=result, img=img_str)

@app.route('/waste/pick-type', methods=['POST'])
def pick_type():
    img_str = request.form.get('im')
    img_bytes = b64decode(img_str)
    result = classify(img_bytes)

    return render_template('type.html', result=result, img=img_str)

@app.route('/confirmation', methods=['POST'])
def confirmation():
    waste_type = request.form['type']

    process_waste(waste_type)
    return render_template('confirmation.html')

if __name__ == "__main__":
    app.run(debug=True)
