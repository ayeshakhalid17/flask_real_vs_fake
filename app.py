
from flask import Flask, request, jsonify, send_file, render_template
import os
import json
from fastai.vision import load_learner, Path
from fastai.vision.image import open_image
app = Flask(__name__)
@app.route('/')
def home():   
    ''' render home.html - page that is served at localhost that allows user to enter model scoring parameters'''
    title_text = "fastai deployment"
    title = {'titlename':title_text}
    return render_template('home.html',title=title) 
    
@app.route('/show_prediction',methods=['POST'])
def show_prediction():
    if request.method == 'POST':
        f = request.files['image_file']
        path=Path('')
        learn=load_learner(path)
        # img = Image.open(f)
        # img = ImageOps.contain(img,(400,400),Image.LANCZOS)
        img = open_image(f)
    prediction = learn.predict(img)
    #print(prediction[0])
    # return 
        
    # img = PILImage.create(full_path)
    # # apply the model to the image
    # pred_class, ti1, ti2 = learner.predict(img)
    # print("pred_class is: ",pred_class)
    # predict_string = "Predicted object is: "+pred_class
    # # build parameter to pass on to show-prediction.html
    prediction = {'prediction_key':str(prediction[0])}
    # # render the page that will show the prediction
    return(render_template('prediction.html',prediction=prediction))
    # return "hello"

if __name__ == '_main_':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True)