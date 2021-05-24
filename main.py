import sklearn.externals
import joblib
from pip._vendor import certifi
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from skimage import feature
from imutils import build_montages
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

from flask import Flask, render_template, request, url_for, session, redirect, flash
from flask_pymongo import PyMongo
import bcrypt
from pymongo import MongoClient
app = Flask(__name__)
client = MongoClient("mongodb+srv://newuser:test123@cluster0.mgjck.mongodb.net/parkinsonsUsers?retryWrites=true&w=majority",tlsCAFile=certifi.where())
db = client.get_database('parkinsonsUsers')
users = db.usersdata



@app.route('/',methods=['GET'])
def landing():
    return render_template("homepage.html")
@app.route('/dashboard/spiral', methods=['GET'])
def home1():
    btnname = "spiral"
    return render_template("view.html", btnname=btnname)



@app.route('/Documentation',methods=['GET'])
def doc():
    return render_template('doc.html')
@app.route('/dashboard/wave', methods=['GET'])
def home2():
    btnname = "wave"
    return render_template("view.html", btnname=btnname)



@app.route('/dashboard', methods=['GET'])
def dashboard():
    imgname = session['username']

    if('filenameSpiral' in session.keys()):
        imgname = imgname.replace(' ', '-').lower()
        finalname1 = "static/images/spiralTest/" + imgname + ".png"
    else:
        finalname1 = "/"

    if('filenameWave' in session.keys()):
        imgname = imgname.replace(' ', '-').lower()
        finalname2 = "static/images/waveTest/" + imgname + ".png"
    else:
        finalname2 = "/"
    imglist = [finalname1, finalname2]
    wavetest = None
    spiraltest = None

    if('a' in session.keys()):
        wavetest = session['a']

    if('s' in session.keys()):
        spiraltest = session['s']

    if(wavetest is None and spiraltest is None):
        report = "Not tested yet"
    if(wavetest == "healthy" and spiraltest == "healthy"):
        report = "Dear "+session['username']+" you are completely healthy"
    elif(wavetest == "parkinson" and spiraltest == "parkinson"):
        report = "Dear "+session['username']+" you are detected positive for paralysis we advice you to concern with doctor as soon as possible."

    elif (wavetest == "parkinson" and spiraltest == "healthy"):
        report = "Dear "+session['username']+" we detected that you may have chances to develope paralysis in future if problem continues please visit hospital."

    elif (wavetest == "healthy" and spiraltest == "parkinson"):
        report = "Dear "+session['username']+" we detected that you may have chances to develope paralysis in future if problem continues please visit hospital."

    else:
        report = "Not tested yet"

    imglist.append(report)
    return render_template("dash.html", imglist=imglist)

@app.route('/login',methods=['POST'])
def login():
    loginUser = users.find_one({"name": request.form['username']})

    if loginUser:
        if bcrypt.hashpw(request.form['pass'].encode('utf-8'), loginUser["password"]) == loginUser["password"]:
            session['username'] = request.form['username']

            return redirect(url_for('dashboard'))
        return render_template("register.html")
    return render_template("register.html")



@app.route('/logout')
def logout():
    session.pop('username',None)
    session.pop('a',None)
    session.pop('s',None)
    session.pop('filenameSpiral',None)
    session.pop('filenameWave',None)
    return redirect(url_for('landing'))


@app.route('/register',methods=['POST', "GET"])
def register():
    if request.method == 'POST':

        existing_user = users.find_one({'name':request.form['username']})

        if existing_user is None:
            hashpass = bcrypt.hashpw(request.form['pass'].encode('utf-8'), bcrypt.gensalt())
            users.insert({'name': request.form['username'], 'password': hashpass})
            session['username'] = request.form['username']
            return redirect(url_for('dashboard'))
        else:
            flash("user already exists")

    return render_template('register.html')



## Machine Learning Algorithm
def quantify_image(image):
    features = feature.hog(image, orientations=9,
                           pixels_per_cell=(10, 10), cells_per_block=(2, 2),
                           transform_sqrt=True, block_norm="L1")
    return features


def load_split(path):
    imagePaths = list(paths.list_images(path))
    data = []
    labels = []

    for imagePath in imagePaths:
        label = imagePath.split(os.path.sep)[-2]
        # [healthy, healthy, parkinson, ....]
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200, 200))

        image = cv2.threshold(image, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        features = quantify_image(image)
        data.append(features)
        labels.append(label)


    return (np.array(data), np.array(labels))



trainingPathSpiral = "parkinson-dataset/spiral/training"
# loading the training and testing data
print("[INFO] loading data...")
(trainX, trainY) = load_split(trainingPathSpiral)
# encode the labels as integers
le = LabelEncoder()
trainY = le.fit_transform(trainY)


trials = {}

for i in range(0, 5):
    print("[INFO] training model {} of {}...".format(i + 1,5))
    model = RandomForestClassifier(n_estimators=100)
    model.fit(trainX, trainY)




def classify_my_image_spiral(image_path):
    image = cv2.imread(image_path)
    output = image.copy()
    output = cv2.resize(output, (128, 128))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (200, 200))
    image = cv2.threshold(image, 0, 255,
                          cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    features = quantify_image(image)
    preds = model.predict([features])


    label = le.inverse_transform(preds)[0]
    session['s'] = label
    color = (0, 255, 0) if label == "healthy" else (0, 0, 255)
    cv2.putText(output, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # show the output montage
    imgname = session['username']
    imgname = imgname.replace(' ', '-').lower()
    cv2.imwrite("static/images/spiralTest/"+imgname+".png", output)
    session['filenameSpiral'] = "static/images/spiralTest/"+imgname + ".png"
    return label


@app.route("/dashboard/spiral", methods=["POST"])
def predictSpiral():
    imagefile = request.files['imagefile']
    imagepath = "./static/images/spiralTest/"+imagefile.filename
    imagefile.save(imagepath)

    session["spiralTestResult"] = classify_my_image_spiral(imagepath)

    btnname = "spiral"
    return redirect(url_for('dashboard'))



##### WAVE PREDICTION ####
trainingPathWave = "parkinson-dataset/wave/training"
# loading the training and testing data
print("[INFO] loading data...")
(trainX1, trainY1) = load_split(trainingPathWave)
# encode the labels as integers
le1 = LabelEncoder()
trainY1 = le1.fit_transform(trainY1)



for i in range(0, 5):
    print("[INFO] training model {} of {}...".format(i + 1,5))
    model2 = RandomForestClassifier(n_estimators=100)
    model2.fit(trainX1, trainY1)




def classify_my_image_wave(image_path):
    image = cv2.imread(image_path)
    output = image.copy()
    output = cv2.resize(output, (128, 128))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (200, 200))
    image = cv2.threshold(image, 0, 255,
                          cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    features = quantify_image(image)
    preds = model2.predict([features])


    label = le1.inverse_transform(preds)[0]
    session['a'] = str(label)
    color = (0, 255, 0) if label == "healthy" else (0, 0, 255)
    cv2.putText(output, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # show the output montage
    imgname = session['username']
    imgname = imgname.replace(' ', '-').lower()
    cv2.imwrite("static/images/waveTest/"+imgname+".png", output)
    session['filenameWave'] = "static/images/waveTest/"+imgname + ".png"


@app.route("/dashboard/wave", methods=["POST"])
def predictWave():
    imagefile = request.files['imagefile']
    imagepath = "./static/images/waveTest/"+imagefile.filename
    imagefile.save(imagepath)
    classify_my_image_wave(imagepath)
    return redirect(url_for('dashboard'))


if __name__ == "__main__":
    app.secret_key = 'mysecret'
    app.run(debug=True)

