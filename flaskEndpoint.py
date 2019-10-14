from med2image import med2image
from keras.models import load_model
from keras.preprocessing import image
import cv2
import numpy as np
import os
from flask import Flask, render_template, request, flash, redirect, request, jsonify,session
import tensorflow as tf
from werkzeug import secure_filename
from sklearn.externals import joblib
import pandas as pd
from flask_mysqldb import MySQL

nii_file=""
csv_file=""
init_val=False
fmri_model=None
em_model=None

def setModel():
    global fmri_model,em_model
    fmri_model=load_model('first_try_model.h5')
    fmri_model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

    em_model=joblib.load('eye_movement_ensembled.pkl') 
    print("called setmodel")
    print(em_model)

def getEyeMovementPrediction():
    global csv_file,em_model,init_val

    if(init_val):
        setModel()

    predict_data=pd.read_csv(csv_file,index_col=[0])
    predict_data['Gender']=pd.get_dummies(predict_data['Gender'],prefix='Gender')

    predictions=em_model.predict(predict_data)
    counts = np.bincount(predictions)
    return np.argmax(counts)
    
def getFMRIPrediction():
    global nii_file,init_val,fmri_model
    print(nii_file)
    c_convert=med2image.med2image_nii(inputFile=nii_file, outputDir="temp9",outputFileStem="image",outputFileType="png", sliceToConvert='-1',frameToConvert='0',showSlices=False, reslice=False)
    med2image.misc.tic()
    c_convert.run()

    if(init_val):
        setModel()

    images=[]

    for img in os.listdir('/home/adhd/adhd_cnn/dataFolder/temp9/'):
        img=cv2.imread('temp9/'+img)
        #img=img.astype('float')/255.0
        img=cv2.resize(img,(73,61))
        img=np.reshape(img,[1,73,61,3])
        images.append(img)

    images=np.vstack(images)

    cls=fmri_model.predict_classes(images,batch_size=10)

    #print(cls)
    print('Possibility of ADHD: ', (cls==0).sum()/len(cls))
    print('Possibility of non-ADHD: ', (cls==1).sum()/len(cls))

    adhd=(cls==0).sum()/len(cls)
    nadhd=(cls==1).sum()/len(cls)
    return adhd

app=Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = '/home/adhd/adhd_cnn/dataFolder/uploads'

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'password'
app.config['MYSQL_DB'] = 'ADHD'

mysql = MySQL(app)

@app.route('/')
def render_homepage():
   return render_template('home.html')

@app.route('/fmri_predict')
def render_fmripage():
   return render_template('health_info.html')

@app.route('/em_predict')
def render_empage():
   return render_template('health_info_em.html')

@app.route('/report')
def render_reportpage():
   return render_template('report.html')

@app.route("/predict",methods=['GET','POST'])
def predict():
    data ={'success':False}

    params=request.json
    if(params==None):
        params=request.args
    print(params)
    adhd=getPrediction()
    nadhd=1-adhd

    if(params!=None):
        data['adhd']=str(adhd)
        data['nadhd']=str(nadhd)
        data['success']=True
    return jsonify(data)

@app.route("/fmri_uploader", methods=['GET','POST'])
def upload_fmri_file():
    global nii_file,init_val
    if request.method=='POST':
        print(request.form)
        f=request.files['file']
        init_val=(nii_file=="")
        nii_file=os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename))
        f.save(nii_file)
        flash('file uploaded suceessfully')

        data ={'success':False}

        params=request.json
        if(params==None):
            params=request.args
        adhd=getFMRIPrediction()
        nadhd=1-adhd

        if(params!=None):
            data['adhd']=str(adhd)
            data['nadhd']=str(nadhd)
            data['success']=True

        if (adhd>nadhd):
            diag='ADHD'
            score=adhd
        else:
            diag='Non-ADHD'
            score=nadhd
        
        r=request.form
        if session.get('email'):
            user=session['email']
        else:
            user="Guest"
        storeData(r['fname'],r['lname'],r['email'],int(r['age']),diag,score,'fmri',user,r['symptoms']);
        
        #return jsonify(data)
        
        return redirect('/report')

def storeData(fname,lname,email,age,diag,score,data_type,user,symptoms):
    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO Diagnosis (Patient_first_name,Patient_last_name,Email,Age,Diagnosis,Composite_Score,Data_Type,User,Symptoms,Test_date,Test_time) VALUES (%s, %s,%s,%s,%s,%s,%s,%s,%s,CURDATE(),CURTIME())",(fname,lname,email,age,diag,score,data_type,user,symptoms))
    mysql.connection.commit()
    cur.close()

@app.route("/get_data", methods=['GET','POST'])
def getData():
    r=request.form
    fname=r['fname']
    lname=r['lname']
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM Diagnosis where Patient_first_name = %s && Patient_last_name =%s",(fname,lname))
    res= cur.fetchall()
    data={}
    for row in res:
        data['Patient_first_name']=row[1]
        data['Patient_last_name']=row[2]
        data['Email']=row[3]
        data['Age']=str(row[4])
        data['Diagnosis']=row[5]
        data['Composite_Score']=str(row[6])
        data['Symptoms']=row[9]

    print(data)
    cur.close()
    return jsonify(data)

    
@app.route("/em_uploader", methods=['GET','POST'])
def upload_em_file():
    global csv_file,init_val
    if request.method=='POST':
        f=request.files['file']
        init_val=(nii_file=="")
        csv_file=os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename))
        f.save(csv_file)
        flash('file uploaded suceessfully')

        data ={'success':False}

        params=request.json
        if(params==None):
            params=request.args
        print(params)
        adhd=getEyeMovementPrediction()
        nadhd=1-adhd

        if(params!=None):
            data['adhd']=str(adhd)
            data['nadhd']=str(nadhd)
            data['success']=True

        if (adhd>nadhd):
            diag='ADHD'
            score=adhd
        else:
            diag='Non-ADHD'
            score=nadhd
        
        r=request.form
        storeData(r['fname'],r['lname'],r['email'],int(r['age']),diag,score,'EM','Admin',r['symptoms']);
        
        return redirect('/report')
    
        #return jsonify(data)
        
@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == 'GET':
        if session.get('name') is not None:
            if session ['name'] != '' and session['email']!='':
                return redirect('/account')
        else:
            return render_template("login.html")
    else:
        fname = request.form['fname']
        lname = request.form['lname']
        email = request.form['email']
        password = request.form['password']

        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO User (first_name,last_name, Email, psw) VALUES (%s,%s,%s,%s)",(fname,lname,email,password,))
        mysql.connection.commit()
        session['name'] = request.form['fname']+request.form['lname']
        session['email'] = request.form['email']
        return render_template("register_success.html")

@app.route('/login',methods=["GET","POST"])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        curl = mysql.connection.cursor()
        curl.execute("SELECT * FROM User WHERE Email=%s",(email,))
        user = curl.fetchone()
        curl.close()

        if len(user) > 0:
            if password == user[3]:
                session['name'] = user[1]+user[2]
                session['email'] = user[0]
                return redirect("/account")
            else:
                return "Error password and email not match"
        else:
            return "Error user not found"
    else:
        if session ['name'] != '' and session['email']!='':
            return redirect('/account')
        else:
            return render_template("login.html")

@app.route('/logout',methods=["GET","POST"])
def logout():
    session.clear()
    return redirect("/")

@app.route('/account',methods=["GET","POST"])
def account():
    print('called account')
    if session ['name'] != '' and session['email']!='':
        print('has session')
        email=session ['email']
        curl = mysql.connection.cursor()
        curl.execute("SELECT * FROM Diagnosis inner join User ON Diagnosis.User = User.Email WHERE Diagnosis.User=%s",(email,))
        data = curl.fetchall()
        curl.close()
        return render_template('account.html',data=data);
    else:
        return render_template("login.html")
    

app.run(host='0.0.0.0')
