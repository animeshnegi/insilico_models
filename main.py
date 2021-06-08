
from flask import Flask,render_template,request,url_for,redirect,send_file
from numpy.core.numeric import True_
from werkzeug.utils import secure_filename
import json
from werkzeug.utils import secure_filename 
import os
import base64
# library for machine learning

import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.svm import SVC
import pickle
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier

date = datetime. now(). strftime("%d_%m_%Y---%I_%M_%S_%p")

with open('config.json','r')  as c:
    param = json.load(c)["param"]

app = Flask(__name__)

#location for test data 
app.config['UPLOAD_FOLDER'] = param['Upload_location']

#location for smile data 
app.config['SMILE_FOLDER'] = param['SMILE_LOCATION']

#location for data_folder
app.config['data_folder'] = param['data_folder']

#location for Model_folder
app.config['Model_folder'] = param['Model_folder']



app.secret_key = 'hello-hi-there'
ALLOWED_EXTENSIONS = {'csv'}
ALLOWED_EXTENSIONS_for_des = {'smi','txt'}

########### Function to take only csv file 
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


########### Function to take only .smi file 
def allowed_file_des(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_for_des


########### Function to convert string 
def che1(check1):
    if check1 == 'on':
        che = '-removesalt'
        return che
    else:
        che = ''
        return che


def che2(check1):
    if check1 == 'on':
        che = '-standardizenitro'
        return che
    else:
        che = ''
        return che

########### Function to genrate bash string 

def bash(desc,fingerprint):
    tree = ET.parse('PaDEL-Descriptor/yo.xml')
    root = tree.getroot()

    if desc == '2D' and bool(fingerprint) == True :
        for Descriptor in root.findall("./Group/[@name='%s']/Descriptor" % (desc) ):
            Descriptor.set('value','true')
           
        
        for Descriptor in root.findall("./Group/[@name='Fingerprint']/Descriptor/[@name='%s']" % (fingerprint)):
            Descriptor.set('value','true')

        tree.write('output.xml')
    
    if desc == '3D' and bool(fingerprint) == True :
        for Descriptor in root.findall("./Group/[@name='%s']/Descriptor" % (desc) ):
            Descriptor.set('value','true')
           
        
        for Descriptor in root.findall("./Group/[@name='Fingerprint']/Descriptor/[@name='%s']" % (fingerprint)):
            Descriptor.set('value','true')

        tree.write('output.xml')

    if desc == 'no' and bool(fingerprint) == True :  
        
        for Descriptor in root.findall("./Group/[@name='Fingerprint']/Descriptor/[@name='%s']" % (fingerprint)):
            Descriptor.set('value','true')

        tree.write('output.xml')

    if desc == 'no' and fingerprint == 'no' :
        for Descriptor in root.findall("./Group/Descriptor" ):
            Descriptor.set('value','false')
        tree.write('output.xml')
        value = True
        return value

    if desc == '2D' and fingerprint == 'no' :
        for Descriptor in root.findall("./Group/[@name='%s']/Descriptor" % (desc) ):
            Descriptor.set('value','true')

        tree.write('output.xml')

    if desc == '3D' and fingerprint == 'no' :
        for Descriptor in root.findall("./Group/[@name='%s']/Descriptor" % (desc) ):
            Descriptor.set('value','true')
            print(Descriptor.attrib)
        tree.write('output.xml')
        
    if desc == 'both' and fingerprint == 'no' :
        two = '2D'
        three = '3D'
        for Descriptor in root.findall("./Group/[@name='%s']/Descriptor" % (two) ):
            Descriptor.set('value','true')

        for Descriptor in root.findall("./Group/[@name='%s']/Descriptor" % (three) ):
            Descriptor.set('value','true')           
        tree.write('output.xml')


# CAL CULATING DESCRIPTOR 
def desc_calc(removesalt,standard,filename):
    # Performs the descriptor calculation
    bashCommand = "java -Xms8G -Xmx64G -Djava.awt.headless=true -jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar %s %s -fingerprints -2d -descriptortypes output.xml -dir %s -file descriptors_output.csv" % (removesalt,standard,filename) 
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    desc = pd.read_csv('descriptors_output.csv')
    return desc

# for define a variable global
def glo(x):
    global fin 
    fin = x

@app.route("/des_cal", methods = ['GET','POST'])
def des_cal():
    if (request.method == 'POST'):
        fil = request.files['des_fil']
        desc = request.form.get("desc")
        check1 = request.form.get("check1")
        check2 = request.form.get("check2")
        fingerprint = request.form.get("fingerprint") 
        glo(fingerprint)
        if fil.filename == '':
            file_value = 3 
            return render_template('/descriptor.html',value = file_value) 
        


        if fil and allowed_file_des(fil.filename):
            filename = secure_filename(fil.filename)
            fil.save(os.path.join(app.config['SMILE_FOLDER'])+'\\'+filename)
            error = 111

# -------------MAking .XML file -----------------
            removesalt = che1(check1)
            standard =  che2(check2)
            bash(desc,fingerprint)
# -----------calculating descriptor-------------------
            desc_calc(removesalt,standard,filename)
            return redirect(url_for('download_des'))


        else:
            error = 404
            return render_template('/descriptor.html',error = error)



@app.route('/download_des.html')
def download_des():
    desc = pd.read_csv('descriptors_output.csv')
    nmol = desc.shape[0]
    ndesc = desc.shape[1]    
    df = desc.to_html(classes="table table-striped")
    return render_template('/download_des.html',csv = df,nmol = nmol, ndesc = ndesc,fin = fin)






#       +_________________________MODEL FIND




@app.route("/upload/<path:model>", methods = ['GET','POST'])
def upload(model):
     if (request.method == 'POST'):
        fil = request.files['fil']
        val = True
        if fil.filename == '':
            file_value = 3 
            return render_template('/model.html',value = file_value, val  = val, file=model) 

        if fil and allowed_file(fil.filename):
            filename = secure_filename(fil.filename)
            fil.save(os.path.join(app.config['UPLOAD_FOLDER'])+'\\'+filename)

    #loading the model from the specific folder
            loaded_model = pickle.load(open(app.config['Model_folder']+'\\'+model,'rb'))


            #loading the model from the specific folder
            train = pd.read_csv(app.config['data_folder']+'\\'+filename)
            train = pd.DataFrame(train)
            #data modification
            test_main = train.iloc[:,1:len(train.columns)]
            test_activity = train.iloc[:,0]
            
        # making prediction
            clf_predict = loaded_model.predict(test_main)
            con_mat = confusion_matrix(test_activity, clf_predict)
            report = classification_report(test_activity, clf_predict)


            df = train.head()
            df = df.to_html(classes="table table-striped")

            return render_template('/any_model.html',report = report,con_mat = con_mat,csv = df,model = model)

        else:
            error = 404
            return render_template('/model.html',error = error, val  = val, file=model)




# --------------------------------------model page ..........

@app.route('/model.html/<path:filenames>', methods = ['GET','POST'])
def model(filenames):
    file = filenames
    val = bool(filenames)
    return render_template('/model.html',file = file, val = val)

@app.route('/model.html', methods = ['GET','POST'])
def modele():
    val = ''
    return render_template('/model.html', val  = val)











#______________SCRIPT FOR SVM MODEL download


@app.route('/model_download/<path:filenames>', methods=['GET', 'POST'])
def model_download(filenames):
    path = "C:/Users/animesh negi/Desktop/Web_app/downloads/models/%s" % filenames
    return send_file(path, as_attachment=True)


#______________SCRIPT FOR SVM MODEL BUILDING 

@app.route("/SVM_model", methods = ['GET','POST'])
def SVM_model():
    if (request.method == 'POST'):
        fil = request.files['train_fil']
        col = request.form.get('col_name')
        
        if fil.filename == '':
            file_value = 3 
            return render_template('models/svm',value = file_value)
        
        if fil and allowed_file(fil.filename):
            C = request.form.get('C')
            gamma = request.form.get('gamma')
            kernal = request.form.get('kernal')
            cache_size = request.form.get('cache_size')

            C= float(C)
            cache_size = float(cache_size)

            filename = secure_filename(fil.filename)
            fil.save(os.path.join(app.config['UPLOAD_FOLDER'])+'\\'+filename)

            #loading the model from the specific folder
            train = pd.read_csv(app.config['data_folder']+'\\'+filename)
            train = pd.DataFrame(train)
            #data modification

            activity = train.iloc[:,0]
            data = train.iloc[:,1:len(train.columns)]
            
            
            # building model
            svm = SVC(C=C, kernel= kernal, gamma= gamma, cache_size = cache_size)
            svm.fit(data, activity)

            filenames  = filename+'_RF_Model_with n_Est=%s_date=%s.pkl' % (kernal,date)

            path  = app.config['Model_folder']+'\\'+filenames


            # building model
            with open(path, 'wb') as f:
                pickle.dump(svm, f)


            df = train.head()
            df = df.to_html(classes="table table-striped")


            # prediction on train dataset 
            clf_predict = svm.predict(data)
            con_mat = confusion_matrix(activity, clf_predict)
            
            report  = classification_report(activity, clf_predict)

 
            return render_template('models/svm_model.html',report = report,con_mat = con_mat,filenames = filenames,csv = df,kernal= kernal)

        else:
            error = 404
            return render_template('models/svm',error = error)




#______________SCRIPT FOR Random Forest  MODEL BUILDING 

@app.route("/RF_model", methods = ['GET','POST'])
def RF_model():
    if (request.method == 'POST'):
        fil = request.files['train_fil']
        col = request.form.get('col_name')

        if fil.filename == '':
            file_value = 3 
            return render_template('models/rf',value = file_value)
        
        if fil and allowed_file(fil.filename):
            n_estimators = request.form.get('n_estimators')
            n_jobs = request.form.get('n_jobs')
            min_sam = request.form.get('min_sam')
            Max_depthint = request.form.get('Max_depthint')
            max_feature = request.form.get('max_feature')

            filename = secure_filename(fil.filename)
            fil.save(os.path.join(app.config['UPLOAD_FOLDER'])+'\\'+filename)

            #loading the model from the specific folder
            train = pd.read_csv(app.config['data_folder']+'\\'+filename)
            train = pd.DataFrame(train)
            #data modification

            activity = train.iloc[:,0]
            data = train.iloc[:,1:len(train.columns)]
            
            
            # building model

            
            model = RandomForestClassifier(n_estimators=100,n_jobs=None,min_samples_split=2,max_depth=None,max_features='auto')

            filenames  = filename+'_RF_Model_with n_Est=%s_date=%s.pkl' % (n_estimators,date)

            path  = app.config['Model_folder']+'\\'+filenames

            df = train.head()
            df = df.to_html(classes="table table-striped")

            # prediction on train dataset 
            model = RandomForestClassifier()
            model.fit(data, activity)

            # Saving model
            with open(path, 'wb') as f:
                pickle.dump(model, f)

            clf_predict = model.predict(data)
            con_mat = confusion_matrix(activity, clf_predict)
            
            report  = classification_report(activity, clf_predict)


            return render_template('models/rf_model.html',filename = filename,filenames = filenames,csv = df,path = path ,
            n_estimators = n_estimators,n_jobs = n_jobs,min_sam =  min_sam ,Max_depthint = Max_depthint,max_feature = max_feature,
            train = train,report = report,con_mat = con_mat )

        else:
            error = 404
            return render_template('models/rf',error = error)





#______________SCRIPT FOR KNN  MODEL BUILDING 

@app.route("/knn_model", methods = ['GET','POST'])
def Knn_model():
    if (request.method == 'POST'):
        fil = request.files['train_fil']
        col = request.form.get('col_name')

        if fil.filename == '':
            file_value = 3 
            return render_template('models/knn',value = file_value)
        
        if fil and allowed_file(fil.filename):
            n_neighbors = request.form.get('n_neighbors')
            leaf_size = request.form.get('leaf_size')
            weights = request.form.get('weights')
            algorithm = request.form.get('algorithm')

            filename = secure_filename(fil.filename)
            fil.save(os.path.join(app.config['UPLOAD_FOLDER'])+'\\'+filename)

            #loading the model from the specific folder
            train = pd.read_csv(app.config['data_folder']+'\\'+filename)
            train = pd.DataFrame(train)
            #data modification

            activity = train.iloc[:,0]
            data = train.iloc[:,1:len(train.columns)]
            
            
            # building model

            
            model = KNeighborsClassifier(n_neighbors= n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size) 

            filenames  = filename+'_KNN_Model_with n_Neigb = %s_date=%s.pkl' % (n_neighbors,date)

            path  = app.config['Model_folder']+'\\'+filenames
            # building model
            # with open(path, 'wb') as f:
            #     pickle.dump(model, f)

            df = train.head()
            df = df.to_html(classes="table table-striped")


            # prediction on train dataset 
            model = RandomForestClassifier()
            model.fit(data, activity)

            # Saving model
            with open(path, 'wb') as f:
                pickle.dump(model, f)

            clf_predict = model.predict(data)
            con_mat = confusion_matrix(activity, clf_predict)
            
            report  = classification_report(activity, clf_predict)


            return render_template('models/knn_model.html',filename = filename,filenames = filenames,csv = df,path = path ,
            n_neighbors = n_neighbors, leaf_size = leaf_size, weights=weights, algorithm = algorithm,
            train = train,report = report,con_mat = con_mat )

        else:
            error = 404
            return render_template('models/knn',error = error)




@app.route('/build_model.html', methods = ['GET','POST'])
def build_model():
    if (request.method == 'POST'):
        model = request.form.get('check')
        if model == 'SVM':
            return render_template('models/svm')
        if model == 'RF':
            return render_template('models/rf')    
        if model == 'KNN':
            return render_template('models/knn')

    return render_template('/build_model.html')



#search page spurce code

@app.route('/search.html')
def search():
    x = next(os.walk('downloads/models'))[2]
    p = len(x)
    return render_template('/search.html',x= x , p = p)  

@app.route('/')
def index():
    return render_template("/index.html")

@app.route('/descriptor.html')
def descripter():
    return render_template("/descriptor.html")

@app.route('/index.html')
def index1():
    return render_template("/index.html")


@app.route('/try.html')
def trya():
    return render_template('/try.html')




@app.route('/about.html')
def about():
    return render_template('/about.html')    

@app.route('/contact.html')
def contact():
    return render_template('/contact.html')    

@app.route('/var_models.html')
def var_models():
    return render_template('/var_models.html')  


app.run(debug=True)




























