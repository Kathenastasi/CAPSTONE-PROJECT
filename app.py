from flask import Flask, render_template, request, redirect, url_for
import pickle
import json
import numpy as np
#from keras.models import Sequential
#import keras.models
#import tensorflow

app = Flask(__name__ , template_folder= 'templates')

model_lstm2 = pickle.load(open('LSTM_multivar_model', 'rb'))

@app.route('/')
def temp():
    return render_template('index.html')

#@app.route('/',methods=['POST','GET'])
#def get_input():
#    if request.method == 'POST':
#       info = request.json
#       print (info)
#       return redirect(url_for('run_pred',values=info))

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_temp
@app.route('/run_pred/<values>')
def run_pred(values):
    import numpy as np 
    print(values)
    #values = values.split(',')
    #values = np.array(values).astype('float')
    #values = values.reshape(1,-1)
    values = values['mytext']
    
    with open('../timeseries_model', 'rb') as file:
        pickle_model = pickle.load(file)
        
    model = pickle_model
    pred = model.predict(values)
    probability = model.predict_proba(values)

    json_object =  {'predicted': pred, 'probability': probability}

    if pred == 0:
        return jsonify(json_object)
    return jsonify(json_object)
    
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5100, debug=True, threaded=True)
