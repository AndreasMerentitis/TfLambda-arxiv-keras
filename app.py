from flask import Flask, render_template, request

import boto3

import datetime
import json
import os
import pickle
import numpy as np

import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import model_from_json, model_from_config, load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import text

import logging

target_name_dict = {'stat.AP' : 0,
                    'stat.CO' : 1,
                    'stat.ME' : 2,
                    'stat.ML' : 3,
                    'stat.OT' : 4
                    }

label2target = { v:k for k,v in target_name_dict.items()}

app = Flask(__name__)

#@app.route("/")
@app.route("/", methods=['GET', 'POST'])
def main():
   logging.warning('Starting the app')

   boto3.Session().resource('s3').Bucket('serverless-ml-1').download_file('tokenizer.pickle', '/tmp/tokenizer.pickle')
   boto3.Session().resource('s3').Bucket('serverless-ml-1').download_file('model_ML.h5', '/tmp/model_ML.h5')
   
   with open('/tmp/tokenizer.pickle', 'rb') as handle:
       tokenizer = pickle.load(handle)

   model = load_model('/tmp/model_ML.h5')
   
   now = datetime.datetime.now()
   timeString = now.strftime("%Y-%m-%d %H:%M")
   cpuCount = os.cpu_count()
   templateData = {
      'title' : 'Web App for classifying abstracts on statistics',
      'time': timeString,
      'cpucount' : cpuCount,
      'tfversion' : tf.__version__
      }
      
   logging.warning('request.method is %s', request.method)
   
   if request.method == 'GET':
      return render_template('index.html', **templateData)
   elif request.method == 'POST':
      resultText = "You wrote: " + request.form['myTextArea']
      
      logging.warning('request text is %s', request.form['myTextArea'])
      
      seq_1 = tokenizer.texts_to_sequences(request.form['myTextArea'])
      seq_2 = pad_sequences(seq_1, padding='post', value=0, maxlen=350)
      
      logging.warning('seq_2 is %s', seq_2)
           
      prob = model.predict(seq_2)
      #prob /= prob.sum()
      prob = prob.sum(axis=0)
      logging.warning('prob is %s', prob)
      ii = np.argmax(prob)
      logging.warning('ii is %s', ii)
      if max(prob) >= 20:
         final_label = label2target[ii]
      else: 
         final_label = 'not a stats abstract'
         
      logging.warning('final_label is %s', final_label)

      results = {'text' : resultText, 'label' : final_label}
      return render_template('index.html', results=results, **templateData)


if __name__ == "__main__":
   app.run()
