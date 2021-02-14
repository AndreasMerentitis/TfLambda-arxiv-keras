# Serverless Machine Learning inference on AWS Lambda with TensorFlow / keras and flask front end

This project classifies a given abstract in different subfields of Machine Learning (using labels from arxiv). 
Under the hood there is a TensorFlow NLP classification model deployed to AWS Lambda using the Serverless framework.
The model is trained locally with keras. The abstract is submited unsing a web form that is powered by flask and
wsgi. 

by: Andreas Merentitis

![relative path 6](/deploy.png?raw=true "deploy.png")
![relative path 1](/infer.png?raw=true "infer.png")

### Prerequisites

The project requires a pre-trained keras model on a certain subset of arxiv. 
For downloading the required data please check the repository arxiv_collector.
The file train_local_data.py can be used to train the model using these data.


#### Setup serverless

```  
sudo npm install -g serverless

sudo serverless plugin install -n serverless-python-requirements

sudo serverless plugin install -n serverless-wsgi

pip install -r requirements.txt

```
#### Setup AWS credentials

Make sure you have AWS access key and secrete keys setup locally, following this video [here](https://www.youtube.com/watch?v=KngM5bfpttA)

### Download the code locally

```  
serverless create --template-url https://github.com/AndreasMerentitis/TfLambda-arxiv-keras --path TfLambda-arxiv
```

### Update S3 bucket to unique name
In serverless.yml:
```  
  environment:
    BUCKET: <your_unique_bucket_name> 
```

### Check the file syntax for any files changed 
```
pyflakes infer.py

```
We can ignore the warning about not using 'unzip_requirements' as its needed to set the requirements for lamda 

### Train the model from scratch

```
source activate py36

python local_train_new_model.py 

```


### Deploy to the cloud  


```
cd tf-lambda

npm install

sudo serverless deploy --stage dev

aws s3 cp model_ML.h5 s3://serverless-ml-1/ --grants read=uri=http://acs.amazonaws.com/groups/global/AllUsers

aws s3 cp tokenizer.pickle s3://serverless-ml-1/ --grants read=uri=http://acs.amazonaws.com/groups/global/AllUsers

curl -X GET https://syrqz8iwfd.execute-api.eu-west-1.amazonaws.com/dev

curl -X GET https://syrqz8iwfd.execute-api.eu-west-1.amazonaws.com/dev/{proxy+}

```

### Clean up (remove deployment) 


```
aws s3 rm s3://serverless-ml-1 --recursive

sudo serverless remove --stage dev 
```

# Using data and extending the basic idea from these sources:
* https://github.com/mikepm35/TfLambdaDemo
* https://medium.com/@mike.p.moritz/running-tensorflow-on-aws-lambda-using-serverless-5acf20e00033
* https://github.com/wingkitlee0/arxiv_explore
* https://github.com/wingkitlee0/arxiv_collector
* https://github.com/alexdebrie/serverless-flask
* https://www.fernandomc.com/posts/developing-flask-based-serverless-framework-apis/
* https://pypi.org/project/serverless-wsgi/









