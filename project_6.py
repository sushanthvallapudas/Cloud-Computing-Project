import pandas as pd
from sklearn.model_selection import  train_test_split
import numpy as np
import matplotlib.pyplot as plt
import argparse
import datetime
import json


def init():
#     Cloud _parameters file contain parameters(cpu,dimm,disk,failure) of previous data
    df=pd.read_csv("cloud_parameters.csv")
    df=df.drop("id",1)
    df['failure']=df['failure'].map({'Y':1,'N':0})
    train,test=train_test_split(df,test_size=0.3,random_state=1)
#     print(train)
#     print(test)
    train_x=train.loc[:,'CPU':'DISK']
    train_y=train.loc[:,['failure']]
    test_x=test.loc[:,'CPU':'DISK']
    test_y=test.loc[:,['failure']]
    train_x=np.asarray(train_x)
    train_y=np.asarray(train_y)
    test_x=np.asarray(test_x)
    test_y=np.asarray(test_y)
    m=model(train_x.T,train_y.T,10000,.000001)
    costs=m["costs"]
    w=m["Y"]
    b=m["N"]
    m["w"]=m["Y"]
    m["b"]=m["N"]
    plt.plot(costs)
    plt.title("Costs vs Iteration")
    plt.show()
    y_prediction_test=predict(test_x.T,w,b)
    print(y_prediction_test)

    print(100-(np.mean(y_prediction_test-test_y))*100)
def initialize(m):
    w=np.zeros((m,1))
    b=0
    return w,b
def sigmoid(X):
    return 1/(1+np.exp((-X)))
def propagate(X,Y,w,b):
    m=X.shape[1]
    Z=np.dot(w.T,X)+b
    A=sigmoid(Z)
    cost= -(1/m)* np.sum(Y * np.log(A)+ (1-Y)* np.log(1-A))
    # Backward Propagation
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)

    grads = {"dw": dw, "db": db}

    return grads, cost
def optimize(X,Y,w,b,n,per):
    c=[]
    for i in range(n):
        grads, cost=propagate(X,Y,w,b)
        dw= grads['dw']
        db = grads['db']
        w=w-per*dw
        b=b-per*db
        if(i%10==0):
            c.append(cost)
    p={"w":w,"b":b}
    g={"dw":dw,"db":db}
    return p,g,c
def predict(X,w,b):
    m=X.shape[1]
    y_pred=np.zeros((1,m))
    w=w.reshape(X.shape[0],1)
    A=sigmoid(np.dot(w.T,X)+b)
    for i in range(A.shape[1]):
        if(A[0,i]<0.5):
            y_pred[0,i]=0
        else:
            y_pred[0,i]=1
	    

            def main(description, project_id, start_date, start_time, source_bucket,access_key_id, secret_access_key, sink_bucket):
                """Create a one-time transfer from Amazon S3 to Google Cloud Storage."""
                storagetransfer = googleapiclient.discovery.build('storagetransfer', 'v1')

                # We can edit this template with our desired parameters
                transfer_job = {
                    'description': description,
                    'status': 'ENABLED',
                    'projectId': project_id,
                    'schedule': {
                    'scheduleStartDate': {
                    'day': start_date.day,
                    'month': start_date.month,
                    'year': start_date.year
                    },
                    'scheduleEndDate': {
                    'day': start_date.day,
                    'month': start_date.month,
                    'year': start_date.year
                    },
                    'startTimeOfDay': {
                    'hours': start_time.hour,
                    'minutes': start_time.minute,
                    'seconds': start_time.second
                    }
                },
                    'transferSpec': {
                    'awsS3DataSource': {
                    'bucketName': source_bucket,
                    'awsAccessKey': {
                    'accessKeyId': access_key_id,
                    'secretAccessKey': secret_access_key
                    }
                },
                    'gcsDataSink': {
                    'bucketName': sink_bucket
                    }
                }
            }

    result = storagetransfer.transferJobs().create(body=transfer_job).execute()
    print('Returned transferJob: {}'.format(
        json.dumps(result, indent=4)))


    if __name__ == '__main__':
        parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument('description', help='Transfer description.')
        parser.add_argument('project_id', help='Your Google Cloud project ID.')
        parser.add_argument('start_date', help='Date YYYY/MM/DD.')
        parser.add_argument('start_time', help='UTC Time (24hr) HH:MM:SS.')
        parser.add_argument('source_bucket', help='AWS source bucket name.')
        parser.add_argument('access_key_id', help='Your AWS access key id.')
        parser.add_argument(
        'secret_access_key',
        help='Your AWS secret access key.'
        )
        parser.add_argument('sink_bucket', help='GCS sink bucket name.')

        args = parser.parse_args()
        start_date = datetime.datetime.strptime(args.start_date, '%Y/%m/%d')
        start_time = datetime.datetime.strptime(args.start_time, '%H:%M:%S')

        main(
        args.description,
        args.project_id,
        start_date,
        start_time,
        args.source_bucket,
        args.access_key_id,
        args.secret_access_key,
        args.sink_bucket)


    return y_pred


def model(Xtrain,Ytrain,n,per):
    dim=Xtrain.shape[0]
    w,b=initialize(dim)
#     print(w,b)
    param,grads,cost=optimize(Xtrain,Ytrain,w,b,n,per)
    w=param["w"]
    b=param["b"]
    m={"w":w,"b":b,"costs":cost}
    return m
init()

