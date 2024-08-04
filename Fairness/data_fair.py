import csv
import numpy as np


fn="propublica_data_for_fairml.csv"


def load_data():
    with open(fn, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        data = np.array(data)

    # remove header
    protected=4
    print(data[0][protected+1],"is protected")
    headers=data[0]
    data=data[1:]

    labels=data[:,0]
    labelnam=headers[0]
    headers=headers[1:]
    data=data[:,1:]

    protected_group=data[:,protected]
    data=np.delete(data,protected,1)
    data=np.concatenate((protected_group[:,None], data),axis=1)
    data=data.astype(float)

    protected_name=headers[protected]
    headers=np.delete(headers,protected)
    headers=[protected_name]+list(headers)

    norm=[d for d,l in zip(data,labels) if l=="0"]
    abnorm=[d for d,l in zip(data,labels) if l=="1"]
    norm=np.array(norm)
    abnorm=np.array(abnorm)

    print(len(norm),len(abnorm))

    border=2500
    train=norm[:border]
    test_n=norm[border:]
    test_a=abnorm
    test=np.concatenate((test_n,test_a),axis=0)
    testy=np.concatenate((np.zeros(len(test_n)),np.ones(len(test_a))),axis=0)

    return train, test, testy, {"labelnam":labelnam,"headers":headers,"protected":protected_name}



if __name__=="__main__":
    x,tx,ty,info=load_data()

    print(x.shape)
    print(tx.shape)
    print(ty.shape)
    print(info)


    np.savez_compressed("data.npz",x=x,tx=tx,ty=ty,**info)


    



