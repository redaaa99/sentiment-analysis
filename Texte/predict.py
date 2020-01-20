import os
import sys
import argparse
import Prediction.tweet_classification as tweet_classification
import gensim
from data.util import data_set
from glob import glob
import numpy as np
from Prediction import model as md
import pandas as pd




def get_options(parser):
    parser.add_argument('model', action='store', nargs='?', const=None, default='cnn_char', type=str, choices=None,
                        metavar=None, help='Name of model to use. (default: cnn_char)')
    return parser.parse_args()


def controller(parser, embed_model):
    model_name= parser.model
    path_dataset=parser.input_file
    out_df= pd.DataFrame(columns={"Id","Review","Golden","Pred_sentiment"})
    # load data
    data = data_set(path_dataset)
    print("data",len(data.get("sentence")))
    print(data)
    architecture, pre_process, model_inputs = tweet_classification.get_model_instance(model_name , embed_model))

    _list = glob("./log/%s/*" % model_name)
    print("Sentence classifier ('q' to quite)")
    print("- saved list (model: %s )" % model_name)
    for __ind, __l in enumerate(_list):
        print("  id: %i, name: %s" % (__ind, __l))


    #print(type(model))
    print("get model")
    model = md.CharCNN(architecture, load_model="./log/cnn_char/last/progress-50-model.ckpt.meta")
    print ("get balanced_validation_split")
    target=data["sentence"]
    label= data["label"]
    idx= data["Id"]
    
    print("data to valid size",len(target))
    print("the data", target)

    _data = processing(target, pre_process)
    print("data preprocesed")
    feed_dict = model_inputs(model, _data)
    print("model feeded")
    feed_dict[model.is_train] = False    
    print("start prediction")

    prediction = model.sess.run([model.prediction], feed_dict=feed_dict)
    print ("end prediction")
    acc = 0      
    print ("start calculate accuracy")
    i=0
    for _p, _l, _s in zip(prediction[0], label, target):
            _e = int(_p > 0.5)
            acc += (_e == _l)
            out_df.loc[i]= [idx[i],_s,_l,_p]
            i=i+1
            print("est: %i (%0.3f), true: %i, sentence: %s" % (_e, _p, _l, _s))
    acc = acc / len(label)
    out_df.to_csv(parser.out_file,sep=";",index=False)
    print("accuracy:%0.3f" % acc)


def processing(x, process):
    if process is not None:
        if type(process) == list:
            return [_process(x) for _process in process]
        else:
            return process(x)
    else:
        return x


def balanced_validation_split(x, y, idx, ratio):
    """y should be 1 dimension array"""
    _ind = [i for i in range(len(x))]
    np.random.seed(0)
    np.random.shuffle(_ind)
    y, x,idx = y[_ind], x[_ind], idx[_ind]
    size = int(np.floor(len(x) * ratio) / 2)
    # binary label index
    _y0 = y[y == 0]
    _y1 = y[y == 1]
    _x0 = x[y == 0]
    _x1 = x[y == 1]
    _idx0= idx[y==0]
    _idx1 = idx[y == 1]
    _ind = int(np.min([np.min([len(_y0), len(_y1)]), size]))
    y_valid = np.hstack([_y0[:_ind], _y1[:_ind]])
    if x.ndim == 1:
        x_valid = np.hstack([_x0[:_ind], _x1[:_ind]])
        idx_valid = np.hstack([_idx0[:_ind], _idx1[:_ind]])
    else:

        x_valid = np.vstack([_x0[:_ind], _x1[:_ind]])
        idx_valid = np.vstack([_idx0[:_ind], _idx1[:_ind]])

    return x_valid, y_valid ,idx_valid



if __name__ == '__main__':
    # Ignore warning message by tensor flow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # model args
    parser = argparse.ArgumentParser (description='Syntactically Controlled Paraphrase Transformer')

    parser.add_argument ('--input_file' , type=str , default="./"
                                                             "news_headlines.csv" ,
                                                              help=' load path')
    parser.add_argument ('--out_file' , type=str , default='./data/news_headlines_CNN2.csv' ,
                         help='data save path')

    args = get_options(parser)
    # w2v
    w2v = gensim.models.KeyedVectors.load_word2vec_format("./data/GoogleNews-vectors-negative300.bin", binary=True)
    # load model
    controller(args, w2v)






