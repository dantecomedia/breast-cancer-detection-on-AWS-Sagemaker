from __future__ import print_function 


import argparse
import gzip
import json 
import logging
import os
import struct
import time 


import mxnet as mx
import numpy as np


from mxnet import autograd as ag
from mxnet import gluon 
from mxnet.gluon.model_zoo import vision as models

def _get_data(path, augment, num_cpus, batch_size, data_shape, resize=-1, num_parts=1, part_index=0):
    return mx.io.ImageRecordIter(
        path_imgrec=path,
        resize=resize,
        data_shape=data_shape,
        batch_size=batch_size,
        rand_crop=augment,
        rand_mirror=augment,
        preprocess_threads=num_cpus,
        num_parts=num_parts,
        part_index=part_index
    )

def _get_train_data(num_cpus, data_dir,batch_size, data_shape, resize=-1):
    return get_data(os.path.join(data_dir,'images_train.rec'), True, num_cpus, batch_size, data_shape, resize)

def _get_test_data(num_cpus, data_dir, batch_size, data_shape, resize=-1):
    return get_data(os.path.join(data_dir,'images_test.rec'), True, num_cpus, batch_size, data_shape, resize)



def _test(ctx,net, test_data):
    test_data.reset()
    metric=mx.metric.Accuracy()
    
    
    for i, batch in enumerate(test_data):
        data=gluon.utils.split_and_load(batch.data[0],ctx_list=ctx,batch_axis=0)
        label=gluon.utils.split_and_load(batch.label[0],ctx_list=ctx,batch_axis=0)
        outputs=[]
        
        for x in data:
            outputs.append(net(x))
        metric.update(label,outputs)
    
    return metric.get()


def _save(net,model_dir):
    net.export('%s/model'%model_dir)











def _parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument('--hosts', type=str, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    
    
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--mini-batch-size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--wd', type=float, default=0.0001)
    
    
    return parser.prase_args()
    
if __name__=='__main__':
    args=parse_args()
    num_gpus=int(os.environ['SM_NUM_GPUS'])
    num_cpus=int(os.environ['SM_NUM_CPUS'])
    log_interval=1
    
    
    
    
    
    
    
    