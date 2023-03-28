import os
import tensorflow as tf
import struct
import numpy as np
from numba import cuda

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress tf messages
#tf.disable_v2_behavior()  # Enable tf v1 behavior as in v2 a lot have changed

def gpu_release():
    '''
    Release gpu memory
    '''

    device = cuda.get_current_device()
    device.reset()
    
def gpu_session():
    '''
    Creates a gpu session
    :return: tf session
    '''
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),
                            log_device_placement=False,
                            allow_soft_placement=True)
    sess = tf.Session(config=config)

    return sess
