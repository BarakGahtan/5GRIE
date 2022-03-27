import os

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf


def get_session():
    """ Limit session memory usage
    """
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    config.gpu_options.allow_growth = True
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # GPU_ID from earlier
    sess = tf.compat.v1.Session(config=config)

    return tf.compat.v1.keras.backend.set_session(sess)


def tfSummary(tag, val):
    """ Scalar Value Tensorflow Summary
    """
    return tf.Summary.scalar(value=[tf.Summary.Value(tag=tag, simple_value=val)])
