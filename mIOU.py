import tensorflow as tf

def mIOU(y, y_hat):
    '''
    from mIOU import mIOU
    
    model.compile(..., metrics=[mIOU])
    '''
    x1_hat = y_hat[:, 0]
    x2_hat = y_hat[:, 1]
    y1_hat = y_hat[:, 2]
    y2_hat = y_hat[:, 3]
    x1 = y[:, 0]
    x2 = y[:, 1]
    y1 = y[:, 2]
    y2 = y[:, 3]
    ix1 = tf.maximum(x1_hat, x1)
    ix2 = tf.minimum(x2_hat, x2)
    iy1 = tf.maximum(y1_hat, y1)
    iy2 = tf.minimum(y2_hat, y2)
    tensor_type = x1.dtype
    y_hat_area = tf.multiply(tf.maximum(tf.cast(0.0, tensor_type), x2_hat - x1_hat),
                             tf.maximum(tf.cast(0.0, tensor_type), y2_hat - y1_hat))
    y_area = tf.multiply(tf.maximum(tf.cast(0.0, tensor_type), x2 - x1),
                         tf.maximum(tf.cast(0.0, tensor_type), y2 - y1))
    i_area = tf.multiply(tf.maximum(tf.cast(0.0, tensor_type), ix2 - ix1),
                         tf.maximum(tf.cast(0.0, tensor_type), iy2 - iy1))
    u_area = y_hat_area + y_area - i_area
    iou = i_area / u_area
    return tf.reduce_mean(iou)