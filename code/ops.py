import tensorflow as tf

def fc(input, 
       in_dim, 
       out_dim, 
       activation=tf.nn.relu, 
       l2_reg_const=0.0,
       isTraining=True):
    
#     n_inputs = int(input.get_shape()[1])
#     norm_mean = 0.0
#     norm_stddev = 2 / np.sqrt(n_inputs)
    
    fc_W = tf.get_variable('weights',
                           (in_dim, out_dim),
                           #initializer=tf.contrib.layers.variance_scaling_initializer(),
                           #initializer=tf.truncated_normal_initializer(norm_mean, norm_stddev),
                           initializer=tf.contrib.layers.xavier_initializer(),
                           regularizer=tf.contrib.layers.l2_regularizer(l2_reg_const))
    variable_summaries(fc_W)
    
    # fc_b = tf.get_variable('biases',
                           # (out_dim),
                           # initializer=tf.zeros_initializer())
    # variable_summaries(fc_b)
    
    # fc = tf.nn.xw_plus_b(input, fc_W, fc_b)
    
    fc = tf.matmul(input, fc_W)
    tf.summary.histogram('pre_activations', fc)
    
    fc = tf.contrib.layers.batch_norm(fc, 
                                      center=True, 
                                      scale=True, 
                                      is_training=isTraining)
    
    if activation is not None:
        fc = activation(fc)
        tf.summary.histogram('activations', fc)
    
    return fc

def conv2d(input,
           filter_shape,
           strides=[1, 1, 1, 1], 
           padding='SAME',
           activation=tf.nn.relu, 
           l2_reg_const=0.0,
           isTraining=True):
    
#     n_inputs = int(input.get_shape()[1])
#     norm_mean = 0.0
#     norm_stddev = 2 / np.sqrt(n_inputs)
    
    filter = tf.get_variable('filters',
                             filter_shape,
                             #initializer=tf.contrib.layers.variance_scaling_initializer(),
                             initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                             #initializer=tf.truncated_normal_initializer(norm_mean, norm_stddev),
                             regularizer=tf.contrib.layers.l2_regularizer(l2_reg_const))
    variable_summaries(filter)
    
    biases = tf.get_variable('biases',
                             (filter_shape[3]),
                             initializer=tf.zeros_initializer())
    variable_summaries(biases)

    conv2d = tf.nn.conv2d(input, filter, strides=strides, padding=padding)
    conv2d = tf.nn.bias_add(conv2d, biases)
    

#     conv2d = tf.contrib.layers.batch_norm(conv2d, 
#                                           center=True, 
#                                           scale=True, 
#                                           is_training=isTraining,
#                                           scope='bn')
    
    if activation is not None:
        conv2d = activation(conv2d)
        tf.summary.histogram('activation', conv2d)
    
    return conv2d

def inception_module(input, 
                     in_channels, 
                     filter_count,
                     l2_reg_const=0.0,
                     isTraining=True):
    
    # Branch 1x1 : 1x1 conv
    with tf.variable_scope("1x1_branch"):
        with tf.variable_scope("1x1_conv2d"):
            branch1x1 = conv2d(input,
                               (1, 1, in_channels, filter_count), 
                               l2_reg_const=l2_reg_const, 
                               isTraining=isTraining)
        
    # Branch 3x3 : 1x1 -> 3x3
    with tf.variable_scope("1x1_3x3_branch"):        
        with tf.variable_scope("1x1_conv2d"):
            branch3x3 = conv2d(input, 
                               (1, 1, in_channels, 1), 
                               l2_reg_const=l2_reg_const,
                               isTraining=isTraining)
            
        with tf.variable_scope("3x3_conv2d"):
            branch3x3 = conv2d(branch3x3, 
                               (3, 3, 1, filter_count), 
                               l2_reg_const=l2_reg_const,
                               isTraining=isTraining)
    
    # Branch 5x5 : 1x1 -> 5x5
    with tf.variable_scope("1x1_5x5_branch"):
        with tf.variable_scope("1x1_conv2d"):
            branch5x5 = conv2d(input, 
                               (1, 1, in_channels, 1), 
                               l2_reg_const=l2_reg_const, 
                               isTraining=isTraining)
            
        with tf.variable_scope("5x5_conv2d"):
            branch5x5 = conv2d(branch5x5, 
                               (5, 5, 1, filter_count), 
                               l2_reg_const=l2_reg_const, 
                               isTraining=isTraining)

    # Branch max pool : 3x3 max pooling -> 1x1
    with tf.variable_scope("max_pool_1x1_branch"):
        with tf.variable_scope("max_pool"):
            branch_pool = tf.nn.max_pool(input, 
                                         ksize=[1, 3, 3, 1], 
                                         strides=[1, 1, 1, 1], 
                                         padding='SAME')
            
        with tf.variable_scope("1x1_conv2d"):
            branch_pool = conv2d(branch_pool, 
                                 (1, 1, in_channels, filter_count), 
                                 l2_reg_const=l2_reg_const, 
                                 isTraining=isTraining)
    
    inception_module = tf.concat(axis=3, 
                                 values=[branch1x1, 
                                         branch3x3, 
                                         branch5x5, 
                                         branch_pool])
    
    return inception_module

def inception_module_v4(input, 
                        in_channels, 
                        filter_count,
                        hidden_filter_count=3,
                        l2_reg_const=0.0, 
                        isTraining=True):
    
    # Branch 1x1 : 1x1 conv
    with tf.variable_scope("1x1_branch"):
        with tf.variable_scope("1x1_conv2d"):
            branch1x1 = conv2d(input, 
                               (1, 1, in_channels, filter_count), 
                               l2_reg_const=l2_reg_const, 
                               isTraining=isTraining)
        
    # Branch 3x3 : 1x1 -> 3x3
    with tf.variable_scope("1x1_3x3_branch"):        
        with tf.variable_scope("1x1_conv2d"):
            branch3x3 = conv2d(input,
                               (1, 1, in_channels, hidden_filter_count),
                               l2_reg_const=l2_reg_const,
                               isTraining=isTraining)
            
        with tf.variable_scope("3x3_conv2d"):
            branch3x3 = conv2d(branch3x3, 
                               (3, 3, hidden_filter_count, filter_count), 
                               l2_reg_const=l2_reg_const,
                               isTraining=isTraining)
    
    # Branch 3x3dbl : 1x1 -> 3x3 -> 3x3
    with tf.variable_scope("1x1_3x3_3x3_branch"):
        with tf.variable_scope("1x1_conv2d"):
            branch3x3dbl = conv2d(input, 
                                  (1, 1, in_channels, hidden_filter_count), 
                                  l2_reg_const=l2_reg_const, 
                                  isTraining=isTraining)
            
        with tf.variable_scope("3x3_conv2d_1"):
            branch3x3dbl = conv2d(branch3x3dbl, 
                                  (3, 3, hidden_filter_count, hidden_filter_count), 
                                  l2_reg_const=l2_reg_const,
                                  isTraining=isTraining)
            
        with tf.variable_scope("3x3_conv2d_2"):
            branch3x3dbl = conv2d(branch3x3dbl, 
                                  (3, 3, hidden_filter_count, filter_count), 
                                  l2_reg_const=l2_reg_const,
                                  isTraining=isTraining)

    # Branch max pool : 3x3 max pooling -> 1x1
    with tf.variable_scope("max_pool_1x1_branch"):
        with tf.variable_scope("max_pool"):
            branch_pool = tf.nn.max_pool(input, 
                                         ksize=[1, 3, 3, 1], 
                                         strides=[1, 1, 1, 1], 
                                         padding='SAME')
            
        with tf.variable_scope("1x1_conv2d"):
            branch_pool = conv2d(branch_pool, 
                                 (1, 1, in_channels, filter_count), 
                                 l2_reg_const=l2_reg_const,
                                 isTraining=isTraining)
    
    inception_module = tf.concat(axis=3, 
                                 values=[branch1x1, 
                                         branch3x3, 
                                         branch3x3dbl, 
                                         branch_pool])
    
    return inception_module
    
def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)