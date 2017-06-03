import tensorflow as tf
from ops import fc, conv2d

def convNet(input, keep_prob, l2_reg_const, isTraining):
    
    # 32x32x1 -> 32x32x7
    with tf.variable_scope("conv2d_1"):
        with tf.variable_scope("conv2d_1"):
            conv2d_1 = conv2d(input, 
                              filter_shape=(3, 3, 1, 3),
                              strides=[1, 1, 1, 1],
                              padding='SAME',
                              l2_reg_const=l2_reg_const, 
                              isTraining=isTraining)
            
        with tf.variable_scope("conv2d_2"):
            conv2d_1 = conv2d(conv2d_1, 
                              filter_shape=(3, 3, 3, 6),
                              strides=[1, 1, 1, 1],
                              padding='SAME',
                              l2_reg_const=l2_reg_const, 
                              isTraining=isTraining)
        
        conv2d_1 = tf.concat(axis=3, values=[conv2d_1, input])
        
    # 32x32x7 -> 16x16x7
    with tf.variable_scope("max_pool_1"):    
        max_pool_1 = tf.nn.max_pool(conv2d_1,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME')
        #max_pool_1 = tf.nn.dropout(max_pool_1, 0.3)
        
    # 16x16x7 -> 16x16x[16+7]23
    with tf.variable_scope("conv2d_2"):
        with tf.variable_scope("conv2d_1"):
            conv2d_2 = conv2d(max_pool_1, 
                              filter_shape=(4, 4, 7, 8),
                              strides=[1, 1, 1, 1],
                              padding='SAME',
                              l2_reg_const=l2_reg_const, 
                              isTraining=isTraining)
        
        with tf.variable_scope("conv2d_2"):
            conv2d_2 = conv2d(conv2d_2, 
                              filter_shape=(4, 4, 8, 16),
                              strides=[1, 1, 1, 1],
                              padding='SAME',
                              l2_reg_const=l2_reg_const, 
                              isTraining=isTraining)
        
        conv2d_2 = tf.concat(axis=3, values=[conv2d_2, max_pool_1])
        
    # 16x16x23 -> 8x8x23
    with tf.variable_scope("max_pool_2"):    
        max_pool_2 = tf.nn.max_pool(conv2d_2,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME')
        #max_pool_2 = tf.nn.dropout(max_pool_2, 0.3)
        
    # 8x8x23 -> 8x8x[32+23]
    with tf.variable_scope("conv2d_3"):
        conv2d_3 = conv2d(max_pool_2, 
                          filter_shape=(3, 3, 23, 32),
                          strides=[1, 1, 1, 1],
                          padding='SAME',
                          l2_reg_const=l2_reg_const, 
                          isTraining=isTraining)
                          
        conv2d_3 = tf.concat(axis=3, values=[conv2d_3, max_pool_2])
        
    # 8x8x55 -> 4x4x55
    with tf.variable_scope("max_pool_3"):    
        max_pool_3 = tf.nn.max_pool(conv2d_3,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME')
        #max_pool_3 = tf.nn.dropout(max_pool_3, 0.3)
    
    
    # 3x3x55 -> 1x1x55
    with tf.variable_scope("conv2d_4"):
        conv2d_4 = conv2d(max_pool_3, 
                          filter_shape=(3, 3, 55, 128),
                          strides=[1, 1, 1, 1],
                          padding='VALID',
                          l2_reg_const=l2_reg_const, 
                          isTraining=isTraining)
        
    # Flatten. 14x14x4=784
    with tf.variable_scope("fc0"):
        fc0 = tf.contrib.layers.flatten(conv2d_4)
    
    # Fully Connected. Input = 784. Output = 384.
    with tf.variable_scope("fc1"):
        fc1 = fc(fc0, 512, 1024, 
                 l2_reg_const=l2_reg_const, 
                 isTraining=isTraining)
        
        fc1 = tf.nn.dropout(fc1, keep_prob)
        
    # Fully Connected. Input = 128. Output = output_n_classes.
    with tf.variable_scope("fc2"):
        logits = fc(fc1, 1024, 43, 
                    activation=None, 
                    l2_reg_const=l2_reg_const, 
                    isTraining=isTraining)
        
        logits = tf.nn.dropout(logits, keep_prob)

    
    return logits

def InceptionV3(input, keep_prob, l2_reg_const, isTraining):
    
    # 32x32x3 -> 32x32x1
    with tf.variable_scope("conv2d_1"):
        conv2d_1 = conv2d(input, 
                          filter_shape=(1, 1, 3, 1),
                          strides=[1, 1, 1, 1],
                          l2_reg_const=l2_reg_const, 
                          isTraining=isTraining)
    
    # 32x32x1 -> 32x32x64
    with tf.variable_scope("inception_module_1"):
        inception_block1 = inception_module_v4(conv2d_1, 1, 16, 
                                               hidden_filter_count=8, 
                                               l2_reg_const=l2_reg_const,
                                               isTraining=isTraining)
        
    
    # 32x32x64 -> 32x32x64 -> 32x32x[64+64+1]
    with tf.variable_scope("inception_module_2"):
        inception_block2 = inception_module_v4(inception_block1, 64, 16, 
                                               hidden_filter_count=8, 
                                               l2_reg_const=l2_reg_const,
                                               isTraining=isTraining)
    
    with tf.variable_scope("concat"):
        inception_block2 = tf.concat(axis=3, values=[inception_block1, inception_block2, conv2d_1])
        
    # 32x32x[64+64+1]129 -> 32x32x4
    with tf.variable_scope("inception_module_3"): #144
        inception_block3 = inception_module_v4(inception_block2, 129, 1, 
                                               hidden_filter_count=8,  
                                               l2_reg_const=l2_reg_const,
                                               isTraining=isTraining)
        
    # 32x32x4 -> 
    with tf.variable_scope("max_pool_1"):    
        max_pool_1 = tf.nn.max_pool(inception_block3,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME')
    # Flatten. 14x14x4=784
    with tf.variable_scope("fc0"):
        fc0 = tf.contrib.layers.flatten(max_pool_1)
    
    # Fully Connected. Input = 784. Output = 384.
    with tf.variable_scope("fc1"):
        fc1 = fc(fc0, 1024, 256, 
                 l2_reg_const=l2_reg_const, 
                 isTraining=isTraining)
        
        fc1 = tf.nn.dropout(fc1, keep_prob)
    
#     # Fully Connected. Input = 384. Output = 128.
#     with tf.variable_scope("fc2"):
#         fc2 = fc(fc1, 256, 128, 
#                  l2_reg_const=l2_reg_const, 
#                  isTraining=isTraining)
        
#         fc2 = tf.nn.dropout(fc2, keep_prob)
    
#     # Fully Connected. Input = 128. Output = output_n_classes.
#     with tf.variable_scope("fc3"):
#         logits = fc(fc2, 128, 43, 
#                     activation=None, 
#                     l2_reg_const=l2_reg_const, 
#                     isTraining=isTraining)
        
    # Fully Connected. Input = 128. Output = output_n_classes.
    with tf.variable_scope("fc2"):
        logits = fc(fc1, 256, 43, 
                    activation=None, 
                    l2_reg_const=l2_reg_const, 
                    isTraining=isTraining)
        
        logits = tf.nn.dropout(logits, keep_prob)
        
    
    # Dropout
#     with tf.variable_scope("dropout"):
#         logits = tf.nn.dropout(logits, keep_prob)
    
    return logits
   


def LeNetWithDropout(input, 
                     keep_prob, 
                     l2_reg_const, 
                     isTraining=True):
    
    #with tf.variable_scope("channel_compreesion"):
    #    compress = conv2d(input, 
    #                      filter_shape=(1, 1, 3, 1),
    #                      strides=[1, 1, 1, 1],
    #                      padding='SAME')
    # 32x32x3 -> 28x28x6
    with tf.variable_scope("conv2d_1"):
        conv2d_1 = conv2d(input, 
                          filter_shape=(5, 5, 1, 6),
                          strides=[1, 1, 1, 1],
                          padding='VALID',
                          isTraining=isTraining)
    
    # 28x28x6 -> 14x14x6
    with tf.variable_scope("max_pool_1"):    
        max_pool_1 = tf.nn.max_pool(conv2d_1,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='VALID')
    
    # 14x14x6 -> 10x10x16
    with tf.variable_scope("conv2d_2"):
        conv2d_2 = conv2d(max_pool_1, 
                          filter_shape=(5, 5, 6, 16),
                          strides=[1, 1, 1, 1],
                          padding='VALID',
                          isTraining=isTraining)
        
    # 10x10x16 -> 5x5x16    
    with tf.variable_scope("max_pool_2"):        
        max_pool_2 = tf.nn.max_pool(conv2d_2,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='VALID')  
        
    # 5x5x16 -> 400
    with tf.variable_scope("fc0"):
        fc0 = tf.contrib.layers.flatten(max_pool_2)
        
    # Fully Connected. Input = 400. Output = 120.
    with tf.variable_scope("fc1"):
        fc1 = fc(fc0, 400, 120, 
                 l2_reg_const=l2_reg_const, 
                 isTraining=isTraining)
        
        #fc1 = tf.nn.dropout(fc1, keep_prob)
    
    # Fully Connected. Input = 120. Output = 84.
    with tf.variable_scope("fc2"):
        fc2 = fc(fc1, 120, 84,
                 l2_reg_const=l2_reg_const,
                 isTraining=isTraining)
        
        #fc2 = tf.nn.dropout(fc2, keep_prob)
    
    # Fully Connected. Input = 128. Output = output_n_classes.
    with tf.variable_scope("fc3"):
        logits = fc(fc2, 84, 43, 
                    activation=None, 
                    l2_reg_const=l2_reg_const,
                    isTraining=isTraining)
    
        logits = tf.nn.dropout(logits, keep_prob)
        
    return logits
        
def LeNetWithDropout2(input, 
                      keep_prob, 
                      l2_reg_const, 
                      isTraining):
    mean = 0.0
    stddev = 0.1
    
    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mean, stddev = stddev))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(input, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)
    
    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # Layer 2: Convolutional. Input = 14x14x6. Output = 10x10x16
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mean, stddev = stddev))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # Activation.
    conv2 = tf.nn.relu(conv2)
    
    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # Flatten. Input = 5x5x16. Output = 400.
    fc0 = tf.contrib.layers.flatten(conv2)
    
    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mean, stddev = stddev))
    fc1_b = tf.Variable(tf.zeros(120))
    
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b
    
    # Activation.
    fc1 = tf.nn.relu(fc1)
    
    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mean, stddev = stddev))
    fc2_b = tf.Variable(tf.zeros(84))
    
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b
    
    # Activation.
    fc2 = tf.nn.relu(fc2)
    
    # Layer 5: Fully Connected. Input = 84. Output = output_n_classes.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mean, stddev = stddev))
    fc3_b = tf.Variable(tf.zeros(43))
    
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    # Dropout
    
    logits = tf.nn.dropout(logits, keep_prob)
    
    return logits

    

def LeNet(input, 
          keep_prob, 
          l2_reg_const, 
          isTraining):
    
    mean = 0.0
    stddev = 0.1
    
    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mean, stddev = stddev))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(input, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)
    
    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # Layer 2: Convolutional. Input = 14x14x6. Output = 10x10x16
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mean, stddev = stddev))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # Activation.
    conv2 = tf.nn.relu(conv2)
    
    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # Flatten. Input = 5x5x16. Output = 400.
    fc0 = tf.contrib.layers.flatten(conv2)
    
    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mean, stddev = stddev))
    fc1_b = tf.Variable(tf.zeros(120))
    
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b
    
    # Activation.
    fc1 = tf.nn.relu(fc1)
    
    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mean, stddev = stddev))
    fc2_b = tf.Variable(tf.zeros(84))
    
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b
    
    # Activation.
    fc2 = tf.nn.relu(fc2)
    
    # Layer 5: Fully Connected. Input = 84. Output = output_n_classes.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mean, stddev = stddev))
    fc3_b = tf.Variable(tf.zeros(43))
    
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits
