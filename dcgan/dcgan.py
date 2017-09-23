import numpy as np
import cv2
import tensorflow as tf

# Batch Normalization
def batch_normalization(shape, input, name="bn", withGamma=False):
    with tf.variable_scope(name):
        gamma_init= tf.truncated_normal_initializer(mean=0.0, stddev=0.05)
        gamma = tf.get_variable('gamma', [shape], initializer=gamma_init)
        beta_init= tf.truncated_normal_initializer(mean=0.0, stddev=0.05)
        beta = tf.get_variable('beta', [shape], initializer=beta_init)
        mean, variance = tf.nn.moments(input, [0])
    if withGamma == False:
        return gamma * (input - mean) / tf.sqrt(variance + 1e-5) + beta
    else:
        return gamma * (input - mean) / tf.sqrt(variance + 1e-5) + beta, gamma, beta

def conv2d(input_, output_dim, 
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="conv2d", with_w=False):
  with tf.variable_scope(name):
    w = tf.get_variable('W', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

    biases = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    if with_w:
        return conv, w, biases
    else:
        return conv


def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d", with_w=False):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('W', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))
    
    try:
      deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    # Support for verisons of TensorFlow before 0.7.0
    except AttributeError:
      deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    biases = tf.get_variable('b', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
      return deconv, w, biases
    else:
        return deconv

def lrelu(x, name="lrelu"):
    return tf.maximum(x, x*0.2, name=name)

BATCH_SIZE = 10

class Generator(object):

    def __init__(self, input, in_num, name='generator'):

        # 重み、バイアスを定義
        self.input = input
    
        with tf.variable_scope(name):

            # 全結合層
            weight_init= tf.truncated_normal_initializer(mean=0.0, stddev=0.05)
            self.W_fc1 = tf.get_variable('W_fc1', [in_num, 4*4*2048], initializer=weight_init)
            bias_init = tf.constant_initializer(value=0.0)
            self.b_fc1 = tf.get_variable('b_fc1', [4*4*2048], initializer=bias_init)

            linarg = tf.matmul(input, self.W_fc1) + self.b_fc1
            bn1, gamma_bn1, beta_bn1 = batch_normalization(4*4*2048, linarg, name='bn1', withGamma=True)
            relu_fc1 = tf.nn.relu(bn1)
    
            # 逆畳み込み層1
            relu1_image = tf.reshape(relu_fc1, [BATCH_SIZE, 4, 4, 2048])
            deconv2, self.W_deconv2, self.b_deconv2 = \
                     deconv2d(relu1_image, [BATCH_SIZE, 8, 8, 1024], name='deconv_2', with_w=True)
            gamma_bn2, beta_bn2 = tf.nn.moments(deconv2, [0,1,2], name='bn2')
            h_conv2 = tf.nn.relu(tf.nn.batch_normalization(deconv2, gamma_bn2, beta_bn2, None , None,1e-5,name='bn2'))

            # 逆畳み込み層2
            deconv3, self.W_deconv3, self.b_deconv3 = \
                     deconv2d(h_conv2, [BATCH_SIZE, 16, 16, 512], name='deconv_3', with_w=True)
            # Batch Normalization, ReLu
            gamma_bn3, beta_bn3 = tf.nn.moments(deconv3, [0,1,2], name='bn3')
            h_conv3 = tf.nn.relu(tf.nn.batch_normalization(deconv3, gamma_bn3, beta_bn3, None , None,1e-5,name='bn3'))

            # 逆畳み込み層3
            deconv4, self.W_deconv4, self.b_deconv4 = \
                     deconv2d(h_conv3, [BATCH_SIZE, 32, 32, 256], name='deconv_4', with_w=True)
            # Batch Normalization, ReLu
            gamma_bn4, beta_bn4 = tf.nn.moments(deconv4, [0,1,2], name='bn4')
            h_conv4 = tf.nn.relu(tf.nn.batch_normalization(deconv4, gamma_bn4, beta_bn4, None , None,1e-5,name='bn4'))

            # 逆畳み込み層4
            deconv5, self.W_deconv5, self.b_deconv5 = \
                     deconv2d(h_conv4, [BATCH_SIZE, 64, 64, 3], name='deconv_5', with_w=True)
            self.output = tf.nn.tanh(deconv5)
            
            
    def output(self):
        return self.output

class Discriminator(object):

    def __init__(self, input, reuse=False, name='discriminator'):

        # 重み、バイアスを定義
        self.input = input

        if not reuse:
            with tf.variable_scope(name):

                # 畳み込み1
                input_image = tf.reshape(input, [-1, 64, 64, 3])
                weight_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.05)
                self.W_conv1 = tf.get_variable('W_conv_1', [5, 5, 3, 256], initializer=weight_init)
                bias_init = tf.constant_initializer(value=0.0)
                self.b_conv1 = tf.get_variable('b_conv_1', [256], initializer=bias_init)
                h_conv1 = lrelu(tf.nn.conv2d(input_image, self.W_conv1, strides=[1,2,2,1], padding='SAME') + self.b_conv1, name='lreru_1')

                # 畳み込み2
                self.W_conv2 = tf.get_variable('W_conv_2', [5, 5, 256, 512], initializer=weight_init)
                self.b_conv2 = tf.get_variable('b_conv_2', [512], initializer=bias_init)
                aff_conv2 = tf.nn.conv2d(h_conv1, self.W_conv2, strides=[1,2,2,1], padding='SAME') + self.b_conv2
                # Batch Normalization
                gamma_bn2, beta_bn2 = tf.nn.moments(aff_conv2, [0,1,2], name='bn2')
                h_conv2 = lrelu(tf.nn.batch_normalization(aff_conv2, gamma_bn2, beta_bn2, None , None,1e-5,name='bn2'),  name='lreru_2')

                # 畳み込み3
                self.W_conv3 = tf.get_variable('W_conv_3', [5, 5, 512, 1024], initializer=weight_init)
                self.b_conv3 = tf.get_variable('b_conv_3', [1024], initializer=bias_init)
                aff_conv3 = tf.nn.conv2d(h_conv2, self.W_conv3, strides=[1,2,2,1], padding='SAME') + self.b_conv3
                # Batch Normalization
                gamma_bn3, beta_bn3 = tf.nn.moments(aff_conv3, [0,1,2], name='bn3')
                h_conv3 = lrelu(tf.nn.batch_normalization(aff_conv3, gamma_bn3, beta_bn3, None , None,1e-5,name='bn3'),  name='lreru_3')

                # 畳み込み4
                self.W_conv4 = tf.get_variable('W_conv_4', [5, 5, 1024, 2048], initializer=weight_init)
                self.b_conv4 = tf.get_variable('b_conv_4', [2048], initializer=bias_init)
                aff_conv4 = tf.nn.conv2d(h_conv3, self.W_conv4, strides=[1,2,2,1], padding='SAME') + self.b_conv4
                # Batch Normalization
                gamma_bn4, beta_bn4 = tf.nn.moments(aff_conv4, [0,1,2], name='bn4')
                h_conv4 = lrelu(tf.nn.batch_normalization(aff_conv4, gamma_bn4, beta_bn4, None , None,1e-5,name='bn4'),  name='lreru_4')

                # 全結合層
                self.W_fc5 = tf.get_variable('W_fc5', [4*4*2048, 1], initializer=weight_init)
                self.b_fc5 = tf.get_variable('b_fc5', [1], initializer=bias_init)
                h_conv4_flat = tf.reshape(h_conv4, [-1, 4*4*2048])
                linarg = tf.matmul(h_conv4_flat, self.W_fc5) + self.b_fc5
                
                h5 = tf.nn.sigmoid(linarg)
                
        else:   # 重み共有
            with tf.variable_scope(name, reuse=True):

                # 畳み込み1
                input_image = tf.reshape(input, [-1, 64, 64, 3])
                self.W_conv1 = tf.get_variable('W_conv_1', [5, 5, 3, 256])
                self.b_conv1 = tf.get_variable('b_conv_1', [256])
                h_conv1 = lrelu(tf.nn.conv2d(input_image, self.W_conv1, strides=[1,2,2,1], padding='SAME') + self.b_conv1, name='lreru_1')

                # 畳み込み2
                self.W_conv2 = tf.get_variable('W_conv_2', [5, 5, 256, 512])
                self.b_conv2 = tf.get_variable('b_conv_2', [512])
                aff_conv2 = tf.nn.conv2d(h_conv1, self.W_conv2, strides=[1,2,2,1], padding='SAME') + self.b_conv2
                # Batch Normalization
                gamma_bn2, beta_bn2 = tf.nn.moments(aff_conv2, [0,1,2], name='bn2')
                h_conv2 = lrelu(tf.nn.batch_normalization(aff_conv2, gamma_bn2, beta_bn2, None , None,1e-5,name='bn2'),  name='lreru_2')

                # 畳み込み3
                self.W_conv3 = tf.get_variable('W_conv_3', [5, 5, 512, 1024])
                self.b_conv3 = tf.get_variable('b_conv_3', [1024])
                aff_conv3 = tf.nn.conv2d(h_conv2, self.W_conv3, strides=[1,2,2,1], padding='SAME') + self.b_conv3
                # Batch Normalization
                gamma_bn3, beta_bn3 = tf.nn.moments(aff_conv3, [0,1,2], name='bn3')
                h_conv3 = lrelu(tf.nn.batch_normalization(aff_conv3, gamma_bn3, beta_bn3, None , None,1e-5,name='bn3'),  name='lreru_3')

                # 畳み込み4
                self.W_conv4 = tf.get_variable('W_conv_4', [5, 5, 1024, 2048])
                self.b_conv4 = tf.get_variable('b_conv_4', [2048])
                aff_conv4 = tf.nn.conv2d(h_conv3, self.W_conv4, strides=[1,2,2,1], padding='SAME') + self.b_conv4
                # Batch Normalization
                gamma_bn4, beta_bn4 = tf.nn.moments(aff_conv4, [0,1,2], name='bn4')
                h_conv4 = lrelu(tf.nn.batch_normalization(aff_conv4, gamma_bn4, beta_bn4, None , None,1e-5,name='bn4'),  name='lreru_4')

                # 全結合層
                self.W_fc5 = tf.get_variable('W_fc5', [4*4*2048, 1])
                self.b_fc5 = tf.get_variable('b_fc5', [1])
                h_conv4_flat = tf.reshape(h_conv4, [-1, 4*4*2048])
                linarg = tf.matmul(h_conv4_flat, self.W_fc5) + self.b_fc5
                
                h5 = tf.nn.sigmoid(linarg)
                
        self.output = h5
        self.output_aff = linarg

    def output(self):
        return self.output, self.output_aff


z = tf.placeholder("float", [None, 100])
images = tf.placeholder("float", [None, 64*64*3])

def make_model(z, images):
    g = Generator(z, 100)
    fake_img = Generator.output(g)
    d_fake = Discriminator(fake_img)                # 偽物用
    d_fake_out, d_logits_f = Discriminator.output(d_fake)
    d_true = Discriminator(images, reuse=True)      # 本物用(重み共有)
    d_true_out, d_logits_t = Discriminator.output(d_true)

    # 損失関数
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_t, labels= tf.ones_like(d_true_out)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_f, labels= tf.zeros_like(d_fake_out)))
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_f, labels= tf.ones_like(d_fake_out)))
                          
    d_loss = d_loss_real + d_loss_fake

    # 変数初期化
    t_vars = tf.trainable_variables()

    d_vars = [var for var in t_vars if 'generator' in var.name]
    g_vars = [var for var in t_vars if 'discriminator' in var.name]

    return d_loss, g_loss, d_vars, g_vars

# モデル作成
d_loss, g_loss, d_vars, g_vars = make_model(z, images)

learning_rate = 0.0001

d_optim = tf.train.AdamOptimizer(learning_rate) \
              .minimize(d_loss, var_list=d_vars)
g_optim = tf.train.AdamOptimizer(learning_rate) \
.minimize(g_loss, var_list=g_vars)

init = tf.global_variables_initializer()

"""    
#with tf.variable_scope(name, reuse=True):
#    self.W_fc2 = tf.get_variable('W_fc1', [in_num, 4*4*2048])

 第1レイヤー
5,5はフィルタサイズ、1は入力チャンネル数、32はフィルタ数
  
# 全結合層1
W_fc1 = weight_variable([100, 4*4*2048])
b_fc1 = bias_variable([4*4*2048])
bn1 = batch_normalization(4*4*2048, tf.matmul(z, W_fc1) + b_fc1)
relu_fc1 = tf.nn.relu(bn1)

# 逆畳み込み層1
relu1_image = tf.reshape(relu_fc1, [BATCH_SIZE, 4, 4, 2048])
c_conv2 = deconv2d(relu1_image, [BATCH_SIZE, 8, 8, 1024],name='deconv2')
# Batch Normalization, ReLu
mean2, var2 = tf.nn.moments(c_conv2, [0,1,2])
h_conv2 = tf.nn.relu(tf.nn.batch_normalization(c_conv2, mean2, var2, None , None,1e-5,name='bn2'))

#print((h_conv2))

# 逆畳み込み層2
c_conv3 = deconv2d(h_conv2, [BATCH_SIZE, 16, 16, 512],name='deconv3')
# Batch Normalization, ReLu
mean3, var3 = tf.nn.moments(c_conv3, [0,1,2])
h_conv3 = tf.nn.relu(tf.nn.batch_normalization(c_conv3, mean3, var3, None , None,1e-5,name='bn3'))

#print((h_conv3))

# 逆畳み込み層3
c_conv4 = deconv2d(h_conv3, [BATCH_SIZE, 32, 32, 256],name='deconv4')
# Batch Normalization, ReLu
mean4, var4 = tf.nn.moments(c_conv4, [0,1,2])
h_conv4 = tf.nn.relu(tf.nn.batch_normalization(c_conv4, mean4, var4, None , None,1e-5,name='bn4'))

#print((h_conv4))

# 逆畳み込み層4
c_conv5 = deconv2d(h_conv4, [BATCH_SIZE, 64, 64, 3],name='deconv5')
h_conv5 = tf.nn.tanh(c_conv5)
return h_conv5
    

def discriminator(image_f, image_t):  # image is 64x64x3

    W_conv1 = weight_variable([5, 5, 3, 256])
    b_conv1 = bias_variable([256])

    W_conv2 = weight_variable([5, 5, 256, 512])
    b_conv2 = bias_variable([512])

    W_conv3 = weight_variable([5, 5, 512, 1024])
    b_conv3 = bias_variable([1024])
    
    W_conv4 = weight_variable([5, 5, 1024, 2048])
    b_conv4 = bias_variable([2048])

    # 畳み込み1
    h_fake1 = tf.nn.conv2d(image_f, W_conv1, strides=[1,2,2,1], padding='SAME') + b_conv1
    h_true1 = tf.nn.conv2d(image_t, W_conv1, strides=[1,2,2,1], padding='SAME') + b_conv1

    # 畳み込み2
    h_fake2 = tf.nn.conv2d(h_fake1, W_conv2, strides=[1,2,2,1], padding='SAME') + b_conv2
    h_true2 = tf.nn.conv2d(h_true1, W_conv2, strides=[1,2,2,1], padding='SAME') + b_conv2

    # 畳み込み3
    h_fake3 = tf.nn.conv2d(h_fake2, W_conv3, strides=[1,2,2,1], padding='SAME') + b_conv3
    h_true3 = tf.nn.conv2d(h_true2, W_conv3, strides=[1,2,2,1], padding='SAME') + b_conv3

    # 畳み込み4
    h_fake4 = tf.nn.conv2d(h_fake3, W_conv4, strides=[1,2,2,1], padding='SAME') + b_conv4
    h_true4 = tf.nn.conv2d(h_true3, W_conv4, strides=[1,2,2,1], padding='SAME') + b_conv4

    print(h_fake4)
    
    # 全結合層
    W_fc1 = weight_variable([4*4*2048, 2])
    b_fc1 = bias_variable([2])
    h_fake4_flat = tf.reshape(h_fake4, [-1, 4*4*2048])
    h_fake5 = tf.nn.sigmoid(tf.matmul(h_fake4_flat, W_fc1) + b_fc1)
    h_true4_flat = tf.reshape(h_true4, [-1, 4*4*2048])
    h_true5 = tf.nn.sigmoid(tf.matmul(h_true4_flat, W_fc1) + b_fc1)
    
    return tf.nn.sigmoid(h_fake5), tf.nn.sigmoid(h_true5)

# 入力
z = tf.placeholder(tf.float32, [None, 100])
image = tf.placeholder(tf.float32, [None, 64*64*3])
image = tf.reshape(image, [-1, 64, 64, 3])

# G
image_fake = generator(z)



# D (0:fake, 1:true)
res_fake, res_true = discriminator(image_fake, image)

print(res_fake)

# Gにとっては0(本物)を出力させたい
loss_g = -tf.reduce_sum([1,0] * tf.log(res_fake))

# Dにとっては1(偽物)を出力させたい
loss_d = -tf.reduce_sum([0,1] * tf.log(res_fake))

# Dにとっては0(本物)を出力させたい
loss_d += -tf.reduce_sum([1,0] * tf.log(res_true))

# 損失とオプティマイザーを定義
g_optim = tf.train.AdamOptimizer(0.0001).minimize(loss_g)
d_optim = tf.train.AdamOptimizer(0.0001).minimize(loss_d)

# イニシャライズ
tf.global_variables_initializer().run()

"""

