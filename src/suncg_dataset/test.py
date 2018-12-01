import tensorflow as tf

point_cloud = tf.zeros([32,2048,3],tf.float32)
is_training = True
bn_decay = 0.1

input_image = tf.expand_dims(point_cloud, -1)
print(input_image)
net = tf.layers.conv2d(input_image, 64, [1,3], padding="VALID",strides=[1,1])
print(net)
net = tf.layers.conv2d(net, 128, [1,1], padding="VALID",strides=[1,1])
print(net)
net = tf.layers.conv2d(net, 1024, [1,1], padding="VALID",strides=[1,1])
print(net)
ksize=(1, 2048, 1, 1)
strides=(1, 2, 2, 1)
net = tf.nn.max_pool(net, ksize=ksize, strides=strides,padding="VALID")
print(net)
net = tf.reshape(net, [32, -1])
print(net)