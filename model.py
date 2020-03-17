import tensorflow as tf
import numpy as np 

input_width = 256
input_height = 256

test_input_width = 5000
test_input_height = 5000
#basic function
def weight_variable(shape, name):
	initial = tf.truncated_normal(shape,stddev = 0.1)
	return tf.Variable(initial_value = initial, name = name)

def bias_variable(shape, name):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial_value = initial, name = name)

def conv2d(x,w,s=1):
	return tf.nn.conv2d(x, w , strides = [1, s , s, 1], padding = 'SAME')

def deconv2d(x,w):
	return tf.nn.conv2d_transpose(x, w, strides = [1,1,1,1], padding = 'SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

def batch(x):
	mean,variance = tf.nn.moments(x,[0,1,2,3])
	return tf.nn.batch_normalization(x,
									 mean = mean,
									 variance = variance,
									 offset = 0,
									 scale = 1,
									 variance_epsilon = 1e-6)

def conv_layer(x, input_channel, output_channel, k_size = 3, relu = True, stride = 1,bn = True, name = 'conv_layer'):
	with tf.name_scope(name):
		w = weight_variable([k_size, k_size, input_channel, output_channel], 'weight')
		b = bias_variable([output_channel],'bias')
		answer = conv2d(x,w,s=stride) + b
		if bn:
			answer = batch(answer)
		if relu:
			answer = tf.nn.relu(answer)
		return answer

def res_conv_layer(x,input_channel,output_channel,relu = True,stride = 1,name='res_conv_layer'):
	with tf.name_scope(name):
		if input_channel == output_channel and stride == 1:
			conv1 = conv_layer(x,input_channel,output_channel,name = 'conv1')
			conv2 = conv_layer(conv1,output_channel,output_channel,name = 'conv2')
			conv3 = conv_layer(conv2,output_channel,output_channel,name = 'conv3')
			answer = conv3 + x
			if relu:
				return tf.nn.relu(answer)
			else:
				return answer
		else:
			conv1 = conv_layer(x,input_channel,output_channel,name='conv1',stride=stride)
			conv2 = conv_layer(conv1,output_channel,output_channel,name='conv2')
			conv3 = conv_layer(conv2,output_channel,output_channel,name='conv3',relu=False)
			conv1_ = conv_layer(x,input_channel,output_channel,name='conv1_',relu=False,stride=stride)
			answer = conv1_+conv3
			if relu:
				return tf.nn.relu(answer)
			else:
				return answer

def SegNet(x, class_num, has_skip = False):
	#x:[batch,row,col,channel]
	input_row = x.shape[1]
	input_col = x.shape[2]
	input_channel = x.shape[3]
	if [input_row,input_col,input_channel] != [input_width,input_height,3]:
		print('Seget Input error...')
		return
	norm = batch(x)
	#Encoder
	net = conv_layer(norm,3,64,name = 'conv1_1')
	net = conv_layer(net,64,64,name = 'conv1_2')
	net = max_pool_2x2(net)
	skip_1 = net

	net = conv_layer(net,64,128,name = 'conv2_1')
	net = conv_layer(net,128,128,name = 'conv2_2')
	net = max_pool_2x2(net)
	skip_2 = net

	net = conv_layer(net,128,256,name = 'conv3_1')
	net = conv_layer(net,256,256,name = 'conv3_2')
	net = conv_layer(net,256,256,name = 'conv3_3')
	net = max_pool_2x2(net)
	skip_3 = net

	net = conv_layer(net,256,512,name = 'conv4_1')
	net = conv_layer(net,512,512,name = 'conv4_2')
	net = conv_layer(net,512,512,name = 'conv4_3')
	net = max_pool_2x2(net)
	skip_4 = net

	net = conv_layer(net,512,512,name = 'conv5_1')
	net = conv_layer(net,512,512,name = 'conv5_2')
	net = conv_layer(net,512,512,name = 'conv5_3')
	net = max_pool_2x2(net)

	#Decoder
	net = tf.image.resize_bilinear(net,[input_width//16,input_width//16],name='upsample_1')
	net = conv_layer(net,512,512,name = 'conv6_1') 
	net = conv_layer(net,512,512,name = 'conv6_2')
	net = conv_layer(net,512,512,name = 'conv6_3')
	if has_skip:
		net = tf.add(net,skip_4)

	net = tf.image.resize_bilinear(net,[input_width//8,input_width//8],name = 'upsample_2')
	net = conv_layer(net,512,256,name = 'conv7_1')
	net = conv_layer(net,256,256,name = 'conv7_2')
	net = conv_layer(net,256,256,name = 'conv7_3')
	if has_skip:
		net = tf.add(net,skip_3)

	net = tf.image.resize_bilinear(net,[input_width//4,input_width//4],name = 'upsample_3')
	net = conv_layer(net,256,128,name = 'conv8_1')
	net = conv_layer(net,128,128,name = 'conv8_2')
	net = conv_layer(net,128,128,name = 'conv8_3')
	if has_skip:
		net = tf.add(net,skip_2)

	net = tf.image.resize_bilinear(net,[input_width//2,input_width//2],name = 'upsample_4')
	net = conv_layer(net,128,64)
	net = conv_layer(net,64,64)
	if has_skip:
		net = tf.add(net.skip_1)

	net = tf.image.resize_bilinear(net,[input_width,input_width],name = 'upsample_5')
	net = conv_layer(net,64,64)
	net = conv_layer(net,64,class_num,relu = False,bn=False)

	return net

#Unet-Resnet
def Unet_ResNet(x,class_num):
	input_row = x.shape[1]
	input_col = x.shape[2]
	input_channel = x.shape[3]
	if [input_row,input_col,input_channel] != [input_width,input_height,3]:
		print('Unet_ResNet Input errors...')
		return
	net_res_1 = res_conv_layer(x,3,64,name = 'res_conv_1',relu = True, stride = 1)
	net_res_2 = res_conv_layer(net_res_1,64,128,name = 'res_conv_2',relu = True,stride = 2)
	net_res_3 = res_conv_layer(net_res_2,128,256,name = 'res_conv_3',relu = True,stride = 2)
	net_res_4 = res_conv_layer(net_res_3,256,512,name = 'res_conv_4',relu = True,stride = 2)
	net_res_5 = res_conv_layer(net_res_4,512,512,name = 'res_conv_5',relu = True,stride = 2)

	net_up6 = tf.image.resize_bilinear(net_res_5,[input_width//4,input_height//4],name = 'upsample1')
	net_res_conv3_cut = res_conv_layer(net_res_3,256,512,name = 'res_conv_3_cut',relu = True,stride = 1)
	net_fp6 = net_up6 + net_res_conv3_cut
	net_res_6 = res_conv_layer(net_fp6,512,512,name = 'res_conv_6',relu = True, stride = 1)

	net_up7 = tf.image.resize_bilinear(net_res_6,[input_width//2,input_height//2],name = 'upsample2')
	net_res_conv2_cut = res_conv_layer(net_res_2,128,512,name= 'res_conv_2_cut',relu = True,stride = 1)
	net_fp7 = net_up7 + net_res_conv2_cut
	net_res_7 = res_conv_layer(net_fp7,512,512,name = 'res_conv_6',relu = True, stride = 1)

	net_up8 = tf.image.resize_bilinear(net_res_7,[input_width,input_height],name = 'upsample3')
	net_res_conv1_cut = res_conv_layer(net_res_1,64,512,name= 'res_conv_1_cut',relu = True,stride = 1)
	net_fp8 = net_up8 + net_res_conv1_cut
	net_res_8 = res_conv_layer(net_fp8,512,512,name = 'res_conv_8',relu = True, stride = 1)
	net_fc = conv_layer(net_res_8,512,class_num,k_size=1,name='fc',relu=False,bn=False,stride=1)

	return net_fc

def U_Net(x, class_num):
	input_row = x.shape[1]
	input_col = x.shape[2]
	input_channel = x.shape[3]
	if [input_row,input_col,input_channel] != [input_width,input_height,3]:
		print('Unet_ResNet Input errors...')
		return
	#norm=batchnorm(x)
	net_conv1 = conv_layer(x,3,64,name='conv1')	#256x256
	net_conv2 = conv_layer(net_conv1,64,64,name='conv2')
	net_pool1 = max_pool_2x2(net_conv2)

	net_conv3 = conv_layer(net_pool1,64,128,name='conv3')	#128x128
	net_conv4 = conv_layer(net_conv3,128,128,name='conv4')
	net_pool2 = max_pool_2x2(net_conv4)

	net_conv5 = conv_layer(net_pool2,128,256,name='conv5')	#64x64
	net_conv6 = conv_layer(net_conv5,256,256,name='conv6')
	net_pool3 = max_pool_2x2(net_conv6)

	net_conv7 = conv_layer(net_pool3,256,512,name='conv7')	#32x32
	net_conv8 = conv_layer(net_conv7,512,512,name='conv8')

	net_conv9 = conv_layer(net_conv8,512,256,name='conv9')	
	net_up1 = tf.image.resize_bilinear(net_conv9, [input_width//4,input_height//4],name='upsample1')	#64x64
	net_concat1 = tf.concat([net_up1,net_conv6],axis=-1,name='concat1')
	net_conv10 = conv_layer(net_concat1,512,256,name='conv10')
	net_conv11 = conv_layer(net_conv10,256,256,name='conv11')

	net_conv12 = conv_layer(net_conv11,256,128,name='conv12')
	net_up2 = tf.image.resize_bilinear(net_conv12,[input_width//2,input_height//2],name='upsample2')	#128x128
	net_concat2 = tf.concat([net_up2,net_conv4],axis=-1,name='concat2')
	net_conv13 = conv_layer(net_concat2,256,128,name='conv13')
	net_conv14 = conv_layer(net_conv13,128,128,name='conv14')

	net_conv15 = conv_layer(net_conv14,128,64,name='conv15')
	net_up3 = tf.image.resize_bilinear(net_conv15,[input_width,input_height],name='upsample3')	#256x256
	net_concat3 = tf.concat([net_up3,net_conv2],axis=-1,name='concat3')
	net_conv16 = conv_layer(net_concat3,128,64,name='conv16')
	net_conv17 = conv_layer(net_conv16,64,64,name='conv17')

	net_conv18 = conv_layer(net_conv17,64,class_num,k_size=1,name='conv18',relu=False)

	return net_conv18
