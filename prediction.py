import numpy as np 
import cv2
import tensorflow as tf 
import os 
import model
import utils
import batch_gen 

#path_img = '/media/raylee/BIGRAYLEE/dataset_aeriel/Dataset/img/'
path_img = '/media/raylee/BIGRAYLEE/dataset_aeriel/AerialImageDataset/test/images/'
path_pred = '/media/raylee/BIGRAYLEE/dataset_aeriel/AerialImageDataset/test/792000'
#city = 'austin_256/validation/'
#city = 'tyrol1/'

model_path = '/media/raylee/BIGRAYLEE/dataset_aeriel/model/austin_256/unet_1'
result_path = path_pred

img_width = 256
img_height = 256
stride = 256
image_size = 256
#save predict results
def save_pre_img(path,name,label):

	building = [255,255,255]
	no_building = [0,0,0]
	r = label.copy()
	g = label.copy()
	b = label.copy()
	label_colours = np.array([no_building,building])
	for l in range(0,2):
		r[label == l] = label_colours[l,0]
		g[label == l] = label_colours[l,1]
		b[label == l] = label_colours[l,2] 
	rgb = np.zeros((label.shape[0],label.shape[1],3))
	file_path = os.path.join(path,'%s.tif'%name) 
	print(file_path)
	rgb[:,:,0] = r/1.0
	rgb[:,:,1] = g/1.0
	rgb[:,:,2] = b/1.0
	img = np.uint8(rgb)
	cv2.imwrite(file_path,img)

#load validation and test img
def load_data(path_img):
	img_id = []
	for pic in os.listdir(path_img):
		img_id.append(pic)
	img_id.sort()
	print(img_id)
	'''image = []
	for i in range(len(img_id)):
		img = cv2.imread(path_img+city+img_id[i])
		img = img[:,:,0:3]
		img = np.uint8(img)
		#img_tf = tf.convert_to_tensor(img)
		#img_tf = tf.cast(img_tf,dtype = tf.float32)
		img = np.array(img,dtype="float") / 255.0
		image.append(img)
	image = np.array(image)'''
	return img_id

def main():
	model_name = 'UNet_itr792000'
	print(result_path)
	model_file = os.path.join(model_path,'%s.ckpt'%model_name)
	class_num = 2
	batch_size = 2

	os.environ['CUDA_VISIBLE_DEVICES'] = '0'

	img_id = load_data(path_img)
	'''print('Loading the test dataset...')
	row = image.shape[1]
	col = image.shape[2]

	sample_num = image.shape[0]'''

	x = tf.placeholder(tf.float32,[1,image_size,image_size,3])
	net = model.U_Net(x,class_num)

	net_softmax = tf.nn.softmax(net)

	#CRF
	x_int = tf.cast(x,dtype=tf.uint8)
	crf = tf.py_func(utils.dense_crf, [net_softmax, x_int], tf.float32)

	thre = 0.8
	result_label = net_softmax > thre
	result_label = tf.cast(result_label,dtype = tf.uint8)
	result = tf.argmax(result_label,axis=-1)
	result = tf.cast(result,dtype = tf.uint8)


	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	print('starting restore parameter...')
	saver.restore(sess,model_file)
	print(model_file)
	print('finishing loading parameter')

	for n in range(len(img_id)):
		img = cv2.imread(path_img+img_id[n])
		img = img[:,:,0:3]
		img = np.uint8(img)
		img = np.array(img,dtype="float") / 255.0
		print('loading No.%d test image...'%n)
		#img = np.reshape(img,[1,row,col,3])
		#clip the oringinal image to 256x256 for the model input
		h,w,d = img.shape
		print(img.shape)
		padding_h = (h//stride + 1) * stride
		padding_w = (w//stride + 1) * stride
		padding_img = np.zeros((padding_h,padding_w,3),dtype = 'float')
		padding_img[0:h,0:w,:] = img[:,:,:]
		#padding_img = np.array(padding_img).astype(np.uint8)
		#padding_img = np.array(padding_img,dtype = 'float')/255.0
		#print(padding_img)
		mask_whole = np.zeros((padding_h,padding_w),dtype = np.uint8)
		for i in range(padding_h//stride):
			for j in range(padding_w//stride):
				crop = padding_img[i*stride:i*stride+image_size,j*stride:j*stride+image_size,0:3]
				ch,cw,_ = crop.shape
				if ch != 256 or cw!=256:
					print('wrong size')
					continue
				#print(crop)
				crop = np.expand_dims(crop,axis = 0)
				sess.run(result,feed_dict={x:crop})
				#print(sess.run(result,feed_dict={x:img}))
				res = result.eval(session=sess,feed_dict={x:crop})
				#softmax = net_softmax.eval(session=sess,feed_dict={x:crop})
				#print(softmax)
				res = res.reshape((256,256)).astype(np.uint8)
				#combine the results
				mask_whole[i*stride:i*stride+image_size,j*stride:j*stride+image_size] = res[:,:]
		print(mask_whole.shape)
		save_pre_img(result_path,img_id[n].split('.')[0],mask_whole)
		print('result saved...')








if __name__ == '__main__':
	main()