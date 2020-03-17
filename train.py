#coding:utf-8
import numpy as np 
import cv2
import tensorflow as tf 
import os
import model
from sklearn.preprocessing import  OneHotEncoder
import batch_gen 
import lovasz_losses_tf as L


path_img = '/media/raylee/BIGRAYLEE/dataset_aeriel/Dataset/img/'
path_gt = '/media/raylee/BIGRAYLEE/dataset_aeriel/Dataset/gt/'
city = 'austin_256/train/'
result_path_img = path_img+'austin_256/save_train/'
result_label_img = path_img+'austin_256/save_gt'
MODEL_SAVE_PATH = '/media/raylee/BIGRAYLEE/dataset_aeriel/model/austin_256/unet_1/'

img_width = 256
img_height = 256

label_values = ([0,0,0],[255,255,255])

def save_pre_img(path,index,label):
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
	file_path = os.path.join(path,'%s_reslut.png'%index) 
	rgb[:,:,0] = r/1.0
	rgb[:,:,1] = g/1.0
	rgb[:,:,2] = b/1.0
	img = np.uint8(rgb)
	cv2.imwrite(file_path,img)

def one_hot_it(label,label_values):
	semantic_map =[]
	for colour in label_values:
		equality = np.equal(label, colour)
		class_map = np.all(equality, axis = -1)
		#print(class_map.shape)
		semantic_map.append(class_map)
	semantic_map = np.stack(semantic_map,axis=-1)
	
	return semantic_map

def load_data(path_img,path_gt,class_num = 2):
	img_id = []
	for pic in os.listdir(path_img+city):
		img_id.append(pic)
	img_id.sort()
	#print(img_id)
	image = []
	label = []
	for i in range(len(img_id)):
		img = cv2.imread(path_img+city+img_id[i])
		img = img[:,:,0:3]
		img = np.uint8(img)
		img = np.array(img,dtype="float") / 255.0
		#img_tf = tf.convert_to_tensor(img)
		#img_tf = tf.cast(img_tf,dtype = tf.float32)
		#cv2.imwrite(result_path_img+"%d.png"%i, img)
		image.append(img)
		
		
		gt = cv2.imread(path_gt+city+img_id[i])
		gt = gt[:,:,0:3]
		#gt = np.uint8(gt)
		#gt_tf =tf.convert_to_tensor(gt)
		'''gt_tf = tf.one_hot(indices = gt,
								depth = class_num,
								on_value = 1,
								off_value = 0)'''
		gt = one_hot_it(gt,label_values).astype(np.uint8)
		#gt = np.argmax(gt,axis = -1)
		#save_pre_img(result_label_img,i,gt)

		label.append(gt)
		#print(gt)
		#print(i)
	
	image = np.array(image)
	label = np.array(label)
	return image,label

def main():
	sub = 'austin_256/unet_1'
	log_path = '/home/raylee/aerialimage_label/log/'+sub
	model_path = '/media/raylee/BIGRAYLEE/dataset_aeriel/model/'+sub #38000 for austin dataset

	patch_size = 512
	batch_size = 8
	class_num = 2

	max_iteration = 900000 #8 epoches 
	momentum = 0.99
	learning_rate = 1e-4
	os.environ["CUDA_VISIBLE_DEVICES"] = '0'

	#gpu training set
	config = tf.ConfigProto() 
	config.gpu_options.per_process_gpu_memory_fraction = 0.9 
	config.allow_soft_placement = True 
	config.log_device_placement = True 
	config.gpu_options.allocator_type = 'BFC'

	#image, label = load_data(path_img,path_gt)

	#print(image.shape)
	#print(label.shape)


	#x = tf.placeholder(tf.float32,image.shape)#[batch_size, img_width, img_height, 3])
	#y = tf.placeholder(tf.float32,label.shape) #[batch_size, img_width, img_height, class_num])

	x_batch = tf.placeholder(tf.float32, [batch_size, img_width, img_height, 3])
	y_batch = tf.placeholder(tf.float32, [batch_size, img_width, img_height, 2] )

	'''dataset = tf.data.Dataset.from_tensor_slices((x,y))
	dataset = dataset.shuffle(1).batch(batch_size).repeat()
 
	iterator = dataset.make_initializable_iterator()
	data_element = iterator.get_next()'''

	y_ = model.U_Net(x_batch,class_num)
	#Lovasz-softmox
	loss = L.lovasz_hinge(y_, y_batch, ignore=None, per_image=True)
	#cross entropy loss
	#loss = tf.reduce_mean((tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_batch,logits = y_)))
	#test the train res
	#y_1 = tf.nn.softmax(y_)
	#y_1 = tf.argmax(y_1,axis=-1)
	correct_prediction = tf.equal(tf.argmax(y_,axis=-1),tf.argmax(y_batch,axis=-1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	#tf.summary.scalar('entropy_loss',loss)
	tf.summary.scalar('lovasz softmax',loss)
	tf.summary.scalar('accuracy',accuracy)
	train_step = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
	

	sess = tf.Session()
	saver = tf.train.Saver()
	#sess.run(iterator.initializer, feed_dict={x: image,y:label})
	sess.run(tf.global_variables_initializer())
	print('variables initialized')
	#threads = tf.train.start_queue_runners(sess=sess)
	merged = tf.summary.merge_all()
	summary_writer = tf.summary.FileWriter(log_path,sess.graph)
	print('Start Traning...')

	#Breakpoint training
	start_iter = 1
	ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
	if ckpt and ckpt.model_checkpoint_path:
		saver.restore(sess,ckpt.model_checkpoint_path)
		pos = ckpt.model_checkpoint_path
		pos = pos.split('.')[0].split('r')[-1]
		start_iter = int(pos)


	#for epoch in range(0,4):
	#epoch 4
	batch = batch_gen.generator(path_img+city,path_gt+city)
	for itr in range(start_iter+1,max_iteration+1):
		#print('training')
		#small data loading 
		#step_batch, step_label = sess.run(data_element)
		step_batch, step_label = next(batch)
		sess.run(train_step,feed_dict={x_batch:step_batch,y_batch:step_label})
		#res = y_1.eval(session=sess,feed_dict={x_batch:step_batch,y_batch:step_label})
		#save_pre_img(result_path,itr,res[0])
		if itr%10==0:
			summary,train_loss,train_accuracy = sess.run([merged,loss,accuracy],feed_dict={x_batch:step_batch,y_batch:step_label})
			print('iteration %d, loss:%f, acc:%f'%(itr,train_loss,train_accuracy))
			summary_writer.add_summary(summary, itr)
		if itr%9000==0:
			save_path = os.path.join(model_path,'UNet_itr%d.ckpt'%(itr))
			saver.save(sess,save_path)
			print('model parameter has been saved in %s.'%model_path)



if __name__ == '__main__':
	main()
