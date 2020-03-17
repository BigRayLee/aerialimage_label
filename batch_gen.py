#use the data list to generate data
import numpy as np 
import cv2
import os

label_path = '/media/raylee/BIGRAYLEE/dataset_aeriel/Dataset/gt/austin_256/train/'
img_path = '/media/raylee/BIGRAYLEE/dataset_aeriel/Dataset/img/austin_256/train/'
label_values = ([0,0,0],[255,255,255])

def one_hot_it(label,label_values):
	semantic_map =[]
	for colour in label_values:
		equality = np.equal(label, colour)
		class_map = np.all(equality, axis = -1)
		#print(class_map.shape)
		semantic_map.append(class_map)
	semantic_map = np.stack(semantic_map,axis=-1)
	
	return semantic_map

class generator():
	def __init__(self,img_path,gt_path,batch_size = 8,data_aug = True):
		self.base_dir = img_path
		self.char_set = os.listdir(img_path)
		self.num_classes = len(self.char_set)
		self.image_names = []
		self.gt_names = []

		for char_class in self.char_set:
			self.image_names.append(os.path.join(self.base_dir,char_class))
			self.gt_names.append(os.path.join(gt_path,char_class))

		self.image_names.sort()
		self.gt_names.sort()
		print(len(self.image_names))
		#if shuffle
		#np.random.shuffle(self.image_names)
		self.index = 0
		self.count = len(self.image_names)
		self.batch_size = batch_size

	def next(self,batch_size = 8):
		images = []
		labels = []
		while True:
			self.index = self.index + 1
			self.index = self.index%self.count
			
			#if self.index == 0:
				#np.random.shuffle(self.image_names)
			image_path = self.image_names[self.index]
			#print(image_path)
			image = cv2.imread(image_path)

			image = image[:,:,0:3]
			image = np.uint8(image)
			image = np.array(image,dtype="float") / 255.0

			images.append(image)

			label_path = self.gt_names[self.index]
			#print(label_path)
			gt = cv2.imread(label_path)
			gt = gt[:,:,0:3]
		
			gt = one_hot_it(gt,label_values).astype(np.uint8)

			labels.append(gt)

			if(len(images) == batch_size):
				break
		images = np.array(images)
		labels = np.array(labels)
		return images,labels
	
	def __next__(self):
		return self.next(self.batch_size)



if __name__ == '__main__':
	batch_gen = generator(img_path,label_path)
	for i in range(10):
		print(i)
		image,label = next(batch_gen)
		print(image.shape)
		print(label.shape)
	