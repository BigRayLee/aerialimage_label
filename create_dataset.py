#gen the tiles from pictures
import cv2
import random
import os
import numpy as np 
from tqdm import tqdm

path = '/media/raylee/BIGRAYLEE/dataset_aeriel/AerialImageDataset/train/'
sub_img_width = 256
sub_img_height = 256

city = 'vienna/'
def gen_dataset(image_num =14400, mode = 'no_aug'):
	print("begin...")

	img_dataset = []
	label_dataset = []

	for pic in os.listdir(path+'images/'+ city):
		img_dataset.append(pic)

	img_each = image_num/len(img_dataset)
	g_count = 0

	for i in tqdm(range(len(img_dataset))):
		count = 0
		img = cv2.imread(path+'images/'+city+img_dataset[i])
		label = cv2.imread(path+'gt/'+ city+img_dataset[i],cv2.IMREAD_GRAYSCALE)

		WIDTH, HEIGHT, _ = img.shape

		while count < img_each:
			random_width = random.randint(0,WIDTH- sub_img_width -1)
			random_height = random.randint(0,HEIGHT-sub_img_height -1)

			sub_img = img[random_height: random_height+sub_img_height, random_width:random_width+sub_img_width]
			sub_label = label[random_height: random_height+sub_img_height, random_width:random_width+sub_img_width]

			if mode == 'aug':
				pass

			visualize = np.zeros((256,256)).astype(np.uint8)
			visualize = sub_label * 50

			#cv2.imwrite(('/media/raylee/BIGRAYLEE/dataset_aeriel/Dataset/img/austin/visuaize/%d.png'%g_count),visualize)
			cv2.imwrite(('/media/raylee/BIGRAYLEE/dataset_aeriel/Dataset/img/austin_256/train/vienna_%d.png'%g_count),sub_img)
			cv2.imwrite(('/media/raylee/BIGRAYLEE/dataset_aeriel/Dataset/gt/austin_256/train/vienna_%d.png'%g_count),sub_label)

			count = count +1
			g_count = g_count +1





if __name__ == '__main__':
	gen_dataset()
