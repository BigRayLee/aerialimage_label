import numpy as np
import cv2
import os

RESULT = ['/media/raylee/BIGRAYLEE/dataset_aeriel/AerialImageDataset/test/clip_697500/','/media/raylee/BIGRAYLEE/dataset_aeriel/AerialImageDataset/test/clip_720000/']#,'/media/raylee/BIGRAYLEE/dataset_aeriel/AerialImageDataset/test/clip_765000/']
VOTE_RESULT = ['/media/raylee/BIGRAYLEE/dataset_aeriel/AerialImageDataset/test/vote_res/']
label_values = ([0,0,0],[255,255,255])

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

# each result has 2 classes : building, no-building
def load_dir(path):
	img_list = []
	for pic in os.listdir(path):
		img_list.append(pic)
	return img_list

def vote_per_result(img_id):
	result_list = []
	for j in range(len(RESULT)):
		print(RESULT[j]+str(img_id))
		img = cv2.imread(RESULT[j]+str(img_id),0)
		'''img = one_hot_it(img,label_values).astype(np.uint8)
		img = np.argmax(img)
		img = np.cast(img).astype(np.uint8)
		print(img)'''
		img[img==255] = 1
		print(img.shape)
		result_list.append(img)

	#vote for each pixel
	height,width = result_list[0].shape
	vote_mask = np.zeros((height,width))
	for h in range(height):
		for w in range(width):
			record = np.zeros((1,2))
			for n in range(len(result_list)):
				mask = result_list[n]
				pixel = mask[h,w]
				#print(pixel)
				record[0,pixel] = record[0,pixel] + 1
			
			label = record.argmax()
			vote_mask[h,w] = label

	vote_mask[vote_mask==1] = 255
	save_pre_img(VOTE_RESULT[0],img_id.split('.')[0],vote_mask)
	print('finish vote:',img_id.split('.')[0])


if __name__ == '__main__':
	for i in range(len(RESULT)):
		img_list = load_dir(RESULT[i])
		for j in range(len(img_list)):
			print(img_list[j])
			vote_per_result(img_list[j])
