from os import listdir
from os.path import isfile, join
import cv2 as cv
import numpy as np
import pandas as pd

onlyfiles = [f for f in listdir("./ddsmROI") if isfile(join("./ddsmROI", f))]
describers = []
print(len(onlyfiles))
def rgbToColor(rgb):
    return (rgb[0] << 16) + (rgb[1] << 8) + rgb[2]
def getPerimeter(d, index_row, index_col, img):
	list = []
	start_index_row = index_row - d
	start_index_col = index_col - d


	while start_index_row <= index_row + d:
		tmp_index_col = start_index_col
		while tmp_index_col <= index_col + d:

			if start_index_col == index_col and start_index_row == index_row:
				tmp_index_col = tmp_index_col + 1
				continue
			list.append(img[start_index_row][start_index_col])
			tmp_index_col = tmp_index_col + 1
		start_index_row = start_index_row +1

	return list

def calculateDescriptor(img):
	correlogram = np.zeros(256, dtype= np.int32)  # init to zeros
	for index_row, row in enumerate(img):
		if index_row == 0 or index_row == len(img) - 1: #avoid borders
			continue

		for index_pixel, pixel in enumerate(row):
			if index_pixel == 0 or index_pixel == len(row) -1: #avoid borders
				continue

			#get perimeter
			perimeter = getPerimeter(1, index_row, index_pixel, img)
			decimal_color = pixel
			correlogram[decimal_color] += perimeter.count(decimal_color)

	return correlogram
describers = []
to_calc = len(onlyfiles)
data = {}
for i in range(256):
	data["col_"+str(i)] = []

data["type"] = []

for f in onlyfiles:
	if f != 'infoDDSM.txt':
		print(to_calc)
		img = cv.imread('./ddsmROI/'+f)
		if img is not None:
			file_name = int(f.split('.')[0])
			type = 0
			if file_name >= 0 and file_name <= 9215:
				type = 0 #NOR
			elif file_name >= 9216 and file_name <= 10103:
				type = 1 # BEN
			elif file_name >= 10104 and file_name <= 11218:
				type = 2 # CAN


			gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
			to_calc -=1
			describer = calculateDescriptor(gray)
			for i,d in enumerate(describer):
				data["col_"+str(i)].append(d)
			data["type"].append(type)

df = pd.DataFrame.from_dict(data)
df.to_csv("data.csv")