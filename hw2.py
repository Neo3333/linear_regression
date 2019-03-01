import numpy as np
import random
import pylab as pl
import time

def calculate(data_list):
	mean = 0
	std = 0
	for data in data_list:
		mean += data
	mean = mean / len(data_list)
	for data in data_list:
		std += (data - mean) ** 2
	std = std / len(data_list)
	std = std ** (0.5)
	#print(mean,std)
	return mean,std

def normalize(data):
	size_list = []
	num_list = []
	price_list = []
	for line in data:
		line_list = line[0:-1].split(",")
		#print(line_list)
		size_list.append(int(line_list[0]))
		num_list.append(int(line_list[1]))
		price_list.append(line_list[2])
	#print(price_list)
	mean_size, std_size = calculate(size_list)
	mean_num, std_num = calculate(num_list)
	file_norm = open("normalize.txt","w")
	for i in range(len(size_list)):
		size_norm = ((size_list[i] - mean_size) / std_size)
		num_norm = ((num_list[i] - mean_num) / std_num)
		file_norm.write(str(size_norm) + "," + str(num_norm) + "," + price_list[i] + '\n')
	return mean_size, std_size, mean_num, std_num

def gradient_decent(data, alpha, max_iter):
	iteration = 0
	x_list = []
	y_list = []
	cost_list = []
	w = np.zeros(3)
	#print(w)
	for line in data:
		line_list = line[0:-1].split(",")
		x = []
		x.append(1)
		x.append(float(line_list[0]))
		x.append(float(line_list[1]))
		x_list.append(np.array(x))
		y_list.append(float(line_list[2]))

	#print(x_list,y_list)
	while (iteration < max_iter):
		cost = 0
		for i in range(len(x_list)):
			cost += (np.dot(w,x_list[i]) - y_list[i]) ** 2
		cost = cost /(2 * len(x_list))
		#print(cost)
		cost_list.append(cost)
		temp_w = []
		for i in range(3):
			gradient = 0
			for j in range(len(x_list)):
				gradient += (np.dot(w,x_list[j]) - y_list[j]) * x_list[j][i]
			gradient = gradient / len(x_list)
			temp = w[i] - alpha * gradient
			temp_w.append(temp)
			#temp_w.append(w[i] - alpha * gradient)
		for i in range(3):
			w[i] = temp_w[i]
		iteration += 1

	return w, iteration,cost_list

def predict(w,x1,x2,x1_mean,x2_mean,x1_std,x2_std):
	x1_norm = (x1 - x1_mean) / x1_std
	x2_norm = (x2 - x2_mean) / x2_std
	x = np.array([1,x1_norm,x2_norm])
	y = np.dot(w,x)
	return y

def random_shuffle(x_list,y_list):
	new_x = []
	new_y = []
	while (len(x_list) > 0):
		index = random.randint(0, len(x_list) - 1)
		temp_x = x_list[index]
		temp_y = y_list[index]
		new_x.append(temp_x)
		x_list.pop(index)
		new_y.append(temp_y)
		y_list.pop(index)
	return new_x,new_y

def stochastic_gradient(data, alpha, max_iter):
	iteration_sg = 0
	x_list = []
	y_list = []
	cost_list = []
	w = np.zeros(3)
	#print(w)
	for line in data:
		line_list = line[0:-1].split(",")
		x = []
		x.append(1)
		x.append(float(line_list[0]))
		x.append(float(line_list[1]))
		x_list.append(np.array(x))
		y_list.append(float(line_list[2]))

	while(iteration_sg < max_iter):
		cost = 0
		for i in range(len(x_list)):
			cost += (np.dot(w,x_list[i]) - y_list[i]) ** 2
		cost = cost /(2 * len(x_list))
		#print(cost)
		cost_list.append(cost)

		for i in range(len(x_list)):
			f = np.dot(x_list[i],w)
			for j in range(len(w)):
				w[j] = w[j] - alpha * (f - y_list[i]) * x_list[i][j]
		x_list, y_list = random_shuffle(x_list,y_list)

		iteration_sg += 1
	return w,cost_list


if __name__ == "__main__":
	file = open("housing.txt","r")
	x1_mean,x1_std,x2_mean,x2_std = normalize(file)
	alpha_list = [0.01,0.03,0.1,0.2,0.5]
	iter_list = [10,20,30,40,50,60,70]
	for i in range(5):
		for j in range(7):
			file_train = open("normalize.txt","r")
			w, iteration, cost_list = gradient_decent(file_train,alpha_list[i],iter_list[j])
			print("using alpha = " + str(alpha_list[i]) + " with " + str(iter_list[j]) + " iterations")
			print(cost_list[-1])
	predict_price = predict(w,2650,4,x1_mean,x1_std,x2_mean,x2_std)
	print("using training results with alpha = 0.5 and 70 iterations, the predicton is " + '\n' + str(predict_price))
	file_train = open("normalize.txt","r")
	start_time_sg = time.time()
	w_sg, cost_list_sg = stochastic_gradient(file_train,0.05,3)
	end_time_sg = time.time()
	time_sg = end_time_sg - start_time_sg
	print("using stochastic gradient decent with alpha = 0.05 and 3 iterations")
	print(cost_list_sg[-1])
	print("computation time = " + str(time_sg))
	file_train = open("normalize.txt","r")
	start_time = time.time()
	w,iteration,cost_list = gradient_decent(file_train,0.05,80)
	end_time = time.time()
	time = end_time - start_time
	print("using standard gradient decent with alpha = 0.05 and 80 iterations")
	print(cost_list[-1])
	print("computation time = " + str(time))

