from __future__ import print_function
import tensorflow as tf
import numpy as np
import csv
import os
import time
import argparse

def get_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument("--DATA_DIR", type=str, default='/home/class17/fuyang/Desktop/ccf')
	parser.add_argument("--DATA_DISTANCE_NAME", type=str, default='m_615_wifiname.csv')
	parser.add_argument("--DATA_POWER_NAME", type=str, default='m_615_wifistr.csv')
	parser.add_argument("--DATA_LABEL", type=str, default='m_615_label.csv')
	parser.add_argument("--RESTORE_FROM", type=str, default="model.ckpt-40000")
	parser.add_argument("--OUTPUT_DIR", type=str, default="/home/class17/fuyang/Desktop/CCF/m_615/")
	parser.add_argument("--DONT_STORE", action="store_true", default=False)
	parser.add_argument("--TOP5", action="store_true", default=False)
	parser.add_argument("--TOP5_NAME", type=str, default='top5')
	parser.add_argument("--PREDICT", action="store_true", default=False)
	parser.add_argument("--PREDICT_FILE", type=str, default="predict_id")
	parser.add_argument("--ID2SHOPID", type=str, default="m_615_shop_id2id.csv")
	parser.add_argument("--OUTPUT_NUM", type=str, default="1/")
	parser.add_argument("--GPU",type=str, default='0')
	parser.add_argument("--STEP",type=int, default=40001)
	parser.add_argument("--WRITELOG",action="store_true", default=False)
	return parser.parse_args()

def readPower(data_dir, data_name):
	reader = csv.reader(open(os.path.join(data_dir, data_name)))
	data_raw = list(reader)
	row_num = [i[0] for i in data_raw]
	row_num = map(int, row_num)
	data_raw = [i[1:] for i in data_raw]
	data = []
	for i in range(len(data_raw)):
		x = data_raw[i][0].split('[')[1].split(']')[0].split(',')
		data.append(x)
	del data_raw
	for i in range(len(data)):
		data[i] = map(float, data[i])
	for i in range(len(data)):
		data[i] = [[data[i][j]] for j in range(len(data[i]))]
	for i in range(len(data)):
		while(len(data[i]) != 10):
			data[i].append([0.0])
	return data, row_num

def readData(data_dir, data_name):
	reader = csv.reader(open(os.path.join(data_dir, data_name)))
	data_raw = list(reader)
	shop_id = [i[0] for i in data_raw]
	shop_id = map(int, shop_id)
	data_raw = [i[1:] for i in data_raw]
	data = []
	for i in range(len(data_raw)):
		x = data_raw[i][0].split('[')[1].split(']')[0].split(',')
		data.append(x)
	del data_raw
	for i in range(len(data)):
		data[i] = map(int, data[i])
	Max = 0
	xxxx = 0
	o = 0
	for i in range(len(data)):
		if len(data[i]) > xxxx:
			xxxx = len(data[i])
			o = i
	for i in range(len(data)):
		if Max < max(data[i]):
			Max = max(data[i])
	out = [[[0 for i in range(Max + 1)] for i in range(10)] for i in range(len(data))]
	for i,x in enumerate(data):
		for j,y in enumerate(x):
			out[i][j][y] = 1
	return out

def readDataTest(data_dir, data_name, WIFINUM):
	reader = csv.reader(open(os.path.join(data_dir, data_name)))
	data_raw = list(reader)
	shop_id = [i[0] for i in data_raw]
	shop_id = map(int, shop_id)
	data_raw = [i[1:] for i in data_raw]
	data = []
	for i in range(len(data_raw)):
		x = data_raw[i][0].split('[')[1].split(']')[0].split(',')
		data.append(x)
	del data_raw
	for i in range(len(data)):
		data[i] = map(int, data[i])
	out = [[[0 for i in range(WIFINUM)] for i in range(10)] for i in range(len(data))]
	for i,x in enumerate(data):
		for j,y in enumerate(x):
			out[i][j][y] = 1
	return out

def save(saver, sess, logdir, step):
	model_name = 'model.ckpt'
	checkpoint_path = os.path.join(logdir, model_name)

	if not os.path.exists(logdir):
		os.makedirs(logdir)
	saver.save(sess, checkpoint_path, global_step=step)


def readLabel(data_dir, data_name):
	reader = csv.reader(open(os.path.join(data_dir, data_name)))
	data_raw = list(reader)
	data_raw = [i[1] for i in data_raw]
	data = map(int, data_raw)
	out = [[0 for i in range(max(data) + 1)] for i in range(len(data))]
	for i,x in enumerate(data):
		out[i][x] = 1
	return out

def validLabel(data_dir, data_name):
	reader = csv.reader(open(os.path.join(data_dir, data_name)))
	data = list(reader)
	data = [i[1] for i in data]
	data = map(int, data)

	return data

def acc(label, predict):
	assert len(label) == len(predict)
	pre = 0
	for i in range(len(label)):
		if label[i] == predict[i]:
			pre += 1
	return 1.0*pre / len(label)

def main():

	args = get_arguments()

	os.environ["CUDA_VISIBLE_DEVICES"]=args.GPU
	TRAIN = True
	BATCH_SIZE = 16
	FEATURE_NUM = 5
	LEARNING_RATE = 0.001
	WIFIPOWER_NUM = 8
	WIFI_NUM = 100
	DATA_DIR = args.DATA_DIR
	DATA_DISTANCE_NAME = args.DATA_DISTANCE_NAME
	DATA_POWER_NAME = args.DATA_POWER_NAME
	ID2SHOPID = args.ID2SHOPID
	DATA_LABEL = args.DATA_LABEL
	RESTORE_FROM = args.OUTPUT_DIR + args.OUTPUT_NUM + 'snapshot/' + args.RESTORE_FROM
	OUTPUT_DIR = args.OUTPUT_DIR + args.OUTPUT_NUM
	EPOCHS = 100001
	MAX_WIFI = 10
	Lambda = 0.0001
	epsilon = 1e-12
	SHOP_NUM = 200
	#tf.set_random_seed(RANDOM_SEED)

	read = open(DATA_DIR + '/' + DATA_LABEL.split('_label')[0] + '_WIFINUM', 'r')
	WIFI_NUM = int(read.readline())
	if args.PREDICT == False:
		training_data_distance = np.array(readData(DATA_DIR, DATA_DISTANCE_NAME))
	else:
		training_data_distance = np.array(readDataTest(DATA_DIR, DATA_DISTANCE_NAME, WIFI_NUM))
	training_data_power, row_num = readPower(DATA_DIR, DATA_POWER_NAME)
	training_data_power = np.array(training_data_power)
	training_data_power = training_data_power / 100.0 + 1
	training_data_power = training_data_power * (1 - (training_data_power == 1))
	training_label = np.array(readLabel(DATA_DIR, DATA_LABEL))

	if args.PREDICT == False:
		Train_label = validLabel(DATA_DIR, DATA_LABEL)
	assert WIFI_NUM == len(training_data_distance[0][0])
	#write = open('/home/dc/xuzhi/CCF/level_20/' + DATA_LABEL.split('_label.csv')[0] + '_WIFINUM', 'w')
	#write.write(str(WIFI_NUM))
	#write.close()
	#exit()
	print('sadsadasd' + str(WIFI_NUM))
	SHOP_NUM = len(training_label[0])

	with tf.name_scope('input') as scope:
		#load input
		data_distance_initializer = tf.placeholder(dtype = tf.float32,
									shape=training_data_distance.shape)
		data_power_initializer = tf.placeholder(dtype = tf.float32,
									shape=training_data_power.shape)
		label_initializer = tf.placeholder(dtype = tf.float32,
									shape=training_label.shape)

		input_distance_data = tf.Variable(data_distance_initializer, dtype = tf.float32, trainable=False, collections=[])
		input_power_data = tf.Variable(data_power_initializer, dtype = tf.float32, trainable=False, collections=[])
		input_label = tf.Variable(label_initializer, dtype = tf.float32, trainable=False, collections=[])

		distance_data, power_data, label = tf.train.slice_input_producer([input_distance_data, input_power_data, input_label], num_epochs=EPOCHS, shuffle=False)

		train_distance, train_power, train_label = tf.train.batch([distance_data, power_data, label], batch_size=BATCH_SIZE, num_threads=1, capacity=64)

	if TRAIN == False:
		train_distance = input_distance_data
		train_power = input_power_data
		train_distance = tf.cast(train_distance, tf.float32)
		train_power = tf.cast(train_power, tf.float32)
		BATCH_SIZE = len(training_data_distance)
	flat_train_distance = tf.reshape(train_distance, [-1, WIFI_NUM])
	flat_train_distance_train = tf.reshape(input_distance_data, [-1, WIFI_NUM])
	BATCH_SIZE_train = len(training_data_distance)

	batch_num = tf.expand_dims(tf.reduce_sum(tf.reduce_sum(train_distance,axis=1),axis=1),axis=1)
	batch_num_train = tf.expand_dims(tf.reduce_sum(tf.reduce_sum(input_distance_data,axis = 1),axis=1),axis=1)
	batch_num = tf.cast(batch_num, tf.float32)
	batch_num_train = tf.cast(batch_num_train, tf.float32)

	with tf.name_scope('Embedding') as scope:
		w_emb = tf.Variable(tf.truncated_normal([WIFI_NUM,FEATURE_NUM], stddev=np.sqrt(2.0/(WIFI_NUM ))),name='weights')
		b_emb = tf.Variable(tf.truncated_normal([FEATURE_NUM]), name='biases')

	flat_out = tf.nn.relu(tf.matmul(flat_train_distance, w_emb) + b_emb)
	flat_out_train = tf.nn.relu(tf.matmul(flat_train_distance_train, w_emb) + b_emb)

	flat_train_power = tf.reshape(train_power, [-1, 1])
	flat_train_power_train = tf.reshape(input_power_data, [-1,1])

	with tf.name_scope('power_hidden1') as scope:
		w1 = tf.Variable(tf.truncated_normal([1, 4], stddev=np.sqrt(2.0)), name='weights')
		b1 = tf.Variable(tf.truncated_normal([4]), name='biases')

	hidden1_power = tf.nn.relu(tf.matmul(flat_train_power, w1) + b1)
	hidden1_power_train = tf.nn.relu(tf.matmul(flat_train_power_train, w1) + b1)

	with tf.name_scope('power_hidden2') as scope:
		w2 = tf.Variable(tf.truncated_normal([4, 8], stddev=np.sqrt(2.0/(4))),name='weights')
		b2 = tf.Variable(tf.truncated_normal([8]))
	hidden2_power = tf.nn.relu(tf.matmul(hidden1_power, w2) + b2)
	hidden2_power_train = tf.nn.relu(tf.matmul(hidden1_power_train, w2) + b2)

	assert (flat_out.shape[0]==hidden2_power.shape[0]), 'shape[0] not match'
	raw_fc_input = tf.concat( [flat_out, hidden2_power], 1,name="concat")
	raw_fc_input_train = tf.concat([flat_out_train, hidden2_power_train], 1, name="concat_train")

	with tf.name_scope('concat_fc') as scope:
		w_fc1 = tf.Variable(tf.truncated_normal([8+FEATURE_NUM, 2*(8+FEATURE_NUM)], stddev=np.sqrt(2.0/(8+FEATURE_NUM))), name='weights')
		b_fc1 = tf.Variable(tf.truncated_normal([2*(8+FEATURE_NUM)]))
	raw_fc_input2 = tf.nn.relu(tf.matmul(raw_fc_input, w_fc1) + b_fc1)
	raw_fc_input2_train = tf.nn.relu(tf.matmul(raw_fc_input_train, w_fc1) + b_fc1)

	with tf.name_scope('concat_fc2') as scope:
		w_fc2 = tf.Variable(tf.truncated_normal([2*(8+FEATURE_NUM), 4*(8+FEATURE_NUM)], stddev=np.sqrt(2.0/(2*(8+FEATURE_NUM)))), name='weights')
		b_fc2 = tf.Variable(tf.truncated_normal([4*(8+FEATURE_NUM)]))
	raw_fc_input3 = tf.nn.relu(tf.matmul(raw_fc_input2, w_fc2) + b_fc2)
	raw_fc_input3_train = tf.nn.relu(tf.matmul(raw_fc_input2_train, w_fc2) + b_fc2)

	with tf.name_scope('concat_fc3') as scope:
		w_fc3 = tf.Variable(tf.truncated_normal([4*(8+FEATURE_NUM), 8*(8+FEATURE_NUM)], stddev=np.sqrt(2.0/(4*(8+FEATURE_NUM)))), name = 'weights')
		b_fc3 = tf.Variable(tf.truncated_normal([8*(8+FEATURE_NUM)]))

	raw_fc_input4 = tf.nn.relu(tf.matmul(raw_fc_input3, w_fc3) + b_fc3)
	raw_fc_input4_train = tf.nn.relu(tf.matmul(raw_fc_input3_train, w_fc3) + b_fc3)

	raw_fc_input_reshape = tf.reshape(raw_fc_input4, [BATCH_SIZE, -1, 8*(8+FEATURE_NUM)])
	raw_fc_input_reshape_train = tf.reshape(raw_fc_input4_train, [BATCH_SIZE_train, -1, 8*(8+FEATURE_NUM)])

	fc_input_sum = tf.reduce_sum(raw_fc_input_reshape, axis=1)
	fc_input_sum_train = tf.reduce_sum(raw_fc_input_reshape_train, axis = 1)

	fc_input = tf.divide(fc_input_sum, batch_num)
	fc_input_train = tf.divide(fc_input_sum_train, batch_num_train)

	with tf.name_scope('hidden1') as scope:
		w_hidden1 = tf.Variable(tf.truncated_normal([8*(8+FEATURE_NUM), 64], stddev=np.sqrt(2.0/(8*(8+FEATURE_NUM)))),name='weights')
		b_hidden1 = tf.Variable(tf.truncated_normal([64]),name='biases')

	hidden1 = tf.nn.relu(tf.matmul(fc_input, w_hidden1) + b_hidden1)
	hidden1_train = tf.nn.relu(tf.matmul(fc_input_train, w_hidden1) + b_hidden1)

	with tf.name_scope('hidden2') as scope:
		w_hidden2 = tf.Variable(tf.truncated_normal([64, 128], stddev=np.sqrt(2.0/(64 ))), name = 'weights')
		b_hidden2 = tf.Variable(tf.truncated_normal([128]), name='biases')

	hidden2 = tf.nn.relu(tf.matmul(hidden1, w_hidden2) + b_hidden2)
	hidden2_train = tf.nn.relu(tf.matmul(hidden1_train, w_hidden2) + b_hidden2)

	with tf.name_scope('hidden3') as scope:
		w_hidden3 = tf.Variable(tf.truncated_normal([128, SHOP_NUM], stddev=np.sqrt(2.0/(128 ))), name = 'weights')
		b_hidden3 = tf.Variable(tf.truncated_normal([SHOP_NUM]), name='biases')

	hidden3 = tf.matmul(hidden2, w_hidden3) + b_hidden3
	hidden3_train = tf.matmul(hidden2_train, w_hidden3) + b_hidden3

	y_pred = tf.nn.softmax(hidden3)
	y_pred_train = tf.nn.softmax(hidden3_train)

	prediction = tf.argmax(y_pred, 1)
	prediction_train = tf.argmax(y_pred_train, 1)

	if args.PREDICT == False:
		#cross_entropy = -tf.reduce_sum(train_label*tf.log(y_pred + epsilon)+(1 - train_label) * tf.log(1 - y_pred + epsilon), 1)
		l2_losses = [Lambda * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weight' in v.name]

		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=train_label, logits=hidden3))
		loss = tf.reduce_mean(cross_entropy) + tf.add_n(l2_losses)
		global_step = tf.Variable(0)
		#learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, 1000, 0.7, staircase=True)
		train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss,global_step=global_step)

	init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
	#summary_writer = tf.summary.FileWriter('./snapshot', graph=tf.get_default_graph())
	saver = tf.train.Saver(var_list = tf.global_variables(), max_to_keep=50)
	#grads = tf.gradients(tf.reduce_mean(cross_entropy), [w_emb, w1, w2, w_fc1, w_hidden1, w_hidden2, w_hidden3])

	tf_config = tf.ConfigProto()
	tf_config.gpu_options.allow_growth = True
	#tf_config.gpu_options.per_process_gpu_memory_fraction = 0.2

	sess = tf.Session(config=tf_config)
	sess.run(init)

	sess.run(input_distance_data.initializer, feed_dict={data_distance_initializer:training_data_distance})
	sess.run(input_power_data.initializer, feed_dict={data_power_initializer:training_data_power})
	sess.run(input_label.initializer, feed_dict={label_initializer:training_label})

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess, coord)

	f='\n\n\n\n'

	if args.PREDICT == True:
		saver = tf.train.Saver()
		saver.restore(sess, RESTORE_FROM)
		predict, soft = sess.run([prediction_train, y_pred_train])
		writer = open(OUTPUT_DIR + args.PREDICT_FILE, 'w')
		for i in predict:
			writer.write(str(i) + '\n')
		writer.close()
		reader = csv.reader(open(args.DATA_DIR + '/' + ID2SHOPID, 'r'))
		shop = list(reader)[1:]
		shop_id = []
		for i in shop:
			shop_id.append(i[1])
		temp = csv.reader(open(os.path.join(DATA_DIR, DATA_DISTANCE_NAME), 'r'))
		temp = list(temp)
		row_id = [i[0] for i in temp]
		writer = open(OUTPUT_DIR + args.PREDICT_FILE + '_final', 'w')
		for i in range(len(predict)):
			writer.write(row_id[i] + ',' + shop_id[int(predict[i])] + '\n')#'',' + str(soft[i][int(predict[i])]) + '\n')
		writer.close()
		coord.request_stop()
		coord.join(threads)
		exit()

	if args.TOP5 == True:
		saver = tf.train.Saver()
		saver.restore(sess, RESTORE_FROM)

		predict = sess.run(prediction_train)
		#accuracy = acc(Train_label, predict)
		#print("the train accuracy is " + str(accuracy))
		yy = sess.run(y_pred_train)
		y_pred_sort = np.argsort(-yy, axis=1)
		y_max = y_pred_sort[:,0:5]
		#for i in range(len(y_max)):
		#	for j in range(4):
		#		if Train_label[i] == y_max[i][j]:
		#			temp = y_max[i][j]
		#			y_max[i][j] = y_max[i][4]
		#			y_max[i][4] = temp
		#			break
		top5 = open(OUTPUT_DIR + args.TOP5_NAME, 'w')
		for i in range(len(y_max)):
			top5.write(str(row_num[i])+',')
			for j in range(4):
				top5.write(str(y_max[i][j]) + ',')
			top5.write(str(y_max[i][4]))
			top5.write('\n')
		top5.close()

		coord.request_stop()
		coord.join(threads)
		exit()
	try:
		log = open(OUTPUT_DIR+'outputlog','w')
		while not coord.should_stop():
			start_time = time.time()
			#_, loss_value, cross, num, W1, W2, W_hidden1, W_hidden2, W_hidden3, W_emb = sess.run([train, loss, cross_entropy, batch_num, w1,w2,w_hidden1,w_hidden2,w_hidden3,w_emb])
			_, loss_value, cross, www,www1,p, step = sess.run([train, loss, cross_entropy, w_emb,w1,y_pred, global_step])

			duration = time.time() - start_time

			if step % 1000 == 0:
				if args.DONT_STORE == False:
					save(saver, sess, OUTPUT_DIR +'snapshot', step)
				predict = sess.run(prediction_train)
				accuracy = acc(Train_label, predict)
				if args.WRITELOG:
					log.write('The checkpoint has been created.' + '\n')
					log.write("the train accuracy is " + str(accuracy) + '\n')
				else:
					print('The checkpoint has been created.')
					print("the train accuracy is " + str(accuracy))

				#write.write(str(accuracy) + '\n')
				#print("the accuracy is stored.")
			step += 1
			if step == args.STEP:
				break
			if step % 50 == 0:
				if args.WRITELOG:
					log.write('step {:d} \t loss = {:.3f}, cross = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, cross, duration) + '\n')
				else:
					print('step {:d} \t loss = {:.3f}, cross = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, cross, duration))
			#print(p)
		log.close()

	except tf.errors.OutOfRangeError:
		print("OutofRangeError")
	finally:
		coord.request_stop()
	coord.join(threads)
if __name__ == '__main__':
	main()
