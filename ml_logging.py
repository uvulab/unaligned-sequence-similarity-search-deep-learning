import numpy as np
import csv
import math

class Logger:
	def __init__(self, path, num_classes, seq_len):
		self.acc_plot = []
		self.conf_mat = None
		self.len_plot = None
		self.len_stats = None
		self.len_buckets = None

		self.path = path
		self.num_classes = num_classes
		self.seq_len = seq_len

	#in case you want to record accuracy during training. Not used currently.
	def record_val_acc(self, time, acc):
		self.acc_plot.append([time, acc])
	
	#data: (x, y) pairs before onehot encoding, pred: predicted class integers
	def confusion_matrix(self, data, pred):
		self.conf_mat = np.zeros((self.num_classes, self.num_classes), dtype=np.int32)
		for i in range(len(data)):
			self.conf_mat[data[i][1], pred[i]] += 1

	#no longer used, replaced by length_histograms
	def length_plot(self, data, pred):
		lengths = []
		for (x, y) in data:
			lengths.append(len(x))
		self.len_plot = []
		for i in range(len(data)):
			correct = 1 if data[i][1] == pred[i] else 0
			self.len_plot.append([lengths[i], correct])

	def length_stats(self, data, pred):
		class_lengths = []
		for i in range(self.num_classes):
			class_lengths.append([])
		for (x, y) in data:
			class_lengths[y].append(len(x))
		self.len_stats = []
		for i in range(self.num_classes):
			l = class_lengths[i]
			self.len_stats.append([i, np.mean(l), np.median(l), np.std(l), np.var(l)])
	
	#requires length_stats called first
	def length_histograms(self, data, pred, bucket_size=50):
		self.len_buckets = []
		for i in range(self.seq_len / bucket_size):
			label = (i+1) * bucket_size
			self.len_buckets.append([label,0,0,label,0,0])
		for i in range(len(data)):
			length = min(len(data[i][0]),self.seq_len)
			len_bucket = int(max(math.floor((length - 1)/bucket_size), 0))
			y = data[i][1]
			dtm = abs(length - self.len_stats[y][1])#mean
			dtm_bucket = int(max(math.floor((dtm - 1)/bucket_size), 0))
			correct = y == pred[i]
			if correct:
				self.len_buckets[len_bucket][1] += 1
				self.len_buckets[dtm_bucket][4] += 1
			else:
				self.len_buckets[len_bucket][2] += 1
				self.len_buckets[dtm_bucket][5] += 1

	def save(self):
		
		if len(self.acc_plot) > 0:
			with open(self.path + '_acc_plot.csv', 'w') as outfile:
				w = csv.writer(outfile)
				for row in self.acc_plot:
					w.writerow(row)
		if not self.conf_mat is None:
			with open(self.path + '_conf_matrix.csv', 'w') as outfile:
				w = csv.writer(outfile)
				w.writerow(['class'] + range(self.num_classes) + ['total', 'acc'])
				cm = self.conf_mat.tolist()
				for i in range(self.num_classes):
					total = 0
					for count in cm[i]:
						total += count
					acc = float(cm[i][i]) / max(total, 1)
					w.writerow([i] + cm[i] + [total, acc])
		
		if not self.len_plot is None:
			with open(self.path + '_length_plot.csv', 'w') as outfile:
				w = csv.writer(outfile)
				for row in self.len_plot:
					w.writerow(row)

		if not self.len_stats is None:
			with open(self.path + '_length_stats.csv', 'w') as outfile:
				w = csv.writer(outfile)
				w.writerow(['class','mean','median','stdev','variance'])
				for row in self.len_stats:
					w.writerow(row)

		if not self.len_buckets is None:
			with open(self.path + '_length_histograms.csv', 'w') as outfile:
				w = csv.writer(outfile)
				w.writerow(['length','correct','incorrect','dist to mean','correct','incorrect'])
				for row in self.len_buckets:
					w.writerow(row)
