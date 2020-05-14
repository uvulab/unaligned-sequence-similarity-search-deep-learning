import csv
import random
import numpy as np
import math

random.seed(3)

#csv format: (label, sequence) or (y, x) because that's how the original data was formatted.
#output is reversed to (x, y)!

def load_csv(input_path, divide=1):
	result = []
	i = 0
	with open(input_path) as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			if i % divide == 0:
				result.append((row[1], int(row[0])))
			i += 1
	return result #(x, y) pairs

#pairs: (sequence, label) or (x, y) pairs as returned from load_csv
#set batch_size = None to onehot encode the entire dataset without changing the order
#mask length should be the sequence length after pooling (look at model.summary()) or None for no masking
#mask must be manually created here because convolutional layers do not support automatic masking
#rand_start: cuts of a random sequence from the beginning of each sequence if. Not used in final results. Should be False.
def get_onehot(pairs, batch_size, num_classes=30, seq_len=1500, is_dna_data=False, mask_len=None, rand_start=False):
	letters=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
	#the amino acid data had random occurrences of all 26 letters, 
	#even if not all of them represent an amino acid. Can make the data slightly smaller
	#by removing the extra letters, which will then be encoded as unknown or all zero
	if is_dna_data:
		letters = ['A','C','G','T']
	aa_dict = dict()
	for i in range(len(letters)):
		aa_dict[letters[i]] = i

	sample = random.sample(pairs, batch_size) if batch_size is not None else pairs
	size = len(sample)	

	has_mask = not mask_len is None

	xData=np.zeros((size,seq_len,len(letters)), dtype=np.int8)
	yData=np.zeros((size,num_classes), dtype=np.int8)
	maskData = np.zeros((size,mask_len,1)) if has_mask else None
	
	total_chars = 0
	unknown_chars = 0

	for i in range(size):
	    y=sample[i][1]
	    if y < num_classes:
	    	yData[i,y] = 1
	    seq = sample[i][0]
	    total_chars += len(seq)
	    counter=0
	    start=0
	    if rand_start and len(seq) > seq_len:
		start = random.randint(0, len(seq)-seq_len)
	    for c in seq[start:]:
		if c in aa_dict:
	        	xData[i,counter,aa_dict[c]] = 1
		else:
			unknown_chars += 1
	        counter=counter+1
		if counter == seq_len:
		    break
	    #if counter == 0:
		#print "empty"

	    if has_mask:
		ratio = float(counter)/seq_len
		stop = int(math.ceil(ratio * mask_len))
		for j in range(stop):
		    maskData[i][j][0] = 1
	#print total_chars, unknown_chars
	
	return xData, yData, maskData

