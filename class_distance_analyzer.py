from keras.models import Model, load_model
from load_data import load_csv, get_onehot
import numpy as np

"""
As described in the paper, selects <num_classes> pairs of proteins,
embeds them, and calculates distances. Calculates the fraction
of classes in which the correct pairing is within the top <top_n> closest pairings.
"""

#EDIT THESE PARAMETERS (see README)--------------------------
is_dna_data = True

num_classes = 10000 #test classes, not train classes
top_n = [1, 10, 20, 50]

model_name = 'blstm_dna_100class_dspace_4500'
seq_len = 4500
data_file = 'my_dir/test.csv'

mask = False
mask_len = 113

model_file = 'model_dir/'+model_name+'.h5'

#-------------------------------------------------------------

model = load_model(model_file)
#the embedding can be found at "lstm_2": output of last LSTM layer
embed_model = Model(inputs=model.input, outputs=model.get_layer("lstm_2").output)
embed_model.summary()

single_dict = dict()
pair_dict = dict()
data = load_csv(data_file)
for (x, y) in data:
	if y in pair_dict:
		continue
	if y in single_dict:
		assert x != single_dict[y]
		pair_dict[y] = [single_dict[y], x]
	else:
		single_dict[y] = x
	if len(pair_dict) == num_classes:
		break

chosen_data = []
for i in range(2):
	for y in pair_dict:
		x = pair_dict[y][i]
#		print len(x)
		chosen_data.append((x, y))

x, y, m = get_onehot(chosen_data, None, is_dna_data=is_dna_data, seq_len=seq_len, mask_len=mask_len if mask else None)
embed = embed_model.predict([x,m] if mask else x)

pos_counts = dict()
correct_counts = dict()
for n in top_n:
	pos_counts[n] = []
	correct_counts[n] = 0.0
	for _ in range(n):
		pos_counts[n].append(0)

for i in range(num_classes):
	distances = dict()
	ex = embed[i + num_classes]
	for j in range(num_classes):
		dist = np.linalg.norm(ex - embed[j])
		distances[j] = dist
	best = sorted(distances, key=distances.get)#[0:top_n]
	#print i, ":", best
	for n in top_n:
		for pos in range(n):
			if best[pos] == i:
				pos_counts[n][pos] += 1
				correct_counts[n] += 1
for n in top_n:
	print("top", n, ":")
	print(pos_counts[n])
	print(correct_counts[n]/num_classes)
