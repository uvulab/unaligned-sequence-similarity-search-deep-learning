from keras.models import Model, load_model
from load_data import load_csv, get_onehot
import numpy as np
import csv

#Saves <ex_per_class> sample embeddings per class

#EDIT THESE PARAMETERS (see README)-------------------------------------

model_name = 'my_model'
num_classes = 100
ex_per_class = 100
data_file = 'my_dir/test.csv'
name_file = 'my_dir/dna_100class_names.csv' #a csv of (class number, name) for each class
out_file = 'my_dir/embed_'+model_name+'.csv'

is_dna_data = True
mask = True
mask_len = 113
seq_len = 4500

model_file = 'model_dir/'+model_name+'.h5'

#----------------------------------------------------------------------

model = load_model(model_file)
embed_model = Model(inputs=model.input, outputs=model.get_layer("lstm_2").output)
embed_model.summary()

counts = np.zeros(num_classes)
data = load_csv(data_file)
chosen_data = []
for (x, y) in data:
	if counts[y] < ex_per_class:
		chosen_data.append((x,y))
		counts[y] += 1

x, y, m = get_onehot(chosen_data, None, is_dna_data=is_dna_data, seq_len=seq_len, mask_len=mask_len if mask else None)
embed = embed_model.predict([x,m] if mask else x)

print(embed.shape)

names = dict()
with open(name_file, 'r') as infile:
	r = csv.reader(infile)
	for row in r:
		y = int(row[0])
		names[y] = row[1]
with open(out_file, 'w') as outfile:
	w = csv.writer(outfile)
	for (i, (x, y)) in enumerate(chosen_data):
		w.writerow([y,names[y]]+embed[i].tolist())
