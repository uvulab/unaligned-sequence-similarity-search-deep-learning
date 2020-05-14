import numpy as np
from keras.models import Model, load_model
from load_data import get_onehot
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from load_data import load_csv, get_onehot

#displays a t-SNE cluster visualization

#EDIT THESE PARAMETERS (see README)-----------------------------
model_name = 'my_model'
input_file = 'my_dir/test.csv'
display_classes = 10
n = 100 #examples per class

is_dna_data = True
seq_len = 4500
mask_len = 113

model_file = 'model_dir/'+model_name+'.h5'
---------------------------------------------------------------

model = load_model(model_file)
embed_model = Model(inputs=model.input, outputs=model.get_layer("lstm_2").output)
embed_model.summary()

counts = np.zeros(display_classes, dtype = np.int8)
data = load_csv(input_file, divide=1)
chosen_data = []

for (x, y) in data:
	if y < display_classes and counts[y] < n:
		counts[y] = counts[y] + 1
		chosen_data.append((x,y))

x, y, m = get_onehot(chosen_data, None, is_dna_data=is_dna_data, seq_len=seq_len, mask_len=mask_len)
embed = embed_model.predict([x,m], batch_size=100, verbose=1)

tsne = TSNE(n_components=2, random_state=0)
xx = tsne.fit_transform(embed)

colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'magenta', 'brown', 'gray', 'black']
for i in range(len(chosen_data)):
	plt.scatter([xx[i,0]], [xx[i,1]], c=colors[chosen_data[i][1]])
plt.show()
