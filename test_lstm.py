
from load_data import load_csv, get_onehot
from model_templates import dna_mask_blstm, aa_mask_blstm
from ml_logging import Logger

#EDIT THESE PARAMETERS (see README)------------------------------------------

is_dna_data = True

model_name = 'my_model'
model_file = 'model_dir/'+model_name+'.h5'
data_dir = 'my_dir'
sequence_length = 4500
random_crop = False #chooses random start positions for test sequences. Recommend keeping False.
print_acc = True #print test accuracy
save_stats = True #save confusion matrix, sequence length stats (see paper).
num_classes = 100

mask = True
mask_len = 113
model_template = dna_mask_blstm

#----------------------------------------------------------------------------

num_letters = 4 if is_dna_data else 26


model = model_template(num_classes, num_letters, sequence_length, embed_size=256, mask_length=mask_len if mask else None)

model.load_weights(model_file)
model.summary()

test_data = load_csv(data_dir + '/test.csv', divide=2 if is_dna_data else 1)
print(len(test_data))

"""
crop_count = 0.0
for seq, y in test_data:
	if len(seq) > sequence_length:
		crop_count += 1
print "percent cropped: ", crop_count / len(test_data)	
"""

test_x, test_y, test_m = get_onehot(test_data, None, is_dna_data=is_dna_data, seq_len=sequence_length, num_classes=num_classes, rand_start=random_crop, mask_len=mask_len if mask else None)
if print_acc:
	print("test accuracy: ", model.evaluate([test_x, test_m] if mask else test_x, test_y, batch_size=100))

if save_stats:
	pred = model.predict([test_x, test_m] if mask else test_x, batch_size=100).argmax(axis=-1)
	log = Logger(model_name, num_classes, sequence_length)
	log.confusion_matrix(test_data,pred)
	#uncomment for length analysis
	#log.length_stats(test_data,pred)
	#log.length_histograms(test_data,pred)
	log.save()


