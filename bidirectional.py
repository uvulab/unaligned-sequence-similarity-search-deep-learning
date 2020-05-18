from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import numpy as np

from load_data import load_csv, get_onehot
from ml_logging import Logger
from model_templates import dna_mask_blstm, aa_mask_blstm


#EDIT THESE PARAMETERS (see README)-------------------------------------------

is_dna_data = False

num_classes = 30
num_letters = 4 if is_dna_data else 26
sequence_length = 1500
embed_size = 64
model_name = 'my_model'
model_template = aa_mask_blstm
data_dir = 'my_dir'

mask = True
mask_len = 113

save_path = 'model_dir/'+model_name+'.h5'

#-----------------------------------------------------------------------------

model = model_template(num_classes, num_letters, sequence_length, embed_size=embed_size, mask_length=mask_len if mask else None)
model.summary()

train_data = load_csv(data_dir + '/train.csv')
print(len(train_data))

num_episodes = 200000
for i in range(num_episodes):
        x, y, m = get_onehot(train_data, 100, num_classes=num_classes, seq_len=sequence_length, is_dna_data=is_dna_data, mask_len=mask_len if mask else None)
        print(i)
        print(model.train_on_batch([x,m] if mask else x, y))
        if (i % 10000 == 0) or i == num_episodes - 1:

                model.save(save_path)
                print('saved to ' + save_path)
del train_data

test_data = load_csv(data_dir + '/test.csv', divide=2 if is_dna_data else 1)
test_x, test_y, test_m = get_onehot(test_data, None, num_classes=num_classes, seq_len=sequence_length, is_dna_data=is_dna_data, mask_len=mask_len if mask else None)
print("test accuracy: ", model.evaluate([test_x, test_m] if mask else test_x, test_y, batch_size=100))
