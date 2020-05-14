Requires Tensorflow and Keras, optionally Matplotlib and Scikit-Learn

**Important files:**

bidirectional.py: trains a new model

load_data.py: helper methods when loading data for a model:

-load_csv: given a csv where each row is (label(int), sequence(string)), returns a list of (sequence, label)
  set divide to n to keep 1/n of the data, which may be needed to save memory.
      
-get_onehot: given a list of (sequence, label), return a onehot encoded batch for input to the network
    
network_templates.py: defines the neural networks. For best results, 
  use dna_mask_blstm for DNA data and aa_mask_blstm for amino acid data

test_lstm.py: loads a model and test data, prints test accuracy, saves confusion matrix



**Important parameters to set for each experiment:**

is_dna_data: true for DNA, false for amino acids. determines alphabet.

num_classes: number of labels (from 0 to num_classes-1) for this dataset.

sequence_length: maximum sequence length. shorter sequences are ok, longer will be cropped.

embed_size: length of embedding layer, just before output layer. Default is 256

model_name: an arbitrary string to name this model, for saving and loading

model_file or save_path: choose a directory in which to save your model

model_template: should be dna_mask_blstm or aa_mask_blstm, see above

data_dir: name of directory containing your data. should include train.csv and test.csv.

mask: whether to use masking or not. should always be true.

mask_len: the length of the mask input, equal to the length of the output from the final max pooling layer. 
  113 for the current models with sequence length 4500. If you change the input size, Keras will give you an error message containing the desired number.
  mask must be created manually because convolutional layers do not support masking.
  
num_episodes: number of random batches on which to train. Default 200,000 with batch size 100 (second input to get_onehot)
