import sys
import os

from train import train
import models

def read_nomad_input_file(argv):
    with open(argv[1], 'r') as nomad_input:
        nomad_input_data = nomad_input.read().splitlines()[0].split(" ")
    return nomad_input_data

input_parameters = read_nomad_input_file(sys.argv)
learning_rate, dropout_keep_prob, l2_reg_const_param = input_parameters

# print(learning_rate)
# print(data_augmentation_target)
# print(dropout_keep_prob)
# print(l2_reg_const_param)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

max_valid_accu, diff_valid_train = train(model=models.LeNetWithDropout,
                                         learning_rate=float(learning_rate), 
                                         data_augmentation_target=3000,
                                         dropout_keep_prob=float(dropout_keep_prob), 
                                         l2_reg_const_param=float(l2_reg_const_param),
                                         EPOCHS=20)
                                       
print("{0:.6f} {1:.6f}".format(1.0/max_valid_accu, diff_valid_train - 0.30))
