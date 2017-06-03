from train import train
import models

max_valid_accu, diff = train(model=models.convNet3,
                             learning_rate=5e-4, 
                             data_augmentation_target=3000, 
                             dropout_keep_prob=0.8, 
                             l2_reg_const_param=1e-5,
                             EPOCHS=20)

print(str(max_valid_accu))
print(diff)
