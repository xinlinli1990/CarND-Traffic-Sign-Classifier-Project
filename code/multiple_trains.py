from train import train
import models

dropout_keep_probs = [0.25, 0.5, 0.75]#[0.8] #0.25, 0.3, 0.35, 0.5, 0.55, 0.6,
learning_rates = [5e-4] #, 1e-4, 1e-3
l2_reg_const_params = [0.0, 1e-5, 1e-4, 1e-3]

for learning_rate in learning_rates:
    for dropout_keep_prob in dropout_keep_probs:
        is_overfitting = False
        for l2_reg_const_param in l2_reg_const_params:
            print("dropout="+str(dropout_keep_prob))
            print("learning_rate="+str(learning_rate))
            print("l2="+str(l2_reg_const_param))
            print("training ...")
            
            max_valid_accu, diff = train(model=models.convNet2,
                                         learning_rate=learning_rate,
                                         data_augmentation_target=3000, 
                                         dropout_keep_prob=dropout_keep_prob, 
                                         l2_reg_const_param=l2_reg_const_param,
                                         EPOCHS=50)
                
            print("max_valid="+str(max_valid_accu))
            print("diff="+str(diff))
            print("")
            
            if max_valid_accu <= 0.1 and abs(diff) <= 0.1:
                print("Underfitting!\n\n")
                break
                
            if diff >= 0.3:
                is_overfitting = True
            
            # if diff > max_diff:
                # max_diff = diff
        
        if is_overfitting == True:
            print("Overfitting! \n\n")
            break
