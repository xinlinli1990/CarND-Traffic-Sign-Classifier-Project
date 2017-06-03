from train import train
import models

dropout_keep_probs = [#0.25, 
                      #0.5,
                      0.5]                    
                      #0.75]#[0.8] #0.25, 0.3, 0.35, 0.5, 0.55, 0.6,
learning_rates = [#5e-4, 
                  #5e-4, 
                  5e-4]
                  #5e-4] #, 1e-4, 1e-3
l2_reg_const_params = [#0.0, 
                       #0.0, 
                       1e-4] 
                       #0.0]
restore_paths = [#'./debug/m=convNet2 lr=0.0005 do=0.25 l2=0.0 - Continue/Best_Solution',
               #'./debug/m=convNet2 lr=0.0005 do=0.5 l2=0.0 - Continue/Best_Solution',
               './debug/m=convNet2 lr=0.0005 do=0.5 l2=0.0001 - Continue/Best_Solution']
               #'./debug/m=convNet2 lr=0.0005 do=0.75 l2=0.0 - Continue/Best_Solution']

for learning_rate, dropout_keep_prob, l2_reg_const_param, restore_path in zip(learning_rates, dropout_keep_probs, l2_reg_const_params, restore_paths):
                              
    print("dropout="+str(dropout_keep_prob))
    print("learning_rate="+str(learning_rate))
    print("l2="+str(l2_reg_const_param))
    print("training ...")
    
    max_valid_accu, diff = train(model=models.convNet2,
                                 restore_path=restore_path,
                                 learning_rate=learning_rate,
                                 data_augmentation_target=3000, 
                                 dropout_keep_prob=dropout_keep_prob, 
                                 l2_reg_const_param=l2_reg_const_param,
                                 initial_EPOCH=51,
                                 EPOCHS=100)
        
    print("max_valid="+str(max_valid_accu))
    print("diff="+str(diff))
    #print("")
