python3 train_atk_model.py -epoch 30 -lr 0.01 -batchsize 128 -model resnet -modelnum 10 -atk badnet -poisoned_portion 0.1 -patch_size 5 -trigger_label 0
python3 fine_tuning.py -epoch 10 -lr 0.01 -batchsize 128 -model resnet -modelnum 10 -atk badnet -poisoned_portion 0.1 -patch_size 5 -trigger_label 0 -clean_rate 0.05
python3 train_atk_model.py -epoch 30 -lr 0.01 -batchsize 128 -model senet -modelnum 10 -atk badnet -poisoned_portion 0.1 -patch_size 5 -trigger_label 0
python3 fine_tuning.py -epoch 10 -lr 0.01 -batchsize 128 -model senet -modelnum 10 -atk badnet -poisoned_portion 0.1 -patch_size 5 -trigger_label 0 -clean_rate 0.05
