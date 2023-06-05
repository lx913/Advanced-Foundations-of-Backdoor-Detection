python3 train.py -model resnet -lr_C 0.01 -bs 128 -n_iters 30 -pc 0.1 -cross_ratio 2 -model_num 10 -target_label 0
python3 fine_tuning.py -model resnet -lr_C 0.01 -bs 128 -n_iters 10 -pc 0.1 -cross_ratio 2 -model_num 10 -target_label 0 -ft_ratio 0.05
python3 train.py -model senet -lr_C 0.01 -bs 128 -n_iters 30 -pc 0.1 -cross_ratio 2 -model_num 10 -target_label 0
python3 fine_tuning.py -model senet -lr_C 0.01 -bs 128 -n_iters 10 -pc 0.1 -cross_ratio 2 -model_num 10 -target_label 0 -ft_ratio 0.05
