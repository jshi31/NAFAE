CUDA_VISIBLE_DEVICES=0 python3 model.py --phase val  --checksession 256 --checkepoch 8 --checkbatch 41 --dropout_rate 0.1 --sample_num 5 --fix_seg_len  --val_vis_freq 1000 --lr_decay_gamma 0.1 --lr_decay_step 20 --workers 4 --epoch 30 --lr 0.001 --lam 0.015 --Delta 5  --statement finalmodel --cuda  --weight_decay 0.00001 --vis_lam 1

