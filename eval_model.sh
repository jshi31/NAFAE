CUDA_VISIBLE_DEVICES=0 python model.py --phase val  --checksession 256 --checkepoch 8 --checkbatch 41 --val_vis_freq 1000 --workers 4 --epoch 30 --lr 0.001 --Delta 5 --vis_lam 1 --statement finalmodel --cuda

