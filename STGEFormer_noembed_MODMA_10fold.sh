
# The script to run Exp of MODMA_10fold

nohup python -u MODMA_10fold.py  \
  --seq_len 501 \
  --enc_in 128 \
  --d_model 128 \
  --n_head 128 \
  --e_layers 4 \
  --d_ff 2048 \
  --dropout 0.1 \
  --activation 'gelu' \
  --output_attention \
  --use_norm True \
  --train_epochs 100 \
  --batch_size 8 \
  --learning_rate 0.000005 \
  --use_gpu true \
  --gpu 0 \
  --device 0 > STGEFormer_noembed_MODMA_10fold_lr5e-6_bs8.log 2>&1




