python lamo_darcy.py \
--gpu 0 \
--model LaMO_Regular_Grid \
--n-hidden 128 \
--n-heads 8 \
--n-layers 8 \
--lr 0.0005 \
--max_grad_norm 0.1 \
--batch-size 4 \
--slice_num 64 \
--unified_pos 1 \
--ref 8 \
--eval 0 \
--downsample 5 \
--save_name LaMO_Darcy \
--num_scales 1 \
--patch_sizes 2 \
--embed_dims 64 \
--depths 8 \
--H_padded 88 \
--W_padded 88 \
--num_heads 8 \
--mlp_ratios 4 



