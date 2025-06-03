python lamo_plas.py \
--gpu 0 \
--model LaMO_Structured_Mesh_2D_unshared \
--n-hidden 128 \
--n-heads 8 \
--n-layers 8 \
--lr 0.001 \
--max_grad_norm 0.1 \
--batch-size 8 \
--slice_num 64 \
--unified_pos 0 \
--ref 8 \
--eval 0 \
--save_name LaMO_Plas

