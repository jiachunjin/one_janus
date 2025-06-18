CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
--config_file config/default_config \
--main_process_port 30002 \
--num_processes 2 \
runner/siglip2pix.py \
--config config/kuaishou/vit_pixel_decoder.yaml