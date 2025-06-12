CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
--config_file config/default_config \
--main_process_port 30001 \
--num_processes 8 \
runner/siglip2pix.py \
--config config/vit_pixel_decoder.yaml