CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
--config_file config/default_config \
--main_process_port 30001 \
runner/decoder_train.py \
--config config/decoder_train.yaml