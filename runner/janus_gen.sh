CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 accelerate launch \
--config_file config/default_config \
--main_process_port 30002 \
runner/janus_gen.py \
--config config/janus_gen.yaml