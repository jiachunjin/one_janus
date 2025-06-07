CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
--config_file config/ds_config \
--main_process_port 30002 \
runner/query_janus.py \
--config config/query_janus.yaml