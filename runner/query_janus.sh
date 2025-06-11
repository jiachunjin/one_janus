CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
--config_file config/ds_config \
--num_processes 4 \
--main_process_port 30002 \
runner/query_janus.py \
--config config/query_janus.yaml