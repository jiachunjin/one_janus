import torch
import glob
from datasets import load_dataset
from torch.utils.data import DataLoader
from janus.models import VLChatProcessor

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


vl_chat_processor = VLChatProcessor.from_pretrained("/data1/ckpts/deepseek-ai_/Janus-Pro-1B")

def collate_fn_blip(batch):
    pixel_values = []
    texts = []
    for sample in batch:
        img = sample["jpg"]
        text = sample["txt"]

        input_ids = vl_chat_processor.tokenizer(text, return_tensors="pt").input_ids[0]
        if input_ids.shape[0] > 150:
            continue

        pixel_values.append(vl_chat_processor.image_processor([img], return_tensors="pt").pixel_values)

        input_ids = torch.cat([input_ids, 100002 * torch.ones(150 - input_ids.shape[0], dtype=torch.long)], dim=0)
        texts.append(input_ids)
        
    pixel_values = torch.cat(pixel_values, dim=0)
    texts = torch.stack(texts, dim=0)

    return {"pixel_values": pixel_values, "texts": texts}

def collate_fn_journeydb(batch):
    try:
        imgs = [sample["jpg"] for sample in batch]
        texts = [sample["txt"] for sample in batch]
        pixel_values = vl_chat_processor.image_processor(imgs, return_tensors="pt").pixel_values
        input_ids = vl_chat_processor.tokenizer(texts, return_tensors="pt", padding=True, truncation=False, max_length=128).input_ids

        return {"pixel_values": pixel_values, "texts": input_ids}
    except Exception as e:
        print(f"Error in collate_fn_journeydb: {e}")
        return None

def get_dataloader(config):
    if config.name == "blip":
        file_pattern = "sa_{:06d}.tar"
        files_to_process = ["BLIP3o-Pretrain-Long-Caption/" + file_pattern.format(i) for i in range(0, 400)]

        ds_BLIP3o = load_dataset("/data1/LargeData/BLIP3o", split="train", data_files=files_to_process, num_proc=64)
        dataloader = DataLoader(ds_BLIP3o, batch_size=config.batch_size, collate_fn=collate_fn_blip, shuffle=True)
    elif config.name == "journeydb":
        data_files = glob.glob("/data1/LargeData/BLIP3o-Pretrain-JourneyDB/*.tar")
        ds_journeydb = load_dataset(
            "webdataset",
            data_files = data_files,
            split      = "train",
            num_proc   = 128
        )
        dataloader = DataLoader(ds_journeydb, batch_size=config.batch_size, collate_fn=collate_fn_journeydb, shuffle=True)


    return dataloader
