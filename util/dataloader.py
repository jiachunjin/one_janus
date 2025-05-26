import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from janus.models import VLChatProcessor

vl_chat_processor = VLChatProcessor.from_pretrained("/data1/ckpts/deepseek-ai_/Janus-Pro-1B")

def collate_fn(batch):
    pixel_values = []
    texts = []
    for sample in batch:
        img = sample["jpg"]
        text = sample["txt"]

        input_ids = vl_chat_processor.tokenizer(text, return_tensors="pt").input_ids[0]
        if input_ids.shape[0] > 150:
            continue

        pixel_values.append(vl_chat_processor.image_processor([img], return_tensors="pt").pixel_values)

        input_ids = torch.cat([input_ids, 100002*torch.ones(150 - input_ids.shape[0], dtype=torch.long)], dim=0)
        # print(input_ids.shape)
        texts.append(input_ids)
        
        # print(text)
    # print(pixel_values)
    pixel_values = torch.cat(pixel_values, dim=0)
    texts = torch.stack(texts, dim=0)
    # print(pixel_values.shape)
    # print(len(batch))
    # img = batch[0]["jpg"]
    # text = batch[0]["txt"]
    # pixel_values = [vl_chat_processor.image_processor(x["jpg"], return_tensors="pt").pixel_values for x in batch]
    # print(pixel_values.shape)
    # pixel_values = [sample["pixel_values"] for sample in batch]
    # print(input_ids)
    return {"pixel_values": pixel_values, "texts": texts}

def get_dataloader(config):
    file_pattern = "sa_{:06d}.tar"
    files_to_process = ["BLIP3o-Pretrain-Long-Caption/" + file_pattern.format(i) for i in range(0, 100)]

    ds_BLIP3o = load_dataset("/data1/LargeData/BLIP3o", split="train", data_files=files_to_process, num_proc=64)
    # ds_BLIP3o = load_dataset("/data1/LargeData/BLIP3o", data_files={"train": path}, split="train")
    # iterable_dataset = ds_BLIP3o.to_iterable_dataset(num_shards=64)
    dataloader = DataLoader(ds_BLIP3o, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=True)

    return dataloader


# def get_dataloader(config):
#     ds_BLIP3o = load_dataset("/data1/LargeData/BLIP3o", split="train", streaming=True)

#     # dataloader = ds_BLIP3o.iter(config.batch_size)

#     return dataloader