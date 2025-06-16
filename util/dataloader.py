import torch
import glob
from datasets import load_dataset, Features, Image as HFImage, Value
from torch.utils.data import DataLoader
from janus.models import VLChatProcessor
import torchvision.transforms as pth_transforms
from PIL import Image
import io
import torchvision
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


vl_chat_processor = VLChatProcessor.from_pretrained("/data1/ckpts/deepseek-ai_/Janus-Pro-1B")


imagenet_transform_train = pth_transforms.Compose([
    pth_transforms.Resize(384, max_size=None),
    pth_transforms.RandomHorizontalFlip(p=0.5),
    pth_transforms.CenterCrop(384),
    pth_transforms.ToTensor(),
    pth_transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # 将单通道转换为三通道
])

def collate_fn_imagenet_wds(batch):
    pixel_values = []
    labels = []
    for sample in batch:
        img = imagenet_transform_train(sample["jpg"])
        if img.shape[0] != 3:
            print("skip", img.shape)
            continue
        labels.append(sample["cls"])
        pixel_values.append(img)
    
    pixel_values = torch.stack(pixel_values, dim=0)
    labels = torch.Tensor(labels)
    # print(pixel_values.shape, labels.shape)
    
    return {"pixel_values": pixel_values, "labels": labels}

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
        valid_samples = []
        for sample in batch:
            try:
                # Handle raw bytes if not already decoded
                jpg_data = sample["jpg"]
                if isinstance(jpg_data, bytes):
                    try:
                        img = Image.open(io.BytesIO(jpg_data))
                        img.load()  # Force load to validate
                    except Exception as e:
                        print(f"Skipping corrupted image: {e}")
                        continue
                elif isinstance(jpg_data, Image.Image):
                    img = jpg_data
                else:
                    print(f"Unexpected image type: {type(jpg_data)}, skipping")
                    continue
                    
                text = sample["txt"]
                valid_samples.append({"jpg": img, "txt": text})
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue
        
        if len(valid_samples) == 0:
            return {"pixel_values": torch.empty(0, 3, 224, 224), "texts": torch.empty(0, 0, dtype=torch.long)}
            
        imgs = [s["jpg"] for s in valid_samples]
        texts = [s["txt"] for s in valid_samples]
        
        pixel_values = vl_chat_processor.image_processor(imgs, return_tensors="pt").pixel_values
        processed_text = vl_chat_processor.tokenizer(texts, return_tensors="pt", padding=True, truncation=False)
        input_ids = processed_text.input_ids
        attention_mask = processed_text.attention_mask

        return {"pixel_values": pixel_values, "texts": input_ids, "attention_mask": attention_mask}
    except Exception as e:
        print(f"Error in collate_fn_journeydb: {e}")
        return {"pixel_values": torch.empty(0, 3, 224, 224), "texts": torch.empty(0, 0, dtype=torch.long)}

def get_dataloader(config):
    if config.name == "blip":
        file_pattern = "sa_{:06d}.tar"
        files_to_process = ["BLIP3o-Pretrain-Long-Caption/" + file_pattern.format(i) for i in range(0, 400)]

        ds_BLIP3o = load_dataset("/data1/LargeData/BLIP3o", split="train", data_files=files_to_process, num_proc=64)
        dataloader = DataLoader(ds_BLIP3o, batch_size=config.batch_size, collate_fn=collate_fn_blip, shuffle=True)
    elif config.name == "journeydb":
        data_files = glob.glob("/data1/LargeData/BLIP3o-Pretrain-JourneyDB/*.tar")
        
        # Define features without automatic image decoding
        features = Features({
            "jpg": Value("binary"),  # Keep as binary to handle manually
            "txt": Value("string")
        })
        
        ds_journeydb = load_dataset(
            "webdataset",
            data_files = data_files,
            split      = "train",
            num_proc   = 128,
            features   = features  # Disable automatic image decoding
        )
        dataloader = DataLoader(ds_journeydb, batch_size=config.batch_size, collate_fn=collate_fn_journeydb, shuffle=True)
    elif config.name == "imagenet":
        if config.siglip_preprocess:
            processor = VLChatProcessor.from_pretrained("/data1/ckpts/deepseek-ai_/Janus-Pro-1B").image_processor
            imagenet_transform_train = lambda x: processor(images=[x], return_tensors="pt").pixel_values[0]
            imagenet_transform_val = imagenet_transform_train
        else:
            imagenet_transform_train = pth_transforms.Compose([
                pth_transforms.Resize(config.img_size, max_size=None),
                pth_transforms.RandomHorizontalFlip(p=0.5),
                pth_transforms.CenterCrop(config.img_size),
                pth_transforms.ToTensor(),
            ])

            imagenet_transform_val = pth_transforms.Compose([
                pth_transforms.Resize(config.img_size, max_size=None),
                pth_transforms.CenterCrop(config.img_size),
                pth_transforms.ToTensor(),
            ])

        imagenet_data_train = torchvision.datasets.ImageFolder(config.train_path, transform=imagenet_transform_train)

        dataloader = torch.utils.data.DataLoader(
            imagenet_data_train,
            batch_size  = config.batch_size,
            shuffle     = True,
            num_workers = config.num_workers,
            drop_last   = True,
        )
        if getattr(config, "val_path", None) is None:
            return dataloader
        else:
            imagenet_data_val = torchvision.datasets.ImageFolder(config.val_path, transform=imagenet_transform_val)
            dataloader_val = torch.utils.data.DataLoader(
                imagenet_data_val,
                batch_size  = config.batch_size,
                shuffle     = False,
                num_workers = config.num_workers,
            )
            return dataloader, dataloader_val
    elif config.name == "imagenet_wds":
        data_files = glob.glob(os.path.join(config.train_path, "*.tar"))

        imagenet_wds_train = load_dataset(
            "webdataset",
            data_files = data_files,
            split      = "train",
            num_proc   = 8,
        )
        dataloader = DataLoader(
            imagenet_wds_train,
            batch_size  = config.batch_size,
            collate_fn  = collate_fn_imagenet_wds,
            shuffle     = True,
            num_workers = config.num_workers,
            drop_last   = True,
        )
        if getattr(config, "val_path", None) is None:
            return dataloader
        else:
            raise NotImplementedError

class SafeDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.batch_size = dataloader.batch_size
        
    def __iter__(self):
        for batch in self.dataloader:
            try:
                # Skip empty batches
                if len(batch["pixel_values"]) == 0:
                    print("Skipping empty batch")
                    continue
                yield batch
            except Exception as e:
                print(f"⚠️ 跳过损坏的批次: {e}")
                continue
                
    def __len__(self):
        return len(self.dataloader)