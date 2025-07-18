# data_loader.py

import os
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

def make_transforms(resolution: int):
    return transforms.Compose([
        transforms.Resize(resolution),
        transforms.CenterCrop(resolution),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

def get_dataloader(
    csv_file: str,
    tokenizer,
    resolution: int,
    batch_size: int,
    num_workers: int = 4,
    split: str = "train",
    shuffle: bool = True
):
    ds = load_dataset("csv", data_files={split: csv_file}, split=split)
    print(f"👀 Dataset size ({split} split): {len(ds)} examples")
    base_dir = os.path.dirname(csv_file)
    tfm = make_transforms(resolution)
    max_length = 77

    def preprocess(example):
        img_path = os.path.join(base_dir, example["image_path"])
        img = Image.open(img_path).convert("RGB")
        pixel_values = tfm(img)
        input_ids = tokenizer(
            example["description"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        ).input_ids
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    ds = ds.map(preprocess,remove_columns=ds.column_names,num_proc=1,)
    ds.set_format(type="torch", columns=["pixel_values", "input_ids"])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_dataloader_stream(
    csv_file: str,
    tokenizer,
    resolution: int,
    batch_size: int,
    num_workers: int = 1,
    split: str = "train"
):
    ds = load_dataset("csv", data_files={split: csv_file}, split=split, streaming=True)
    base_dir = os.path.dirname(csv_file)
    tfm = make_transforms(resolution)
    max_length = 77

    def preprocess(example):
        img_path = os.path.join(base_dir, example["image_path"])
        img = Image.open(img_path).convert("RGB")
        pixel_values = tfm(img)
        input_ids = tokenizer(
            example["description"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        ).input_ids[0]
        return {"pixel_values": pixel_values, "input_ids": input_ids}
    
    ds = ds.map(preprocess, remove_columns=ds.column_names)
    ds = ds.with_format("torch")

    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers)