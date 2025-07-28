# i2t similarity
# ref https://github.com/openai/CLIP
# ref https://github.com/Karine-Huang/T2I-CompBench

import os
import torch
import clip
import spacy
import json
import pandas as pd
import sys
import argparse
from tqdm import tqdm
from PIL import Image

nlp = spacy.load('en_core_web_sm')
eval_root = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(eval_root, "..", "data")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Create mutually exclusive group for dataset selection
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        "--csv_path",
        type=str,
        help="Relative path to the CSV file containing descriptions and img_paths (for training dataset evaluation).",
    )
    dataset_group.add_argument(
        "--test_dataset",
        action="store_true",
        help="Evaluate test dataset from data/eval/lora_output"
    )
    
    parser.add_argument(
        "--outpath",
        type=str,
        default=None,
        required=True,
        help="Relative path to save output scores"
    )
    parser.add_argument(
        "--complex",
        type=bool,
        default=False,
        help="To evaluate on samples in complex category or not"
    )
    parser.add_argument(
        "--evallen",
        type=int,
        default=-1,
        help="Number of samples to evaluate"
    )
    args = parser.parse_args()
    return args


def get_similarity(args, prompt, image):

    if (args.complex):
        doc = nlp(prompt)
        prompt_without_adj = ' '.join([token.text for token in doc if token.pos_ != 'ADJ']) #remove adj
        text = clip.tokenize(prompt_without_adj).to(device)
    else:
        text = clip.tokenize(prompt).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image.to(device))
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        # Calculate the cosine similarity between the image and text features
        cosine_similarity = (image_features @ text_features.T).squeeze().item()
    similarity = cosine_similarity

    return similarity


def score_traindataset(args):
    try:
        df = pd.read_csv(os.path.join(data_root, args.csv_path))
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)
    if args.evallen == -1:
        args.evallen = len(df)
    print(f"Ready to evaluate first {args.evallen} samples in {args.csv_path}")

    cnt = 0
    sim_list = []
    for i in tqdm(range(args.evallen), desc="CLIP sim processing"):
        row = df.iloc[i]
        prompt = row['description']
        image_path = row['image_path']
        image_path = os.path.join(data_root, image_path)
        if not os.path.exists(image_path):
            print(f"Image not found at: {image_path}")
            continue
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        similarity = get_similarity(args, prompt, image)

        cnt += 1
        if (cnt % 100 == 0):
            print(f"CLIP image-text:{cnt} prompt(s) have been processed!")
        sim_list.append(similarity)

    #save
    output_dir = os.path.join(eval_root, args.outpath)
    sim_dict = []
    for i in range(len(sim_list)):
        tmp = {}
        tmp['question_id'] = i
        tmp["answer"] = sim_list[i]
        sim_dict.append(tmp)
    json_file = json.dumps(sim_dict)
    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/vqa_result_{args.evallen}.json', 'w') as f:
        f.write(json_file)
    print(f"save to {output_dir}/vqa_result_{args.evallen}.json")

    # score avg
    score = 0
    for i in range(len(sim_dict)):
        score += float(sim_dict[i]['answer'])
    with open(f'{output_dir}/score_avg_{args.evallen}.txt', 'w') as f:
        f.write('score avg: ' + str(score / len(sim_dict)))
    print("score avg: ", score / len(sim_dict))


def score_testdataset(args):
    # Path to the test dataset directory
    test_data_path = os.path.join(data_root, "eval", "lora-v3_output")
    
    if not os.path.exists(test_data_path):
        print(f"Test data path not found: {test_data_path}")
        sys.exit(1)
    
    # Get all subdirectories
    subdirs = [d for d in os.listdir(test_data_path) if os.path.isdir(os.path.join(test_data_path, d))]
    subdirs.sort(key=lambda x: int(x.split("_")[-1]))  # Sort for consistent ordering

    if args.evallen != -1:
        assert args.evallen < len(subdirs), f"evallen {args.evallen} exceeds available subdirs {len(subdirs)}"
        subdirs = subdirs[:args.evallen]
    print(f"Ready to evaluate {len(subdirs)} test samples from {test_data_path}")
    
    cnt = 0
    sim_list = []
    processed_samples = []
    for subdir in tqdm(subdirs, desc="CLIP sim processing test dataset"):
        subdir_path = os.path.join(test_data_path, subdir)
        
        # Read prompt from prompt.txt
        prompt_file = os.path.join(subdir_path, "prompt.txt")
        if not os.path.exists(prompt_file):
            print(f"Prompt file not found: {prompt_file}")
            continue
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
        except Exception as e:
            print(f"Error reading prompt file {prompt_file}: {e}")
            continue
        
        # Read image output_0.png
        image_file = os.path.join(subdir_path, "output_0.png")
        if not os.path.exists(image_file):
            print(f"Image file not found: {image_file}")
            continue
        try:
            image = preprocess(Image.open(image_file)).unsqueeze(0).to(device)
        except Exception as e:
            print(f"Error loading image {image_file}: {e}")
            continue
        
        # Calculate similarity
        similarity = get_similarity(args, prompt, image)
        
        cnt += 1
        if cnt % 10 == 0:
            print(f"CLIP image-text: {cnt} test samples have been processed!")
        
        sim_list.append(similarity)
        processed_samples.append({
            'folder': subdir,
            'prompt': prompt,
            'similarity': similarity
        })
    
    if not sim_list:
        print("No valid samples found to evaluate!")
        return
    
    # Save results
    output_dir = os.path.join(eval_root, args.outpath)
    sim_dict = []
    for i, sample in enumerate(processed_samples):
        tmp = {
            'question_id': i,
            'folder': sample['folder'],
            'prompt': sample['prompt'],
            'answer': sample['similarity']
        }
        sim_dict.append(tmp)
    
    json_file = json.dumps(sim_dict, indent=2, ensure_ascii=False)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    with open(f'{output_dir}/test_vqa_result_{len(sim_list)}.json', 'w', encoding='utf-8') as f:
        f.write(json_file)
    print(f"Detailed results saved to {output_dir}/test_vqa_result_{len(sim_list)}.json")
    
    # Calculate and save average score
    score_avg = sum(sim_list) / len(sim_list)
    with open(f'{output_dir}/test_score_avg_{len(sim_list)}.txt', 'w') as f:
        f.write(f'test score avg: {score_avg}')
    print(f"Test score average: {score_avg}")
    
    return score_avg


if __name__ == "__main__":
    args = parse_args()

    if args.test_dataset:
        score_testdataset(args)
    else:
        score_traindataset(args)


