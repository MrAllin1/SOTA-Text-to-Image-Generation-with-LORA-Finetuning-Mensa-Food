# i2t similarity
# ref https://github.com/openai/CLIP
# ref https://github.com/Karine-Huang/T2I-CompBench

import os
import torch
import clip
import spacy
import json
import tqdm
import pandas as pd
import sys
import argparse
from PIL import Image

nlp = spacy.load('en_core_web_sm')
eval_root = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(eval_root, "..", "data")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Relative path to the CSV file containing image descriptions and paths.",
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
        default=100,
        help="Number of samples to evaluate"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    try:
        df = pd.read_csv(os.path.join(data_root, args.csv_path))
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)

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
    with open(f'{output_dir}/score_avg.txt', 'w') as f:
        f.write('score avg: ' + str(score / len(sim_dict)))
    print("score avg: ", score / len(sim_dict))


if __name__ == "__main__":
    main()


