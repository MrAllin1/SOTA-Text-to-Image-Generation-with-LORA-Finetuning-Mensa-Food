# reference: T2I-CompBench MLLM_eval

import base64
import requests
import re
import argparse
import os
import spacy
from tqdm import tqdm
import json
import time
import sys
import pandas as pd

# OpenAI API Key
api_key = ""  # TODO: Add your API key here
eval_root = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(eval_root, "..", "data")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation by GPT-4V.")
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the CSV file containing image descriptions and paths.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="texture",
        help="Category of the image to be evaluated. eg. color, texture, complex, etc.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index of the image to be evaluated.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Number of images to process.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="gpt4v_results",
        help="Directory to save evaluation results.",
    )
    return parser.parse_args()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def main():
    args = parse_args()
    
    try:
        df = pd.read_csv(args.csv_path)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)

    required_columns = ['description', 'image_path']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: CSV file is missing required column '{col}'")
            sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    gpt4v_record = []
    gpt4v_result = []
    
    # Process images in the specified range
    end_idx = min(args.start + args.step, len(df))
    for i in tqdm(range(args.start, end_idx), desc="GPT-4V processing"):
        row = df.iloc[i]
        prompt_name = row['description']
        image_path = row['image_path']
        image_path = os.path.join(data_root, image_path)
        if not os.path.exists(image_path):
            print(f"Image not found at: {image_path}")
            continue
            
        # Process based on category
        if args.category in ["color", "shape", "texture"]:
            # Use spacy to extract the noun phrase
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(prompt_name)
            num_np = len(list(doc.noun_chunks))
            
            prompt = []  # save as the format as np, noun, adj.
            for chunk in doc.noun_chunks:
                # extract the noun and adj. separately
                chunk_text = chunk.text
                nouns = []
                adjs = []
                for token in chunk:
                    if token.pos_ in ["NOUN", "PROPN"]:
                        nouns.append({
                            'text': token.text,
                            'position': token.i,  # Original position in sentence
                            'is_plural': token.tag_ == 'NNS',
                            'is_proper': token.pos_ == 'PROPN'
                        })
                    elif token.pos_ == 'ADJ' or token.dep_ == 'amod':
                        adjs.append(token.text)
                if nouns:
                    # Prefer: proper nouns > last noun > plural nouns > first noun
                    main_noun = max(nouns, key=lambda x: (
                        2 if x['is_proper'] else 
                        1.5 if x['is_plural'] else
                        -x['position']
                    ))['text']
                else:
                    main_noun = chunk.root.text if chunk.root.pos_ in ['NOUN','PROPN'] else "item"
                noun_targets = [n['text'] for n in nouns]

                prompt_each = [
                    [chunk_text],
                    [main_noun],
                    noun_targets,
                    adjs if adjs else ["no specific attributes"]
                ]
                prompt.append(prompt_each)
            
            question_for_gpt4v = []
            
            for k in range(num_np):
                text = f"You are my assistant to identify any objects and their {args.category} in the image. \
                    According to the image, evaluate if there are '{prompt[k][0]}' in the image. \
                    Give a score from 0 to 100, according the criteria:\n\
                    5: there are {prompt[k][2]}, and {args.category} is good. {prompt[k][3]} are close to describe the contents.\n\
                    4: there are {prompt[k][2]}, {args.category} is generally okay. {prompt[k][3]} are not related.\n\
                    3: there are {prompt[k][2]}, but {args.category} is bad.\n\
                    2: there is {prompt[k][1]}, but not all {prompt[k][2]} appear.\n\
                    1: no {prompt[k][1]} in the image.\n\
                    Provide your analysis and explanation in JSON format with the following keys: score (e.g., 1), \
                    explanation (within 20 words)."
                dic = {"type": "text", "text": text}
                question_for_gpt4v.append(dic)
                
        elif args.category in ["spatial", "3d_spatial"]:
            question_for_gpt4v = []
            num_np = 1
            text = f"You are my assistant to identify objects and their spatial layout in the image. \
                According to the image, evaluate if the text \"{prompt_name}\" is correctly portrayed in the image. \
                Give a score from 0 to 100, according the criteria: \n\
                    5: correct spatial layout in the image for all objects mentioned in the text. \
                    4: basically, spatial layout of objects matches the text. \
                    3: spatial layout not aligned properly with the text. \
                    2: image not aligned properly with the text. \
                    1: image almost irrelevant to the text. \
                    Provide your analysis and explanation in JSON format with the following keys: score (e.g., 2), \
                    explanation (within 20 words)."
            dic = {"type": "text", "text": text}
            question_for_gpt4v.append(dic)   
        
        elif args.category == "numeracy":
            question_for_gpt4v = []
            num_np = 1
            text = f"You are my assistant to identify objects and their quantities in the image. \
            According to the image and your previous answer, evaluate how well the image aligns with the text prompt: \"{prompt_name}\" \
            Give a score from 0 to 100, according the criteria: \n\
                5: correct numerical content in the image for all objects mentioned in the text \
                4: basically, numerical content of objects matches the text \
                3: numerical content not aligned properly with the text \
                2: image not aligned properly with the text \
                1: image almost irrelevant to the text \
                Provide your analysis and explanation in JSON format with the following keys: score (e.g., 2), explanation (within 20 words)."
            dic = {"type": "text", "text": text}
            question_for_gpt4v.append(dic) 
        
        elif args.category == "complex":
            question_for_gpt4v = []
            num_np = 1
            text = "You are my assistant to evaluate the correspondence of the image to a given text prompt. \
                focus on the objects in the image and their attributes (such as color, shape, texture), spatial layout and action relationships. \
                According to the image and your previous answer, evaluate how well the image aligns with the text prompt: \"{prompt_name}\"  \
                        Give a score from 0 to 100, according the criteria: \n\
                        5: the image perfectly matches the content of the text prompt, with no discrepancies. \
                        4: the image portrayed most of the actions, events and relationships but with minor discrepancies. \
                        3: the image depicted some elements in the text prompt, but ignored some key parts or details. \
                        2: the image did not depict any actions or events that match the text. \
                        1: the image failed to convey the full scope in the text prompt. \
                        Provide your analysis and explanation in JSON format with the following keys: score (e.g., 2), explanation (within 20 words)."
            dic = {"type": "text", "text": text}
            question_for_gpt4v.append(dic)
        
        # Getting the base64 string
        try:
            base64_image = encode_image(image_path)
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            continue
            
        # Prepare content for API call
        content_list = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ] + [question_for_gpt4v[num_q] for num_q in range(num_np)]
            
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": content_list
                }
            ],
            "max_tokens": 300
        }
        
        # Make API call with retry logic
        max_attempts = 3
        attempt_count = 0
        average_score = 0
        
        while attempt_count < max_attempts:
            try:
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
                time.sleep(30)
                response.raise_for_status()
                
                # Parse response
                response_data = response.json()
                content = response_data["choices"][0]["message"]["content"]
                
                # Extract scores
                pattern = r'"score": (\d+),'
                score_strings = re.findall(pattern, content)
                scores = [int(score) for score in score_strings]
                average_score = sum(scores) / len(scores) if len(scores) > 0 else 0
                
                break
            except requests.exceptions.RequestException as e:
                print(f"API request failed (attempt {attempt_count + 1}/{max_attempts}): {e}")
                attempt_count += 1
                time.sleep(20)
            except KeyError as e:
                print(f"Error parsing response (attempt {attempt_count + 1}/{max_attempts}): {e}")
                attempt_count += 1
                time.sleep(10)
            except Exception as e:
                print(f"Unexpected error (attempt {attempt_count + 1}/{max_attempts}): {e}")
                attempt_count += 1
                time.sleep(10)
        
        # Save results
        gpt4v_record.append({
            "row_index": i,
            "image_path": image_path,
            "prompt": prompt_name,
            "response": response.json() if 'response' in locals() else None
        })
        
        gpt4v_result.append({
            "row_index": i,
            "image_path": image_path,
            "prompt": prompt_name,
            "score": average_score
        })
        
        # Save intermediate results
        with open(f"{args.output_dir}/gpt4v_record_{args.start}_{end_idx}.json", "w") as f:
            json.dump(gpt4v_record, f)
        
        with open(f"{args.output_dir}/gpt4v_result_{args.start}_{end_idx}.json", "w") as f:
            json.dump(gpt4v_result, f)
        
        # Respect rate limits
        time.sleep(20)
    
    # Calculate and save average score
    if gpt4v_result:
        score_list = [result["score"] for result in gpt4v_result if result["score"] > 0]
        if score_list:
            avg_score = sum(score_list) / len(score_list)
            print(f"\nProcessing complete. Average score: {avg_score:.2f}")
            with open(f"{args.output_dir}/avg_score_{args.start}_{end_idx}.txt", "w") as f:
                f.write(f"Average score: {avg_score:.2f}\n")
                f.write(f"Processed rows: {args.start} to {end_idx-1}\n")
                f.write(f"Total processed: {len(score_list)} images\n")
        else:
            print("\nProcessing complete. No valid scores were obtained.")

if __name__ == "__main__":
    main()