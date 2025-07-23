# reference: T2I-CompBench MLLM_eval

import base64
import requests
import re
import argparse
import os
import spacy
import json
import time
import sys
from openai import OpenAI
import pandas as pd
from tqdm import tqdm

eval_root = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(eval_root, "..", "data")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation by GPT-4V.")
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="API key for QWen",
    )
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
        "--evallen",
        type=int,
        default=-1,
        help="Number of images to be evaluated.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="agent_results",
        help="Directory to save evaluation results.",
    )
    return parser.parse_args()


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# ref https://help.aliyun.com/zh/model-studio/qwen-vl-compatible-with-openai?spm=a2c4g.11186623.0.i1
def call_bailian_api(args, base64_image, question_prompt):
    client = OpenAI(
        api_key=args.api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        model="qwen-vl-plus",
        messages=[
            {
              "role": "user",
              "content": [
                {
                  "type": "text",
                  "text": question_prompt
                },
                {
                  "type": "image_url",
                  "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                  }
                },
              ]
            }
          ]
        )

    response = completion.model_dump_json()
    return json.loads(response)


def main():
    args = parse_args()
    
    try:
        df = pd.read_csv(os.path.join(data_root, args.csv_path))
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)

    required_columns = ['description', 'image_path']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: CSV file is missing required column '{col}'")
            sys.exit(1)

    # Create output directory
    output_dir = os.path.join(eval_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    agent_record = []
    agent_result = []
    
    # Process images in the specified range
    if args.evallen == -1:
        args.evallen = len(df)
    for i in tqdm(range(args.evallen), desc="Agent processing"):
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
            
            question_for_agent = []
            
            for i, k in enumerate(range(num_np)):
                text = f"""
                Evaluate if there are '{prompt[k][0]}' in the image according the criteria:\n\
                A: there are {prompt[k][2]}, and {args.category} is good. {prompt[k][3]} are appropriate for the contents.\n\
                B: there are {prompt[k][2]}, but {args.category} is bad.\n\
                C: there is {prompt[k][1]}, but not all {prompt[k][2]} appear.\n\
                D: no {prompt[k][1]} in the image.\n\
                Provide a score (0-100) and explanation in JSON format(omit all formatting chars like \'\n\') \
                eg: \"obj_id\": {i+1}, \"score\": 82, \"explanation\": \"...(within 30 words)\".\n
                """
                    
                dic = {"type": "text", "text": text}
                question_for_agent.append(dic)
                
        elif args.category in ["spatial", "3d_spatial"]:
            question_for_agent = []
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
            question_for_agent.append(dic)   
        
        elif args.category == "numeracy":
            question_for_agent = []
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
            question_for_agent.append(dic) 
        
        elif args.category == "complex":
            question_for_agent = []
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
            question_for_agent.append(dic)
        
        # Getting the base64 string
        try:
            base64_image = encode_image(image_path)
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            continue
            
        
        # Make API call with retry logic
        question_prompt = 'You are my assistant to identify any objects and their {args.category} in the image.\n' + \
            '\n'.join([question_for_agent[num_q]['text'] for num_q in range(num_np)])
        max_attempts = 3
        attempt_count = 0
        average_score = 0
        while attempt_count < max_attempts:
            try:
                response = call_bailian_api(args, base64_image, question_prompt)
                if not response:
                    raise ValueError("Empty API response")
                time.sleep(1)
                
                # Parse response
                # response_data = response.json()
                content = response["choices"][0]["message"]["content"]
                
                # Extract scores
                pattern = r'"score": (\d+),'
                score_strings = re.findall(pattern, content)
                scores = [int(score) for score in score_strings]
                average_score = sum(scores) / len(scores) if len(scores) > 0 else 0
                
                break
            except Exception as e:
                print(f"Error (attempt {attempt_count + 1}/{max_attempts}): {e}")
                attempt_count += 1
                time.sleep(10)
        
        # Save results
        agent_record.append({
            "row_index": i,
            "image_path": image_path,
            "prompt": prompt_name,
            "response": response if 'response' in locals() else None
        })
        
        agent_result.append({
            "row_index": i,
            "image_path": image_path,
            "prompt": prompt_name,
            "score": average_score
        })
        
        # Save intermediate results
        with open(f"{output_dir}/agent_record_{args.evallen}.json", "w") as f:
            json.dump(agent_record, f)
        
        with open(f"{output_dir}/agent_result_{args.evallen}.json", "w") as f:
            json.dump(agent_result, f)
        
        # Respect rate limits
        time.sleep(10)
    
    # Calculate and save average score
    if agent_result:
        score_list = [result["score"] for result in agent_result if result["score"] > 0]
        if score_list:
            avg_score = sum(score_list) / len(score_list)
            print(f"\nProcessing complete. Average score: {avg_score:.2f}")
            with open(f"{output_dir}/avg_score_{args.evallen}.txt", "w") as f:
                f.write(f"Average score: {avg_score:.2f}\n")
                # f.write(f"Processed rows: {args.evallen}\n")
                f.write(f"Total processed: {len(score_list)} images\n")
        else:
            print("\nProcessing complete. No valid scores were obtained.")

if __name__ == "__main__":
    main()