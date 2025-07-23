import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import argparse
import tempfile
import shutil
from pytorch_fid import fid_score
import torchvision.transforms as transforms

eval_root = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(eval_root, "..", "data")


def get_image_transform():

    return transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
    ])


def parse_args():
    parser = argparse.ArgumentParser(description="FID Evaluation between ground truth and generated images.")
    parser.add_argument(
        "--gt_path",
        type=str,
        default="eval/gt_set",
        help="Relative path to ground truth images directory"
    )
    parser.add_argument(
        "--gen_path", 
        type=str,
        default="eval/lora_output",
        help="Relative path to generated images directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/fid_eval",
        help="Directory to save FID evaluation results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for FID calculation (cuda/cpu)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for FID calculation"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading"
    )
    parser.add_argument(
        "--output_index",
        type=int,
        default=0,
        help="Which output image to use from each prediction folder (0, 1, 2, etc.)"
    )
    return parser.parse_args()


def prepare_gt_images(gt_dir, temp_gt_dir):
    gt_files = []
    img_files = [f for f in os.listdir(gt_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    transform = get_image_transform()
    
    for filename in sorted(img_files, key=lambda x: int(x.split('.')[0])):
        src_path = os.path.join(gt_dir, filename)
        dst_path = os.path.join(temp_gt_dir, filename)
        
        try:
            with Image.open(src_path) as img:
                img_transformed = transform(img)
                # Save as JPEG with high quality
                img_transformed.save(dst_path, 'JPEG', quality=95)
            gt_files.append(filename)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    print(f"Prepared {len(gt_files)} ground truth images (resized to 512x512)")
    return gt_files


def prepare_generated_images(gen_dir, temp_gen_dir, output_index=0):
    gen_files = []
    
    # Get all prediction folders and sort them
    prediction_folders = [d for d in os.listdir(gen_dir) 
                         if os.path.isdir(os.path.join(gen_dir, d)) and d.startswith('prediction_')]
    prediction_folders.sort(key=lambda x: int(x.split('_')[1]))
    
    # Get transform pipeline
    transform = get_image_transform()
    
    for i, folder in enumerate(prediction_folders):
        folder_path = os.path.join(gen_dir, folder)
        output_file = f"output_{output_index}.png"
        src_path = os.path.join(folder_path, output_file)
        
        if os.path.exists(src_path):
            # Create a standardized filename for the generated image
            dst_filename = f"generated_{i+1:02d}.png"
            dst_path = os.path.join(temp_gen_dir, dst_filename)
            
            try:
                with Image.open(src_path) as img:
                    img_transformed = transform(img)
                    # Save as JPEG with high quality
                    img_transformed.save(dst_path, 'JPEG', quality=95)
                gen_files.append(dst_filename)
            except Exception as e:
                print(f"Error processing {output_file} in {folder}: {e}")
                continue
        else:
            print(f"Warning: {output_file} not found in {folder}")
    
    print(f"Prepared {len(gen_files)} generated images (resized to 512x512)")
    return gen_files


def calculate_fid_score(gt_dir, gen_dir, args):
    """Calculate FID score between two directories"""
    print("Calculating FID score...")
    try:
        fid_value = fid_score.calculate_fid_given_paths(
            [gt_dir, gen_dir],
            batch_size=args.batch_size,
            device=args.device,
            dims=2048,
            num_workers=args.num_workers
        )
        return fid_value
    except Exception as e:
        print(f"Error calculating FID score: {e}")
        return None


def evaluate_fid_by_pairs(args):
    gt_dir = os.path.join(data_root, args.gt_path)
    gen_dir = os.path.join(data_root, args.gen_path)
    
    if not os.path.exists(gt_dir):
        print(f"Ground truth directory not found: {gt_dir}")
        return
    
    if not os.path.exists(gen_dir):
        print(f"Generated images directory not found: {gen_dir}")
        return
    
    output_dir = os.path.join(eval_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_gt_dir = os.path.join(temp_dir, "gt_images")
        temp_gen_dir = os.path.join(temp_dir, "gen_images")
        os.makedirs(temp_gt_dir, exist_ok=True)
        os.makedirs(temp_gen_dir, exist_ok=True)
        
        # Prepare images
        gt_files = prepare_gt_images(gt_dir, temp_gt_dir)
        gen_files = prepare_generated_images(gen_dir, temp_gen_dir, args.output_index)
        
        if len(gt_files) == 0 or len(gen_files) == 0:
            print("No valid images found for FID calculation")
            return
        
        print(f"Ground truth images: {len(gt_files)}")
        print(f"Generated images: {len(gen_files)}")
        
        # Calculate FID score
        fid_value = calculate_fid_score(temp_gt_dir, temp_gen_dir, args)
        
        if fid_value is not None:
            print(f"\nFID Score: {fid_value:.4f}")
            
            # Save results
            results = {
                "fid_score": fid_value,
                "gt_images_count": len(gt_files),
                "gen_images_count": len(gen_files),
                "output_index_used": args.output_index,
                "gt_files": gt_files,
                "gen_files": gen_files
            }
            
            # Save detailed results
            result_file = os.path.join(output_dir, f"fid_results_output_{args.output_index}.txt")
            with open(result_file, 'w') as f:
                f.write(f"FID Score: {fid_value:.4f}\n")
                f.write(f"Ground Truth Images: {len(gt_files)}\n")
                f.write(f"Generated Images: {len(gen_files)}\n")
                f.write(f"Output Index Used: {args.output_index}\n")
                # f.write(f"\nGround Truth Files:\n")
                # for gt_file in gt_files:
                #     f.write(f"  {gt_file}\n")
                # f.write(f"\nGenerated Files:\n")
                # for gen_file in gen_files:
                #     f.write(f"  {gen_file}\n")
            
            print(f"Results saved to: {result_file}")
        else:
            print("Failed to calculate FID score")


def evaluate_all_outputs(args):
    print("Evaluating FID scores for all available output images...")
    
    gen_dir = os.path.join(data_root, args.gen_path)
    
    # Find the maximum number of output images available
    max_outputs = 0
    prediction_folders = [d for d in os.listdir(gen_dir) 
                         if os.path.isdir(os.path.join(gen_dir, d)) and d.startswith('prediction_')]
    
    for folder in prediction_folders:
        folder_path = os.path.join(gen_dir, folder)
        output_files = [f for f in os.listdir(folder_path) if f.startswith('output_') and f.endswith('.png')]
        if output_files:
            max_index = max([int(f.split('_')[1].split('.')[0]) for f in output_files])
            max_outputs = max(max_outputs, max_index + 1)
    
    print(f"Found up to {max_outputs} output images per prediction folder")
    
    fid_scores = []
    for output_idx in range(max_outputs):
        print(f"\n{'='*50}")
        print(f"Evaluating output_{output_idx}.png")
        print(f"{'='*50}")
        
        # Temporarily change the output_index
        original_output_index = args.output_index
        args.output_index = output_idx
        
        # Calculate FID for this output index
        evaluate_fid_by_pairs(args)
        
        # Restore original setting
        args.output_index = original_output_index
    
    print(f"\n{'='*50}")
    print("FID Evaluation Complete for all outputs")
    print(f"{'='*50}")


def main():
    args = parse_args()
    
    print("\nFID Evaluation")
    print(f"  Ground Truth Path: {os.path.join(data_root, args.gt_path)}")
    print(f"  Generated Images Path: {os.path.join(data_root, args.gen_path)}")
    print(f"  \t using output_{args.output_index} series")
    print(f"  Output Directory: {os.path.join(eval_root, args.output_dir)}\n")

    # evaluate_all_outputs(args)
    evaluate_fid_by_pairs(args)


if __name__ == "__main__":
    main()