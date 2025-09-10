# mensa_t2i
**SoTA Text-to-Image Adapting and Finetuning for University Mensa Food**  

---

## 📖 Overview
This project fine-tunes **Stable Diffusion** with **LoRA adapters** and a custom `MensaFood` token to generate photorealistic previews of meals served in the University of Freiburg mensa.  
Instead of showing only a text menu, the model creates images of the exact dishes you’ll find in the cafeteria.  

---

## 🚀 Training
To start a training job on the cluster:  

```bash
chmod +x train_lora_job.sh
sbatch train_lora_job.sh
```

Check job status:  
```bash
squeue --me
sacct -j <JOB_ID> --format=JobID,State,Elapsed,MaxRSS
```

---

## 🔮 Prediction
Generate images from a prompt using a trained LoRA adapter:  

```bash
cd lora
python predict_lora.py   --prompt "Mensafood Breaded pork schnitzel or vegetable schnitzel, roast gravy, and French fries"   --num_images 2   --height 512   --width 512   --lora_weights_dir ./lora-adapters-forth-train/checkpoint-30000
```

---

## 📏 Evaluation
We use both quantitative metrics and simple NLP checks to evaluate outputs.  

### LLM-based Evaluation
Install spaCy for noun-phrase extraction:  
```bash
python -m spacy download en_core_web_sm
```

Run evaluation scripts with:
- **CLIP similarity** – visual fidelity  
- **QWen scores** – text alignment  
- **FID** – distribution matching  

---

## 📊 Example Results

You can check some results in the research poster below.

---

## 📄 Poster
👉 [View the Research Poster (PDF)](./poster.pdf)  

---
