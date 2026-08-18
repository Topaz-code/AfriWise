import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🌍 AfriWise: Fine-Tuning Phi-3 Mini for Southern Nigerian Languages\n",
                "\n",
                "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Emmanuel-O/Africa_AI_Project/blob/main/notebooks/AfriWise_Finetune.ipynb)\n",
                "[![Unsloth](https://img.shields.io/badge/Unsloth-2x_Faster_LLM_Finetuning-blue)](https://github.com/unslothai/unsloth)\n",
                "\n",
                "Welcome to the **AfriWise** training notebook! In this notebook, we fine-tune Microsoft's **Phi-3 Mini (3.8B)** / **Phi-3.5 Mini** using **Unsloth** QLoRA 4-bit quantization on our curated 22,900+ multi-task dataset spanning:\n",
                "- **Ibibio**: Lexicon, grammar classifications, and conversational idioms\n",
                "- **Igbo**: Central Igbo vocabulary, grammatical classifications, and cultural proverbs (ilu)\n",
                "- **Edo / Bini**: Ancient kingdom folklore, historical timelines, and linguistic proverbs\n",
                "\n",
                "### 🚀 Training Pipeline:\n",
                "1. **Unsloth 4-bit Load**: Load `unsloth/Phi-3-mini-4k-instruct` in 4-bit (<4GB VRAM footprint).\n",
                "2. **LoRA Adapters**: Target `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.\n",
                "3. **ChatML Dataset**: Load `training_dataset_chatml.jsonl` with AfriWise system prompt.\n",
                "4. **SFTTrainer**: Supervised fine-tuning with cosine LR schedule, AdamW 8-bit, and mixed precision (`fp16`/`bf16`).\n",
                "5. **Inference Verification**: Test cultural storytelling & translation before and after training.\n",
                "6. **GGUF Export**: Export 4-bit quantized GGUF (`q4_k_m`) directly for local **Ollama** deployment."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 📦 Step 1: Install Unsloth & Dependencies\n",
                "We install Unsloth (which accelerates training by 2x-5x and cuts memory usage by 70%), along with `xformers`, `trl`, `peft`, `accelerate`, and `bitsandbytes`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%%capture\n",
                "import torch\n",
                "major_version, minor_version = torch.cuda.get_device_capability()\n",
                "\n",
                "# Install Unsloth & dependencies for Colab\n",
                "!pip install --no-deps \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\"\n",
                "!pip install --no-deps \"xformers<0.0.27\" \"trl<0.9.0\" peft accelerate bitsandbytes\n",
                "!pip install datasets triton\n",
                "\n",
                "print(f\"CUDA Available: {torch.cuda.is_available()}\")\n",
                "if torch.cuda.is_available():\n",
                "    print(f\"Device: {torch.cuda.get_device_name(0)}\")\n",
                "    print(f\"CUDA Capability: {major_version}.{minor_version}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🧠 Step 2: Load Phi-3 Mini 4-bit with Unsloth\n",
                "We load `unsloth/Phi-3-mini-4k-instruct` with 4-bit quantization. This allows training on a free Google Colab T4 GPU (16GB VRAM) using less than 6GB of VRAM."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from unsloth import FastLanguageModel\n",
                "import torch\n",
                "\n",
                "max_seq_length = 2048  # Supports RoPE scaling internally\n",
                "dtype = None           # Auto-detect: Float16 for Tesla T4, Bfloat16 for Ampere/Ada/Hopper\n",
                "load_in_4bit = True    # 4-bit quantization saves ~70% VRAM\n",
                "\n",
                "model, tokenizer = FastLanguageModel.from_pretrained(\n",
                "    model_name = \"unsloth/Phi-3-mini-4k-instruct\",\n",
                "    max_seq_length = max_seq_length,\n",
                "    dtype = dtype,\n",
                "    load_in_4bit = load_in_4bit,\n",
                ")\n",
                "print(\"Base model and tokenizer loaded successfully!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## ⚙️ Step 3: Add LoRA / QLoRA Adapters\n",
                "We configure LoRA parameter-efficient fine-tuning on all linear attention and MLP projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "model = FastLanguageModel.get_peft_model(\n",
                "    model,\n",
                "    r = 16,  # LoRA rank (16 provides high expressive capacity for linguistic tasks)\n",
                "    target_modules = [\n",
                "        \"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\",\n",
                "        \"gate_proj\", \"up_proj\", \"down_proj\",\n",
                "    ],\n",
                "    lora_alpha = 16,\n",
                "    lora_dropout = 0,  # Unsloth optimized with 0 dropout\n",
                "    bias = \"none\",     # \"none\" is optimized\n",
                "    use_gradient_checkpointing = \"unsloth\",  # Unsloth gradient checkpointing reduces memory footprint\n",
                "    random_state = 3407,\n",
                "    use_rslora = False,\n",
                "    loftq_config = None,\n",
                ")\n",
                "\n",
                "print(\"LoRA adapters configured successfully:\")\n",
                "model.print_trainable_parameters()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 📚 Step 4: Load and Format ChatML Dataset\n",
                "We apply the ChatML chat template to format the multi-turn conversations in `training_dataset_chatml.jsonl`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "from datasets import load_dataset\n",
                "from unsloth.chat_templates import get_chat_template\n",
                "\n",
                "# Setup ChatML template\n",
                "tokenizer = get_chat_template(\n",
                "    tokenizer,\n",
                "    chat_template = \"chatml\",\n",
                "    mapping = {\"role\": \"role\", \"content\": \"content\", \"user\": \"user\", \"assistant\": \"assistant\"},\n",
                ")\n",
                "\n",
                "def formatting_prompts_func(examples):\n",
                "    convos = examples[\"messages\"]\n",
                "    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]\n",
                "    return { \"text\": texts }\n",
                "\n",
                "# Check for dataset file\n",
                "dataset_file = \"training_dataset_chatml.jsonl\"\n",
                "if not os.path.exists(dataset_file):\n",
                "    if os.path.exists(\"data/processed/training_dataset_chatml.jsonl\"):\n",
                "        dataset_file = \"data/processed/training_dataset_chatml.jsonl\"\n",
                "    else:\n",
                "        print(f\"Dataset {dataset_file} not found in root. Please upload training_dataset_chatml.jsonl to Colab.\")\n",
                "\n",
                "# Load dataset\n",
                "dataset = load_dataset(\"json\", data_files={\"train\": dataset_file}, split=\"train\")\n",
                "dataset = dataset.map(formatting_prompts_func, batched = True)\n",
                "\n",
                "print(f\"Total training examples loaded: {len(dataset):,}\")\n",
                "print(\"\\n--- Formatted Sample (First 300 chars) ---\")\n",
                "print(dataset[0][\"text\"][:300])"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🔍 Step 5: Baseline Inference Check (Pre-Training)\n",
                "Let's test the baseline un-tuned Phi-3 model on cultural Nigerian proverbs to observe pre-training behavior."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "FastLanguageModel.for_inference(model)  # Enable 2x faster inference\n",
                "\n",
                "test_prompt = [\n",
                "    {\"role\": \"system\", \"content\": \"You are AfriWise — a culturally accurate assistant and storyteller for southern Nigerian languages (Ibibio, Igbo, and Edo/Bini). You prioritize authenticity, respectful dialogue, and proverbs.\"},\n",
                "    {\"role\": \"user\", \"content\": \"Explain the Bini proverb 'Agb\u1ecdn vbe egbe' and its cultural meaning.\"}\n",
                "]\n",
                "\n",
                "inputs = tokenizer.apply_chat_template(\n",
                "    test_prompt,\n",
                "    tokenize = True,\n",
                "    add_generation_prompt = True,\n",
                "    return_tensors = \"pt\",\n",
                ").to(\"cuda\")\n",
                "\n",
                "outputs = model.generate(input_ids = inputs, max_new_tokens = 128, temperature = 0.3, use_cache = True)\n",
                "print(\"--- Baseline Zero-Shot Output ---\")\n",
                "print(tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🏋️ Step 6: Configure SFTTrainer & Fine-Tune AfriWise\n",
                "We configure HuggingFace's `SFTTrainer` with Unsloth speedups, cosine learning rate decay, AdamW 8-bit optimizer, and mixed precision (`fp16`/`bf16`)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from trl import SFTTrainer\n",
                "from transformers import TrainingArguments\n",
                "from unsloth import is_bfloat16_supported\n",
                "\n",
                "# Configure trainer\n",
                "trainer = SFTTrainer(\n",
                "    model = model,\n",
                "    tokenizer = tokenizer,\n",
                "    train_dataset = dataset,\n",
                "    dataset_text_field = \"text\",\n",
                "    max_seq_length = max_seq_length,\n",
                "    dataset_num_proc = 2,\n",
                "    packing = False,  # Packing can speed up training for short sequences\n",
                "    args = TrainingArguments(\n",
                "        per_device_train_batch_size = 2,\n",
                "        gradient_accumulation_steps = 4,\n",
                "        warmup_steps = 20,\n",
                "        max_steps = 120,  # Set to 1-2 epochs or 300+ steps for full run\n",
                "        learning_rate = 2e-4,\n",
                "        fp16 = not is_bfloat16_supported(),\n",
                "        bf16 = is_bfloat16_supported(),\n",
                "        logging_steps = 10,\n",
                "        optim = \"adamw_8bit\",\n",
                "        weight_decay = 0.01,\n",
                "        lr_scheduler_type = \"cosine\",\n",
                "        seed = 3407,\n",
                "        output_dir = \"outputs\",\n",
                "        save_strategy = \"no\",\n",
                "    ),\n",
                ")\n",
                "\n",
                "# Start training\n",
                "print(\"Starting AfriWise fine-tuning...\")\n",
                "trainer_stats = trainer.train()\n",
                "print(\"Training complete! Metrics:\", trainer_stats.metrics)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🎯 Step 7: Post-Training Inference Evaluation\n",
                "Let's evaluate the fine-tuned model across Ibibio translation, Igbo proverbs, and Edo / Bini history."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "FastLanguageModel.for_inference(model)\n",
                "\n",
                "eval_prompts = [\n",
                "    \"Translate 'good morning, how is the family?' into Ibibio and Igbo.\",\n",
                "    \"Tell me the Edo / Bini story of the creation of the world and Osanobua.\",\n",
                "    \"Explain the Igbo proverb 'Onye fee eze, eze eruo ya aka' and how it guides social ethics.\",\n",
                "    \"What is the cultural significance of the Oba of Benin in Edo tradition?\",\n",
                "]\n",
                "\n",
                "print(\"=== AfriWise Post-Training Evaluations ===\\n\")\n",
                "for query in eval_prompts:\n",
                "    messages = [\n",
                "        {\"role\": \"system\", \"content\": \"You are AfriWise \u2014 a culturally accurate assistant and storyteller for southern Nigerian languages (Ibibio, Igbo, and Edo/Bini). You prioritize authenticity, respectful dialogue, and proverbs.\"},\n",
                "        {\"role\": \"user\", \"content\": query},\n",
                "    ]\n",
                "    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors=\"pt\").to(\"cuda\")\n",
                "    outputs = model.generate(input_ids=inputs, max_new_tokens=256, temperature=0.3, top_p=0.9, use_cache=True)\n",
                "    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)\n",
                "    print(f\"\ud83d\udde3\ufe0f User: {query}\\n\")\n",
                "    print(f\"\ud83e\udd16 AfriWise:\\n{response.strip()}\\n\")\n",
                "    print(\"=\" * 60 + \"\\n\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 💾 Step 8: Export Model to GGUF (Quantized 4-bit for Ollama)\n",
                "Unsloth allows direct merging of LoRA weights and exporting directly to `GGUF` format (`q4_k_m`) with a single function call."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Save GGUF model for direct local Ollama deployment\n",
                "# Quantization options: \"q4_k_m\" (recommended), \"q8_0\", \"q5_k_m\", \"f16\"\n",
                "output_gguf_dir = \"afriwise_phi3_q4\"\n",
                "model.save_pretrained_gguf(output_gguf_dir, tokenizer, quantization_method = \"q4_k_m\")\n",
                "\n",
                "print(f\"GGUF model exported to '{output_gguf_dir}' successfully!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 📥 Step 9: Download GGUF Model for Local Ollama Deployment\n",
                "Run this cell to download the `.gguf` file to your local computer. Once downloaded:\n",
                "1. Place the `.gguf` file in your `Africa_AI_Project` root directory as `afriwise-phi3-q4_k_m.gguf`.\n",
                "2. Run `ollama create afriwise -f Modelfile`\n",
                "3. Chat with `ollama run afriwise`"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import glob\n",
                "from google.colab import files\n",
                "\n",
                "gguf_files = glob.glob(\"afriwise_phi3_q4/*.gguf\") + glob.glob(\"*.gguf\")\n",
                "if gguf_files:\n",
                "    print(f\"Found GGUF export: {gguf_files[0]}\")\n",
                "    print(\"Initiating browser download...\")\n",
                "    files.download(gguf_files[0])\n",
                "else:\n",
                "    print(\"No GGUF file found in afriwise_phi3_q4/. Please check export cell output.\")"
            ]
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "provenance": [],
            "toc_visible": True
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = os.path.join(os.path.dirname(__file__), "..", "notebooks", "AfriWise_Finetune.ipynb")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
print("Saved notebooks/AfriWise_Finetune.ipynb")
