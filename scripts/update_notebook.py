import json
from pathlib import Path

nb_path = Path("notebooks/AfriWise_Finetune.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Step 7: Fix generation sampling parameters, repetition penalty, eval mode, and EOS tokens
nb["cells"][12]["source"] = [
    "model.eval()  # Set model to evaluation mode\n",
    "\n",
    "eval_prompts = [\n",
    "    \"Translate 'good morning, how is the family?' into Ibibio and Igbo.\",\n",
    "    \"Tell me the Edo / Bini story of the creation of the world and Osanobua.\",\n",
    "    \"Explain the Igbo proverb 'Onye fee eze, eze eruo ya aka' and how it guides social ethics.\",\n",
    "    \"What is the cultural significance of the Oba of Benin in Edo tradition?\",\n",
    "]\n",
    "\n",
    "# Find all end-of-sequence token IDs for Phi-3\n",
    "eos_ids = [tokenizer.eos_token_id]\n",
    "for end_token in [\"<|end|>\", \"<|endoftext|>\"]:\n",
    "    token_id = tokenizer.convert_tokens_to_ids(end_token)\n",
    "    if token_id is not None and token_id not in eos_ids:\n",
    "        eos_ids.append(token_id)\n",
    "\n",
    "print(\"=== AfriWise Post-Training Evaluations ===\\n\")\n",
    "for query in eval_prompts:\n",
    "    messages = [\n",
    "        {\"role\": \"system\", \"content\": \"You are AfriWise — a culturally accurate assistant and storyteller for southern Nigerian languages (Ibibio, Igbo, and Edo/Bini). You prioritize authenticity, respectful dialogue, and proverbs.\"},\n",
    "        {\"role\": \"user\", \"content\": query},\n",
    "    ]\n",
    "    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\n",
    "    inputs = tokenizer(prompt, return_tensors=\"pt\").to(\"cuda\")\n",
    "    \n",
    "    with torch.no_grad():\n",
    "        outputs = model.generate(\n",
    "            **inputs,\n",
    "            max_new_tokens=256,\n",
    "            temperature=0.7,\n",
    "            top_p=0.9,\n",
    "            repetition_penalty=1.15,\n",
    "            do_sample=True,\n",
    "            eos_token_id=eos_ids,\n",
    "            pad_token_id=tokenizer.pad_token_id,\n",
    "        )\n",
    "    \n",
    "    response = tokenizer.decode(outputs[0][inputs[\"input_ids\"].shape[1]:], skip_special_tokens=True)\n",
    "    print(f\"🗣️ User: {query}\\n\")\n",
    "    print(f\"🤖 AfriWise:\\n{response.strip()}\\n\")\n",
    "    print(\"=\" * 60 + \"\\n\")\n"
]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Updated Step 7 with robust generation parameters!")
