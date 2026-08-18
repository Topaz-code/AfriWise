# AfriWise

AfriWise is a "Translator of Knowledge" model fine-tuned for Southern Nigerian languages (Igbo, Efik/Ibibio, and Bini/Edo). It is based on the Phi-3-Mini instruct architecture.

## Run Locally

1. Install [Ollama](https://ollama.com/)
2. Run the model:
   ```bash
   ollama run emmy_rabs/afriwise
   ```

## Training Pipeline

The repository contains the scripts and notebooks used to fine-tune the model via Google Colab.

### Key Components

- **`afriwise_colab_training.ipynb`**: The main Google Colab notebook. Runs the Unsloth LoRA fine-tuning process, merges to 16-bit, and converts to GGUF format.
- **`scripts/generate_afriwise_colab_notebook.py`**: Generates the notebook and hardens the installation commands for `unsloth`, `xformers`, and `bitsandbytes`.
- **`scripts/build_augmented_dataset.py`**: Compiles raw language datasets and hardcoded cultural Q&A pairs into a ChatML-formatted JSONL file.
- **`Modelfile`**: Production Ollama configuration defining the system prompt and sampling parameters.

### Data Sources
* Igbo: `masakhane/mafand`, `masakhane/afrixnli`, `masakhane/masakhaner2`
* Efik: `Davlan/ibom-mt-en-efi`, `masakhane/masakhanews`
* Bini: `bible-nlp/biblenlp-corpus`

### System Prompt
> You are AfriWise - a highly capable, knowledgeable AI. When asked a question in Igbo, Efik/Ibibio, or Bini/Edo, you access your vast general knowledge and output the answer flawlessly in that specific language. You also converse fluently and intelligently in English.
