# Fine-Tune Local LLM

This project fine-tunes `HuggingFaceTB/SmolLM2-135M-Instruct` on a small caveman-style instruction dataset, then converts the trained model to GGUF so it can be imported and run locally with Ollama.

## Project Structure

```text
.
├── caveman_dataset.json        # Training dataset
├── fine_tune.py                # Fine-tunes SmolLM2-135M-Instruct
├── main.py                     # Runs the fine-tuned Hugging Face model
├── Modelfile                   # Ollama model definition
├── outputs/
│   ├── final_model/            # Fine-tuned Hugging Face model
│   ├── caveman-f16.gguf        # Converted GGUF model
│   └── caveman-q8_0.gguf       # Quantized GGUF model used by Ollama
└── llama-convert/              # Local converter copy; not required if using llama.cpp
```

## Requirements

- Python 3.9+
- A Python virtual environment
- Ollama installed
- `llama.cpp` conversion tools
- `llama-quantize`, usually installed through `llama.cpp` or Homebrew

Check Ollama:

```bash
ollama --version
```

Check `llama-quantize`:

```bash
llama-quantize --help
```

## Setup

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch transformers datasets trl sentencepiece
```

## Fine-Tune the Model

Run:

```bash
python fine_tune.py
```

This trains the model and saves the final Hugging Face model to:

```text
outputs/final_model
```

You can test the Hugging Face version directly:

```bash
python main.py
```

## Convert to GGUF

Ollama cannot run `model.safetensors` directly. Convert the fine-tuned Hugging Face model to GGUF first.

Clone the official `llama.cpp` repo:

```bash
git clone https://github.com/ggml-org/llama.cpp.git /tmp/llama.cpp
```

Install converter dependencies:

```bash
.venv/bin/pip install -r /tmp/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
.venv/bin/pip install sentencepiece
```

Convert the model:

```bash
.venv/bin/python /tmp/llama.cpp/convert_hf_to_gguf.py \
  outputs/final_model \
  --outfile outputs/caveman-f16.gguf \
  --outtype f16
```

## Quantize the GGUF Model

Quantize to `Q8_0`:

```bash
llama-quantize outputs/caveman-f16.gguf outputs/caveman-q8_0.gguf Q8_0
```

The quantized model is the file used by Ollama:

```text
outputs/caveman-q8_0.gguf
```

## Import into Ollama

Make sure Ollama is running. Open the Ollama desktop app, or run this in another terminal:

```bash
ollama serve
```

Then create the Ollama model from the `Modelfile`:

```bash
ollama create caveman -f Modelfile
```

## Run with Ollama

Run a one-off prompt:

```bash
ollama run caveman "what is a lion?"
```

Expected style:

```text
lion big cat. live in jungle. hunt deer. fierce.
```

Start an interactive chat:

```bash
ollama run caveman
```

## Current Ollama Modelfile

```text
FROM ./outputs/caveman-q8_0.gguf

TEMPLATE """### Instruction: {{ .Prompt }}
### Response:"""

PARAMETER stop "### Instruction:"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.7
```

## Troubleshooting

If conversion fails with:

```text
ModuleNotFoundError: No module named 'sentencepiece'
```

Install it:

```bash
.venv/bin/pip install sentencepiece
```

If `ollama create` or `ollama run` cannot connect:

```bash
ollama serve
```

or open the Ollama desktop app.

If quantization says the GGUF file does not exist, the conversion step failed. Fix the conversion error first, then rerun `llama-quantize`.

my huggingface URL : huggingface.co/Yousuf008/caveman-smollm2