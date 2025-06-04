# Fast-DLLM

Fast-DLLM is a diffusion-based Large Language Model (LLM) inference acceleration framework that supports efficient inference for models like Dream and LLaDA.

## Project Structure

```
.
├── dream/          # Dream model related code
├── llada/          # LLaDA model related code
└── .gitignore      # Git ignore configuration
```

## Features

- Fast inference support for Dream and LLaDA models
- Multiple inference optimization strategies
- Code generation and evaluation capabilities
- Interactive chat interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/fast-dllm.git
cd fast-dllm
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Using LLaDA Model

#### Interactive Chat
```bash
python llada/chat.py --gen_length 128 --steps 128 --block_size 32
```

Parameter descriptions:
- `--gen_length`: Maximum length of generated text
- `--steps`: Number of sampling steps
- `--block_size`: Cache block size
- `--use_cache`: Whether to use cache
- `--if_cache_position`: Whether to use dual cache
- `--threshold`: Confidence threshold

#### Model Evaluation
For detailed evaluation instructions on GSM8K and HumanEval benchmarks, please refer to [LLaDA Evaluation Guide](llada/eval.md).

### 2. Using Dream Model

For detailed evaluation instructions on GSM8K and HumanEval benchmarks, please refer to [Dream Evaluation Guide](dream/eval.md).

## Contributing

Issues and Pull Requests are welcome!

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details. 