# Fast-DLLM
[![arXiv](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2505.22618 )

Fast-DLLM is a diffusion-based Large Language Model (LLM) inference acceleration framework that supports efficient inference for models like Dream and LLaDA.

<div align="center">
  <img src="asset/speedup.jpg" alt="End-to-end speedup over vanilla LLaDA baseline" width="800"/>
  <p>End-to-end speedup over vanilla LLaDA baseline</p>
</div>

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

### Key Features

1. **Key-Value Cache for Block-Wise Decoding**
   We propose an efficient block-wise decoding KV Cache mechanism for Masked Diffusion Models (MDMs). By reusing attention Key-Value activations across multiple steps within each block, our approach avoids redundant computation and significantly accelerates inference. Furthermore, our DualCache extension also caches masked suffix tokens, enabling even greater speedup with negligible accuracy loss.

<div align="center">
  <img src="asset/kvcache.jpg" alt="KV Cache for block-wise decoding" width="800"/>
  <p>KV Cache for block-wise decoding</p>
</div>

2. **Confidence-Aware Parallel Decoding**
   Instead of decoding tokens sequentially, we introduce a confidence-aware parallel decoding scheme. At each step, only tokens with confidence over a threshold are unmasked in parallel, while uncertain ones remain masked for future steps. This selective approach effectively balances decoding efficiency and output quality.

<div align="center">
  <img src="asset/output.gif" alt="Decoding comparison" width="800"/>
  <p><b>Left:</b> Standard decoding (LLaDA). <b>Right:</b> Confidence-aware parallel decoding.</p>
</div>

<div align="center">
  <img src="asset/pseudo_code.jpg" alt="Pseudo code for our method" width="800"/>
  <p>Pseudo code for our method</p>
</div>

3. **Overall Performance**
   Overall, introducing the KV Cache mechanism yields significant speed improvements for all tasks and sequence lengths, typically achieving a 2x to 3.6x speedup compared to the vanilla backbone. When the parallel decoding strategy is applied individually, we see additional acceleration, often pushing speedups to 4x-6x for the evaluated settings, particularly as the generation length increases.

<div align="center">
  <img src="asset/overall_performance.jpg" alt="Overall performance" width="800"/>
  <p>Overall performance comparison</p>
</div>

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

#### Web Demo
We also provide a web demo using Gradio. First, install Gradio:
```bash
pip install gradio
```

Then run the demo:
```bash
cd llada
python app.py
```

#### Model Evaluation
For detailed evaluation instructions on GSM8K and HumanEval benchmarks, please refer to [LLaDA Evaluation Guide](llada/eval.md).

### 2. Using Dream Model

For detailed evaluation instructions on GSM8K and HumanEval benchmarks, please refer to [Dream Evaluation Guide](dream/eval.md).

## Contributing

Issues and Pull Requests are welcome!

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details. 

## Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{wu2025fastdllmtrainingfreeaccelerationdiffusion,
      title={Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding}, 
      author={Chengyue Wu and Hao Zhang and Shuchen Xue and Zhijian Liu and Shizhe Diao and Ligeng Zhu and Ping Luo and Song Han and Enze Xie},
      year={2025},
      eprint={2505.22618},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.22618}, 
}
```

## Acknowledgements

We would like to thank the authors of [LLaDA](https://github.com/llada-project/llada) and [Dream](https://github.com/dream-project/dream) for their excellent work and open-source contributions. 