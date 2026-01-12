# LongDA: Long-Document Data Analysis Benchmark

[![Paper](https://img.shields.io/badge/arXiv-2601.02598-b31b1b.svg)](https://arxiv.org/pdf/2601.02598)
[![Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/EvilBench/LongDA)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**LongDA** is a data analysis benchmark for evaluating LLM-based agents under documentation-intensive analytical workflows. Unlike existing benchmarks that assume well-specified schemas, LongDA targets real-world settings where navigating long documentation and complex data is the primary bottleneck.

## 📖 Overview

We manually curate **17 U.S. national surveys** with their complete documentation and extract **505 analytical queries** from expert-written publications. Solving these queries requires agents to:

1. **Retrieve and integrate** key information from multiple unstructured documents (~263K tokens on average)
2. **Navigate long documentation** including codebooks, technical reports, and user guides
3. **Perform multi-step computations** with proper sampling weights and survey design considerations
4. **Write executable code** to extract variables and compute results

This benchmark captures the reality of documentation-intensive data analysis where information gathering is often the dominant bottleneck.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Yiyang-Ian-Li/LongDA
cd LongDA

# Install dependencies
pip install -r requirements.txt
```

### Download Data

Download the **complete benchmark dataset** including all survey data and documentation from Hugging Face:

```bash
# Install Git LFS (required for large files)
git lfs install

# Clone the complete dataset
cd /path/to/your/workspace
git clone https://huggingface.co/datasets/EvilBench/LongDA benchmark

# Your directory should now contain:
# benchmark/
# ├── benchmark.csv
# └── [17 survey folders with data/ and docs/]
```

Or download programmatically:
```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="EvilBench/LongDA",
    repo_type="dataset",
    local_dir="./benchmark"
)
```

**Note**: The complete dataset (~1.8GB) is required. The `benchmark.csv` file alone is insufficient as evaluation requires access to raw survey data and documentation.

### Run Evaluation

1. **Configure your agent**: Copy and modify the example config

```bash
cp configs/example_config.yaml configs/my_config.yaml
# Edit my_config.yaml with your API key
```

2. **Run the benchmark**:

```bash
python main.py --config_file configs/my_config.yaml
```

3. **View results**: Results are saved in `results/TIMESTAMP_MODEL/`
   - `run_summary.json`: Overall metrics (match rate, token usage, runtime)
   - `block_metrics.json`: Per-survey-source performance
   - `answers_progress.csv`: All answers and correctness
   - `messages/`: Detailed traces for each query

## 📊 Benchmark Statistics

- **505 queries** across 17 U.S. national surveys
- **6 federal agencies**: covering health, labor, economics, education, and social sciences
- **30 expert-written publications** used for query extraction
- **~263K tokens** average context per query (much longer than existing benchmarks)
- **Surveys**: NHANES, CPS-ASEC, GSS, NSDUH, NHIS, NSCG, NSFG, ATUS, HERD, RHFS, SDR, SSERF, STC, NTEWS, ASFIN, ASPEP, ASPP

<!-- ## 🏆 Results

See our paper for complete evaluation results. Key findings:
- Substantial performance gaps even among state-of-the-art models
- Documentation navigation remains challenging for current LLM agents
- Tool usage and multi-step reasoning are critical for success -->

## 📝 Evaluation Metrics

LongDA evaluates agents on:

- **Match Rate**: Proportion of queries answered within tolerance (default: 5% relative error for numbers)
- **Token Efficiency**: Total tokens consumed across all queries
- **Runtime**: Total time to complete the benchmark
- **Steps**: Average number of agent-tool interactions per query

Answers are validated with flexible matching for numerical values and exact matching for list structures.

## 🛠️ Project Structure

```
LongDA/
├── benchmark/              # Benchmark data and documentation
│   ├── benchmark.csv      # 505 queries with ground truth
│   └── [SURVEY]/          # Survey-specific folders
│       ├── data/          # Raw data files
│       └── docs/          # Long documentation (codebooks, guides, etc.)
├── configs/               # Model configuration templates
├── tools/                 # Custom tools for the agent framework
├── main.py               # Main evaluation script
├── evaluate_results.py   # Post-hoc evaluation and analysis
├── metric.py             # Evaluation metrics implementation
├── my_agent.py           # LongTA agent framework
└── utils.py              # Utility functions
```

## 📚 Citation

If you use LongDA in your research, please cite:

```bibtex
@article{li2026longda,
  title={LongDA: Benchmarking LLM Agents for Long-Document Data Analysis},
  author={Li, Yiyang and Zhang, Zheyuan and Ma, Tianyi and Wang, Zehong and Murugesan, Keerthiram and Zhang, Chuxu and Ye, Yanfang},
  journal={arXiv preprint arXiv:2601.02598},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or issues, please:
- Open a GitHub issue
- Contact: yli62@nd.edu

---

**Note**: This benchmark is for research purposes only. Please comply with data usage policies when using the survey data. 