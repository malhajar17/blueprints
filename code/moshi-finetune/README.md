# Minimal Moshi-Finetune Setup

This is a minimal setup for fine-tuning Moshi models, extracted from the original [moshi-finetune](https://github.com/kyutai-labs/moshi-finetune) repository.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd /workspace/fcs-experiments-private
pip install -r code/moshi-finetune/requirements.txt
```

### 2. Prepare Configuration
Edit `example/moshi_7B.yaml` to set your paths:
- `data.train_data`: Path to your training data (`.jsonl` file)
- `run_dir`: Directory where checkpoints will be saved

### 3. Run Training
```bash
# Single GPU
torchrun --nproc-per-node 1 -m train example/moshi_7B.yaml

# Multiple GPUs (8)
torchrun --nproc-per-node 8 --master_port $RANDOM -m train example/moshi_7B.yaml
```

## 📊 Example Configuration

The `example/moshi_7B.yaml` contains optimized settings for training:
- LoRA rank: 128
- Batch size: 16
- Learning rate: 2e-6
- Duration: 100 seconds
- Max steps: 2000

## 🔧 Data Format

Your training data should be a `.jsonl` file where each line contains:
```json
{"path": "relative/path/to/audio.wav", "duration": 24.5}
```

Audio files should be stereo:
- Left channel: Moshi-generated audio
- Right channel: User input audio

Each audio file should have a corresponding `.json` transcript file.

## 📚 Full Documentation

For complete documentation, dataset preparation, and advanced configuration, see the original [moshi-finetune repository](https://github.com/kyutai-labs/moshi-finetune).

## 🎯 What's Different in This Minimal Setup

- **Simplified dependencies**: Only core requirements without torch (uses pre-installed torch 2.8)
- **Minimal moshi package**: Resolves dependency conflicts without pytest issues
- **Clean structure**: Only essential files for training
- **Relative paths**: Works from the experiments directory