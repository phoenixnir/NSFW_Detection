## Installation

Create conda environment:
```bash
conda env create -f environment.yml
conda activate HiddenGuard
```

## Usage

### Data Preparation

Due to ethical concerns regarding the NSFW training data, the original dataset is not publicly available. However, access can be provided to qualified researchers for legitimate research purposes upon request.

### Training Process

1. **Generate attention data**:
   ```bash
   python gen_data.py
   ```
   This script processes prompts and extracts attention weights from all layers of CLIP's text encoder, saving them as `.npy` files.

2. **Train the classifier**:
   ```bash
   python train.py
   ```
   This trains Linear Discriminant Analysis models on the attention data to find optimal separating hyperplanes, saving the coefficients to `coefs.npy`.

### Testing and Evaluation

```bash
python test.py
```
Evaluates the model performance on a test set, calculating accuracy, true positive rate, and false positive rate.

### Inference

For predicting individual prompts or batch processing:

```bash
# For a single prompt
python inference.py --prompt "your prompt here"

# For a file containing multiple prompts (one per line)
python inference.py --file_path path/to/prompts.txt
```

The inference script will output 0 for SFW content and 1 for NSFW content.

## Ethical Considerations

Due to ethical concerns arising from the substantial number of NSFW prompts in our training data, we are not making the dataset publicly available. However, access will be provided to qualified researchers for legitimate research purposes upon request.