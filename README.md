# optimization_human_preference
## Task 1
Dataset Name: Dahoas/full-hh-rlhf
Source: https://huggingface.co/datasets/Dahoas/full-hh-rlhf
### Description:
Dahoas/full-hh-rlhf is an RLHF (Reinforcement Learning from Human Feedback) dataset.
It contains human preference rankings for AI-generated responses.
The dataset is specifically designed for training models in helpfulness and harmlessness (HH).
It is commonly used for Direct Preference Optimization (DPO) training, which fine-tunes models to align with human preferences.

### Dataset Preprocessing
To make this dataset suitable for preference-based training, I applied preprocessing steps:

1. Load the dataset

2. Extract Relevant Fields

The dataset contains:
prompt: The input text given to the AI.
response: AI-generated reply
chosen: The AI-generated response preferred by humans.
rejected: The AI-generated response not preferred by humans.
Extracted and structured these fields for preference-based learning.
3. Structure for Training
The dataset is structured into:
prompt: The input question or command.
chosen: The preferred response (good AI behavior).
rejected: The non-preferred response (undesirable AI behavior).

## Task 2
1) 
- Loads a pre-trained GPT-2 model and tokenizer.
- Sets EOS token as the padding token (GPT-2 doesn't have a default pad
  token).
- Loads Dahoas/full-hh-rlhf dataset.
- Extracts prompt, chosen, and rejected fields for preference learning.
- Configure Training Hyperparameters
## Task 3
## Task 4
https://huggingface.co/ToobiBabe/gpt_dpo/tree/main 