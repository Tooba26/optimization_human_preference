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
## 🎯 Hyperparameters Used for DPO Training

| **Hyperparameter**                 | **Value**  | **Description** |
|-------------------------------------|------------|----------------|
| `learning_rate`                     | `5e-5`     | Standard learning rate for fine-tuning GPT models. |
| `per_device_train_batch_size`       | `8`        | Number of samples processed per GPU. |
| `per_device_eval_batch_size`        | `8`        | Batch size for evaluation. |
| `gradient_accumulation_steps`       | `2`        | Accumulates gradients over multiple steps to simulate larger batches. |
| `max_steps`                         | `1000`     | Total training steps before stopping. |
| `num_train_epochs`                  | `3`        | Number of full dataset passes. |
| `save_strategy`                     | `"epoch"`  | Saves the model at the end of each epoch. |
| `evaluation_strategy`               | `"epoch"`  | Evaluates the model at the end of each epoch. |
| `logging_steps`                     | `50`       | Logs training metrics every 50 steps. |
| `fp16`                              | `True`     | Enables Mixed Precision for faster training and lower memory usage. |
| `gradient_checkpointing`            | `True`     | Saves memory by recomputing gradients during backpropagation. |
| `padding_value`                     | `tokenizer.pad_token_id` | Sets padding token for batch processing. |
| `max_prompt_length`                 | `128`      | Limits input prompt length. |
| `max_completion_length`             | `128`      | Limits response length. |
| `beta`                              | `0.1`      | Regularization factor for DPO to balance learning. |
| `report_to`                         | `"wandb"`  | Logs training metrics to Weights & Biases (wandb). |

2) Training Performance
![Training](/images/performance.png)

## Task 3
## Task 4
https://huggingface.co/ToobiBabe/gpt_dpo/tree/main 