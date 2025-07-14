# rlhf-tuning-llama-2-7B-using-vertex-ai
Using Vertex AI to tune Llama 2 7B with Reinforcement Learning from Human Feedback (RLHF) for text summarization tasks.

# About the project
This is a Google Colab notebook for using Vertex AI Pipelines from Google Cloud Platform (GCP) to fine-tune Llama 2 7B using RLHF and using it to perform batch inference for text summarization tasks. The goal is not however to focus on the performance aspect or the quality of the summarization outputs, but to go through the end-to-end workflow required for such a project.

**UPDATE:** This tutorial refers to GCP's [documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/models/tune-text-models-rlhf) about tuning Pathways Language Models (PaLM) text models by using RLHF tuning at multiple places. The link to this documentation was valid at the time of writing. However, due to the rapidly evolving cloud documentation, this link is no longer active because Google has updated and consolidated its documentation as its AI models and tuning methods have advanced. However, the other resources provided in this tutorial serve as sufficient context and reference for this work.

The work uses OpenAI's [`summarize_from_feedback`](https://huggingface.co/datasets/openai/summarize_from_feedback) dataset from Hugging Face, which is a collection of Reddit posts based on the `TL;DR` dataset, as the preference dataset. The [`reddit_tifu`](https://huggingface.co/datasets/ctr4si/reddit_tifu) dataset from Hugging Face, which is a collection of Reddit posts with short and long summaries, is used for the prompt and evaluation datasets. The data is first cleaned to align with the requirements of the RLHF pipeline on Vertex AI.

The RLHF pipeline on Vertex AI takes care of the reward model and Reinforcement Learning (RL) training, deployment, and inference. The RLHF pipeline job on Vertex AI is run for an epoch each of the reward model and RL training. Even though the RLHF pipeline job on Vertex AI is run for an epoch each of the reward model and RL training, we confirm that 1 epoch of training is sufficient to demonstrate the RLHF tuning workflow using Llama 2 7B and Vertex AI's RLHF pipelines for text summarization tasks. However, this is a quick and dirty implementation and there is definitely a decent scope for improvement, which we discuss in the work.

# Tools Used
Tools, libraries, and services used in this project include Datasets (from Hugging Face), pandas, scikit-learn, dotenv, TensorBoard, Kubeflow Pipelines, as well as IAM, Cloud Storage, and Vertex AI Pipelines from GCP.

# Who can benefit from the project?
Anyone can use the project to get started with the basics of RLHF tuning Large Language Models (LLMs) using Vertex AI for text summarization tasks.

# Getting Started
Anyone interested in getting started with Machine Learning, Deep Learning, Natural Language Processing (NLP), or Generative Artificial Intelligence (GenAI), specifically, fine-tuning LLMs with RLHF for text summarization using Vertex AI on GCP, can clone or download the project to get started.

# References
The most important points of reference for the project are as follows.
1. DeepLearning.AI's [course](https://www.deeplearning.ai/short-courses/reinforcement-learning-from-human-feedback/) on RLHF in association with Google Cloud.
2. GCP's [documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/models/tune-text-models-rlhf) about tuning PaLM text models by using RLHF tuning. Although PaLM models are deprecated, the information is very helpful in the context of RLHF tuning LLMs using Vertex AI.
3. GCP's [blog](https://cloud.google.com/blog/products/ai-machine-learning/rlhf-on-google-cloud) on *RLHF Tuning with Vertex AI*.
4. GCP's [documentation](https://cloud.google.com/vertex-ai/docs/pipelines/introduction) about *Introduction to Vertex AI Pipelines*.
5. GCP's [Colab notebook](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/model_garden/model_garden_pytorch_llama2_rlhf_tuning.ipynb) on *Vertex AI Model Garden - LLaMA2 (RLHF)*.
6. GCP's [Colab notebook](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/generative_ai/rlhf_tune_llm.ipynb) on *Vertex AI LLM Reinforcement Learning from Human Feedback*.
7. GCP's [Colab notebook](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/generative_ai/batch_eval_llm.ipynb) on *Vertex AI LLM Batch Inference with RLHF-tuned Models*.

# Additional Notes
1. The preference dataset is available [here](https://huggingface.co/datasets/openai/summarize_from_feedback) and the prompt / evaluation dataset is available [here](https://huggingface.co/datasets/ctr4si/reddit_tifu). If the datasets are taken down in the future, please feel free to reach out to me at ankanatwork@gmail.com if you would like to learn more about the data. However, I may not be able to share the datasets with you due to licensing restrictions.
2. The project is a basic one in nature and is not currently being maintained.
3. [Here](https://researchguy.in/rlhf-using-llama-2-7b-and-vertex-ai/) is the blog post covering this work.
