from datasets import load_dataset

raw_datasets = load_dataset("pythainlp/han-instruct-dataset-v1.0")
raw_datasets.keys()
raw_datasets['train'][0]