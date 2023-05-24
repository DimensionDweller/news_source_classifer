import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer

def clean_text(text):
    text = text.replace('\n', ' ').strip()
    return text

def get_data(file_path):
    df = pd.read_csv(file_path)
    df = df[['source_name', 'author', 'content']]
    df['content'] = df['content'].apply(lambda x: '' if pd.isna(x) else str(x))
    df['content'] = df['content'].apply(clean_text)
    return df

def split_data(df, test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['source_name'], random_state=random_state)
    return train_df, test_df

def get_indices(df):
    source_to_idx = {source: idx for idx, source in enumerate(df['source_name'].unique())}
    idx_to_source = {idx: source for source, idx in source_to_idx.items()}
    return source_to_idx, idx_to_source

class NewsDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, source_to_idx):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.source_to_idx = source_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        content = row['content']
        source = row['source_name']

        inputs = self.tokenizer.encode_plus(
            content,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )

        return {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'targets': torch.tensor(self.source_to_idx[source], dtype=torch.long)
        }

def get_tokenizer(pretrained_model_name='distilbert-base-uncased'):
    tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model_name)
    return tokenizer

def create_tokenized_data_loaders(train_df, test_df, tokenizer, source_to_idx, max_len, batch_size):
    train_set = NewsDataset(train_df, tokenizer, max_len, source_to_idx)
    test_set = NewsDataset(test_df, tokenizer, max_len, source_to_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, test_loader
