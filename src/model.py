import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel

class TrademarkDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        description = str(self.data.loc[index, 'description'])
        
        # Adjust class_id to 0-based index for CrossEntropyLoss
        class_id = int(self.data.loc[index, 'class_id']) - 1
        
        # Check for valid class_id range (0 to n_classes - 1)
        if class_id < 0 or class_id >= 45:  # Assuming 45 classes
            raise ValueError(f"Invalid class_id {class_id + 1} at index {index}. It should be between 1 and 45.")

        encoding = self.tokenizer.encode_plus(
            description,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'class_id': torch.tensor(class_id, dtype=torch.long)
        }

class TrademarkClassifier(nn.Module):
    def __init__(self, n_classes):
        super(TrademarkClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs['pooler_output']
        output = self.drop(pooled_output)
        return self.out(output)
