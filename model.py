import torch
from transformers import DistilBertModel

class SourceClassifier(torch.nn.Module):
    def __init__(self, num_sources):
        super(SourceClassifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = torch.nn.Dropout(0.3)
        self.output = torch.nn.Linear(768, num_sources)

    def forward(self, ids, mask):
        output = self.distilbert(ids, attention_mask=mask)
        output = self.dropout(output[0][:, 0, :])  # Use the [CLS] token representation
        output = self.output(output)
        return output
