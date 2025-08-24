from transformers import AutoModel
import torch

class MultiLabelModel(torch.nn.Module):
    def __init__(
        self,
        model_name="bert-base-uncased",
        num_labels_A=1,
        num_labels_B=3,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, cache_dir="./cache/")
        self.classifier1 = torch.nn.Linear(self.model.config.hidden_size, num_labels_A)
        self.classifier2 = torch.nn.Linear(self.model.config.hidden_size, num_labels_B)

        self.config = self.model.config
    
    def forward(self, **kwargs):
        model_kwargs = {k: v for k, v in kwargs.items() if k not in ['labels']}
        outputs = self.model(**model_kwargs).last_hidden_state[:, 0, :]
        outputs1 = self.classifier1(outputs)
        outputs2 = self.classifier2(outputs)
        return outputs1, outputs2