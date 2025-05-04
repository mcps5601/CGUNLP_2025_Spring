from transformers import AutoModel
import torch

# TODO2: Construct your model
class MultiLabelModel(torch.nn.Module):
    def __init__(
        self,
        model_name="bert-base-uncased",
        num_labels_A=1,
        num_labels_B=3,
    ):
        super().__init__()
        # Write your code here
        # Define what modules you will use in the model

    def forward(self, **kwargs):
        # Write your code here
        # Forward pass



     
