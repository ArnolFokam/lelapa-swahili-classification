import torch.nn as nn
from transformers import AutoModel


class HuggingFaceTextClassificationModel(nn.Module):
    """
    Generic class to fine-tune Hugging-Face
    model for text classification
    """
    
    def __init__(
        self,
        model_name: str,
        num_classes: int):
        """

        Args:
            model_name (str): model name on Hugging-Face
            num_classes (int): number of classes for classification
        """
        
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, return_dict=False)
        self.uses_token_type_ids = ("token_type_ids" in self.encoder.__call__.__code__.co_varnames)
        self.projector = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768, num_classes)
        )
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        
        # build arguments w.r.t model
        arguments = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        if self.uses_token_type_ids:
            arguments.update(token_type_ids=token_type_ids)
            
        _, output = self.encoder(**arguments)
        output = self.projector(output)
        return output