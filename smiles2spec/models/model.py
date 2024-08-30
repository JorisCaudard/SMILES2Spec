##Libraries
from transformers import AutoModelForSequenceClassification

import torch
import torch.nn as nn

##Classes

class Smile2Spec(nn.Module):
    """
    Basic structure of a Smile2Spec model.


    Parameters
    ----------
    args : dict, optional
        Dictionnaire d'arguments pour configurer le modèle. Les clés suivantes sont attendues :
        - 'model_name' (str) : Name of the LLM model in HuggingFace. By default, "DeepChem/ChemBERTa-5M-MTR".
        - 'output_activation' (str) : 'exp' or 'ReLU'. Final activation dunction. By default, 'exp'.
        - 'norm_range'(None or tuple) : Region of the spectrum normalized by the model. By default, None.
        - 'dropout' (float) : Dropour rate used by the model. By default, 0.2.
        - 'activation' (nn.Module) : Activation function used by the FFN module of the model. By default, nn.ReLU().
        - 'ffn_num_layers' (int) : Number of layers in the FFN module. By default, 3.
        - 'ffn_input_dim' (int) : Dimension of the LLM encoding. By default for ChemBERTa models, 199.
        - 'ffn_hidden_size' (int) : Number of neurons in the layers of the FFN module. By default, 2200.
        - 'ffn_output_dim' (int) : Dimension of the preditced spectrum. By default, 1801.
        - 'ffn_num_layers' (int) : Number of layers in the FFN module. By default, 3.
    """
    def __init__(self, args):
        """
        Initializes the Smile2Spec model.
        :param args: argument for building the model."""

        super(Smile2Spec, self).__init__()

        default_args = {
                    'model_name':"DeepChem/ChemBERTa-5M-MTR",
                    'output_activation':'exp',
                    'norm_range':None,
                    'dropout':0.2,
                    'activation':nn.ReLU(),
                    'ffn_num_layers':3,
                    'ffn_input_dim':199,
                    'ffn_hidden_size':2200,
                    'ffn_output_dim':1801,
                    'ffn_num_layers':3
                        }
        
        if args is None:
            args = default_args
        else:
            for key, value in default_args.items():
                args.setdefault(key, value)

        self.args = args

        #Create LLM head.
        self.LLM = AutoModelForSequenceClassification.from_pretrained(args.get('model_name')) # type: ignore

        #Create output objects.
        self.output_activation = args.get('output_activation')
        self.norm_range = args.get('norm_range')

        #Create FFN params.
        dropout = nn.Dropout(args.get('dropout')) # type: ignore
        activation = args.get('activation')

        # Create FFN layers
        if args.get('ffn_num_layers') == 1:
            ffn = [
                dropout,
                nn.Linear(args.get('ffn_input_dim'), args.get('ffn_output_dim')) # type: ignore
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(args.get('ffn_input_dim'), args.get('ffn_hidden_size')) # type: ignore
            ]
            for _ in range(args.get('ffn_num_layers') - 2): # type: ignore
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.get('ffn_hidden_size'), args.get('ffn_hidden_size')) # type: ignore
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.get('ffn_hidden_size'), args.get('ffn_output_dim')) # type: ignore
            ])

        self.ffn = nn.Sequential(*ffn)

    def forward(self,
                input_ids = None,
                attention_mask = None,
                labels=None):
        """
        Runs the Smile2Spec model on input.
        
        Parameters
        ----------
        input_ids (torch.Tensor) : Tensor of dimension (batch_size, input_size) containing the input_ids to feed to the LLM (Tokenized SMILES)
        attention_mask (torch.Tensor) : Tensor of dimension (batch_size, input_size) containing the attention_masks to feed to the LLM (Tokenized SMILES)
        labels (torch.Tensor) : Tensor of dimension (batch_size, input_size) containing the labels (can be None, just for ease of use with the Trainer class for training)

        Returns
        -------
        output (torch.Tensor) : Tensor of dimension (batch_size, output_size) containing predictions by the model
        """

        #Compute LLM output.
        LLM_output = self.LLM(input_ids,
            attention_mask=attention_mask).logits # type: ignore

        #Compute ffn output.
        output = self.ffn(LLM_output)

        #Positive value mapping.
        if self.output_activation == 'exp':
            output = torch.exp(output)
        if self.output_activation == 'ReLU':
            f = nn.ReLU()
            output = f(output)

        #Normalization mapping.
        if self.norm_range is not None:
            norm_data = output[:, self.norm_range[0]:self.norm_range[1]]
            norm_sum = torch.sum(norm_data, 1)
            norm_sum = torch.unsqueeze(norm_sum, 1)
            output = torch.div(output, norm_sum)

        return output

##Function