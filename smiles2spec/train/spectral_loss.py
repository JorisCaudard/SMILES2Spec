##Libraries
import torch
import torch.nn as nn

##Classes

##Functions

class SIDLoss(nn.Module):
    """
    Implementation of the SID as a loss function, in the form of a nn.Module


    Parameters
    ----------
    None
    """
    def __init__(self):
        super().__init__()

    def forward(self, model_spectra, target_spectra, threshold = 10e-12, mean=True):
        """
        Calculate the SID loss between two spectras.
        
        Parameters
        ----------
        model_spectra (torch.Tensor) : Spectrum predicted by the model (or batch).
        target_spectra (torch.Tensor) : True spectrum (or batch).
        threshold (torch.Tensor) : Optional, to guarantee strict postivity of both spectras. By default, 10e-12.
        mean (Bool) : Wether or not to cimpute the mean loss of the batch. By default, True.

        Returns
        -------
        loss (torch.Tensor) : Tensor of dimension (batch_size, 1) containing losses between spectras in the batch (or the mean loss in the batch).
        """

        model_spectra[model_spectra <= threshold] = threshold
        target_spectra[target_spectra <= threshold] = threshold


        loss = torch.ones_like(target_spectra)

        loss = torch.mul(torch.log(torch.div(model_spectra, target_spectra)), model_spectra) \
                + torch.mul(torch.log(torch.div(target_spectra, model_spectra)), target_spectra)
        
        loss = torch.sum(loss, dim=1)

        return loss.mean() if mean else loss