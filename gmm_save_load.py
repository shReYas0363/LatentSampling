import torch
from gmm import GaussianMixture

def save_gmm(gmm_model, filepath="gmm_checkpoint.pth"):
    """
    Saves the GMM weights and architectural parameters.
    """
    checkpoint = {
        'n_components': gmm_model.n_components,
        'n_features': gmm_model.n_features,
        'covariance_type': gmm_model.covariance_type,
        'init_params': gmm_model.init_params,
        'eps': gmm_model.eps,
        'state_dict': gmm_model.state_dict(),
        'params_fitted': gmm_model.params_fitted,
        'log_likelihood': gmm_model.log_likelihood
    }
    
    torch.save(checkpoint, filepath)
    print(f"GMM successfully saved to {filepath}")


def load_gmm(filepath="gmm_checkpoint.pth", device="cuda"):

    checkpoint = torch.load(filepath, map_location=device)
    
    gmm = GaussianMixture(
        n_components=checkpoint['n_components'],
        n_features=checkpoint['n_features'],
        covariance_type=checkpoint['covariance_type'],
        init_params=checkpoint['init_params'],
        eps=checkpoint['eps']
    )
    
    gmm.load_state_dict(checkpoint['state_dict'])
    
    gmm.params_fitted = checkpoint['params_fitted']
    gmm.log_likelihood = checkpoint['log_likelihood']

    gmm.to(device)
    gmm.eval() 
    
    print(f"GMM successfully loaded onto {device} and ready for sampling.")
    return gmm