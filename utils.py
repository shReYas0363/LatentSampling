import torch

def calculate_matmul_n_times(n_components, mat_a, mat_b):

    return torch.matmul(mat_a, mat_b)


def calculate_matmul(mat_a, mat_b):

    assert mat_a.shape[-2] == 1 and mat_b.shape[-1] == 1
    scaled_distance = torch.sum(mat_a.squeeze(-2) * mat_b.squeeze(-1), dim=2, keepdim=True)
    
    return scaled_distance.unsqueeze(-1)