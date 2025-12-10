import torch
import torch.nn.functional as F


def agnostic_loss(queries, key_idx, key_list, lambda_agnostic, lambda_kk):
    '''
    Calculate the loss between query/key and between keys in the selector key list

    Args:
        queries : torch.Tensor[float32]
            Query vector [N, C]
        key_idx : int
            Index of the key in the key list
        key_list : torch.Tensor[float32]
            Key list K x [N]

        lambda_agnostic : float
            Domain-agnostic loss weight
        lambda_kk : float
            Key-key loss weight
    '''
    loss = 0.0
    selected_key = key_list[key_idx]
    cosine_sim_qk = F.cosine_similarity(queries, selected_key.transpose(-2,-1), dim=1)
    loss_qk = 1 - cosine_sim_qk.mean()
    loss += loss_qk

    loss_kk = 0.0
    for i in range(len(key_list)):
        if i != key_idx:
            key_list_i = key_list[i].detach().clone()
            cosine_sim_kk = F.cosine_similarity(selected_key, key_list_i, dim=0).squeeze()
            loss_kk += cosine_sim_kk
    if len(key_list) > 1:
        loss += lambda_kk * loss_kk / (len(key_list) - 1)


    return lambda_agnostic * loss
