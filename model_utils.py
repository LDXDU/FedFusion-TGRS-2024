import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import svd_lowrank

def gather_trans(in_t: torch.Tensor):
    """
    Transmit through all_gather and directly return the processed tensor list.
    """
    in_t = in_t.contiguous()
    world_size = dist.get_world_size()
    tensor_list = [torch.empty_like(in_t) for _ in range(world_size)]
    dist.all_gather(tensor_list, in_t)
    return tensor_list


def svd_gather(in_t: torch.Tensor, k=None, sparse_ratio=0.3, quant_bits=8, type = 'default'):
    """
    Perform SVD decomposition on the input tensor, transmit through all_gather, and finally reconstruct to the original shape.
    in_t: Input 4D tensor.
    k: The number of top k rows and columns selected during SVD decomposition.
    Returns the reconstructed tensor list.
    """
        
    ori_shape = in_t.shape
    in_t = in_t.view(ori_shape[0] * ori_shape[1], -1)

    original_size = in_t.element_size() * in_t.numel()  # 字节
    
    if k == None:
        k = adaptive_k_selection(in_t)
        # k = 5
    U, sigma, VT = svd_lowrank(in_t, q=k, niter=20)
    
    if type == 'default':
        # Transmit the SVD decomposed tensors through all_gather
        U_list = gather_trans(U)
        sigma_list = gather_trans(sigma)
        VT_list = gather_trans(VT)

        # Reconstruct the original tensor
        out_t = []
        for i in range(len(U_list)):
            sigma_diag = torch.diag_embed(sigma_list[i])  # Convert sigma to a diagonal matrix
            re_t = torch.mm(U_list[i], torch.mm(sigma_diag, VT_list[i].t()))
            re_t = re_t.view(ori_shape)
            out_t.append(re_t)

        compressed_size = (U.element_size()*U.numel() + 
                        sigma.element_size()*sigma.numel() + 
                        VT.element_size()*VT.numel())

    elif type == 'enhanced':

        VT_T = VT.T 
        U_sparse, U_indices, U_values = sparse_compress(U, k=sparse_ratio)
        VT_sparse, VT_indices, VT_values = sparse_compress(VT_T, k=sparse_ratio)
        
        # Quantization
        U_quant, U_scale, U_zp = quantize_tensor(U_sparse, num_bits=quant_bits)
        sigma_quant, sigma_scale, sigma_zp = quantize_tensor(sigma, num_bits=quant_bits)
        VT_quant, VT_scale, VT_zp = quantize_tensor(VT_sparse, num_bits=quant_bits) 

        U_q = gather_trans(U_quant); U_s = gather_trans(U_scale); U_zp_l = gather_trans(U_zp)
        sigma_q = gather_trans(sigma_quant); sigma_s = gather_trans(sigma_scale);  sigma_zp_l = gather_trans(sigma_zp)
        VT_q = gather_trans(VT_quant); VT_s = gather_trans(VT_scale); VT_z = gather_trans(VT_zp)
        
        compressed_size = sum([t.element_size() * t.numel() for t in [U_quant,U_scale,U_zp, sigma_quant,sigma_scale,sigma_zp, VT_quant,VT_scale,VT_zp]]) 
        
        # Reconstruct features
        out_t = []
        for i in range(dist.get_world_size()):
            # Dequantization
            U_dequant = dequantize_tensor(U_q[i], U_s[i], U_zp_l[i])      # [M,k]
            sigma_dequant = dequantize_tensor(sigma_q[i], sigma_s[i], sigma_zp_l[i])  # [k]
            VT_dequant = dequantize_tensor(VT_q[i], VT_s[i], VT_z[i])
            re_t = U_dequant @ (torch.diag(sigma_dequant) @ VT_dequant)
            out_t.append(re_t.view(ori_shape))
    else:
        raise KeyError("svd type error")
    if k == None:
        print(f"Compression ratio: {original_size/compressed_size:.1f}x")
    return out_t

def quantize_tensor(tensor, num_bits=8):
    """Dynamic range quantization"""
    min_val = tensor.min()
    max_val = tensor.max()
    scale = (max_val - min_val) / (2**num_bits - 1)
    zero_point = torch.round(-min_val / scale)
    q_tensor = torch.clamp(torch.round((tensor - min_val) / scale), 0, 2**num_bits-1).to(torch.uint8)
    return q_tensor, scale, zero_point

def dequantize_tensor(q_tensor, scale, zero_point):
    """Dequantization"""
    return scale * (q_tensor.float() - zero_point)

def sparse_compress(in_t: torch.Tensor, k=0.2):
    """Top-K sparsification"""
    k_val = int(k * in_t.numel())
    values, indices = torch.topk(in_t.abs().reshape(-1), k_val)
    sparse_mask = torch.zeros_like(in_t).reshape(-1)
    sparse_mask[indices] = 1
    return in_t * sparse_mask.view(in_t.shape), indices, values


def adaptive_k_selection(in_tensor, max_error=0.25):
    """
    Automatically select k value based on energy threshold
    Args:
        in_tensor: Input tensor
        max_error: Maximum allowable energy loss, range (0,1)
    Returns:
        k: Selected rank
    """
    U, s, VT = torch.linalg.svd(in_tensor, full_matrices=False)
    total_energy = torch.cumsum(s**2, dim=0) / (torch.sum(s**2) + 1e-10)
    mask = total_energy > (1 - max_error)
    if mask.any():
        k = torch.where(mask)[0][0] + 1
    else:
        k = len(s) 
    return k.item()
