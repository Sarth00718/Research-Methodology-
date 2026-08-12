import copy
import torch

def aggregate_weights(client_results):
    """
    Performs Federated Averaging (FedAvg).
    
    client_results: List of tuples (state_dict, num_samples) from each client.
    Returns: The aggregated state_dict.
    """
    total_samples = sum([num_samples for _, num_samples in client_results])
    
    # Initialize aggregated weights using the first client's state_dict keys
    aggregated_weights = copy.deepcopy(client_results[0][0])
    
    # Zero out the weights initially
    for key in aggregated_weights.keys():
        aggregated_weights[key] = torch.zeros_like(aggregated_weights[key], dtype=torch.float32)
        
    for client_state_dict, num_samples in client_results:
        weight = num_samples / total_samples
        for key in aggregated_weights.keys():
            # Ensure we are using float32 for aggregation to prevent integer type issues
            if client_state_dict[key].dtype in [torch.int64, torch.long, torch.bool]:
                aggregated_weights[key] += (client_state_dict[key].float() * weight).type(client_state_dict[key].dtype)
            else:
                aggregated_weights[key] += client_state_dict[key] * weight
                
    return aggregated_weights
