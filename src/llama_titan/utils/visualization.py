"""
Visualization utilities for the Titans architecture.
"""

import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Optional

def plot_memory_usage(component_budgets: Dict[str, int], save_path: str = 'memory_usage.png'):
    """Generate memory usage visualization."""
    # Convert bytes to GB for visualization
    usage_gb = {k: v / (1024**3) for k, v in component_budgets.items()}
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    components = list(usage_gb.keys())
    usage = [usage_gb[k] for k in components]
    
    # Plot bars
    bars = plt.bar(components, usage)
    
    # Customize plot
    plt.title('VRAM Usage by Component')
    plt.ylabel('Memory Usage (GB)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}GB',
                ha='center', va='bottom')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_memory_timeline(
    usage_history: List[Dict[str, float]],
    save_path: str = 'memory_timeline.png'
):
    """Generate timeline visualization of memory usage."""
    plt.figure(figsize=(12, 6))
    
    # Extract components and timestamps
    components = list(usage_history[0].keys())
    timestamps = range(len(usage_history))
    
    # Plot line for each component
    for component in components:
        usage = [point[component] / (1024**3) for point in usage_history]
        plt.plot(timestamps, usage, label=component, marker='o')
    
    # Customize plot
    plt.title('Memory Usage Timeline')
    plt.xlabel('Time Step')
    plt.ylabel('Memory Usage (GB)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_component_distribution(
    memory_manager,
    save_path: str = 'component_distribution.png'
):
    """Visualize component distribution across GPUs."""
    if not torch.cuda.is_available():
        return
    
    plt.figure(figsize=(10, 6))
    
    # Get device mapping
    device_map = memory_manager.device_map
    num_gpus = torch.cuda.device_count()
    
    # Create mapping of GPU to components
    gpu_components = {i: [] for i in range(num_gpus)}
    for component, gpu in device_map.items():
        gpu_components[gpu].append(component)
    
    # Plot bars for each GPU
    x = range(num_gpus)
    total_height = [0] * num_gpus
    
    for component in ['core', 'long_term', 'persistent']:
        heights = []
        for gpu in range(num_gpus):
            if component in gpu_components[gpu]:
                height = memory_manager.component_budgets[component] / (1024**3)
            else:
                height = 0
            heights.append(height)
        
        plt.bar(x, heights, bottom=total_height, label=component)
        total_height = [t + h for t, h in zip(total_height, heights)]
    
    # Customize plot
    plt.title('Component Distribution Across GPUs')
    plt.xlabel('GPU ID')
    plt.ylabel('Memory Allocation (GB)')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i in range(num_gpus):
        if total_height[i] > 0:
            plt.text(i, total_height[i], f'{total_height[i]:.1f}GB',
                    ha='center', va='bottom')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
