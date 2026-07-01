"""
End-to-end demo script for satellite frame interpolation.
Shows the complete workflow from data loading to visualization.
"""
import torch
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import our modules
from backend.models.rife import SatelliteRIFE
from backend.utils.metrics import MetricsCalculator, print_metrics


def demo_model_inference():
    """Demonstrate model inference on synthetic satellite data."""
    print("=" * 70)
    print("Satellite Frame Interpolation Demo")
    print("=" * 70)
    
    # 1. Create synthetic satellite-like data
    print("\n[1/5] Generating synthetic satellite frames...")
    
    def generate_satellite_frame(seed, size=(256, 256)):
        """Generate a synthetic thermal IR satellite frame."""
        h, w = size
        # Base pattern simulating Earth's thermal radiation
        y, x = np.mgrid[0:h, 0:w]
        
        # Large-scale temperature gradient (equator to poles)
        base_temp = 0.4 + 0.3 * np.sin(x / w * np.pi)
        
        # Cloud patterns
        cloud1 = 0.2 * np.exp(-((x - w*0.3)**2 + (y - h*0.4)**2) / (w*0.15)**2)
        cloud2 = 0.15 * np.exp(-((x - w*0.7)**2 + (y - h*0.6)**2) / (w*0.1)**2)
        cloud3 = 0.1 * np.sin(x/20 + seed) * np.cos(y/15 + seed*0.5)
        
        # Noise
        noise = 0.02 * np.random.randn(h, w)
        
        frame = base_temp + cloud1 + cloud2 + cloud3 + noise
        return np.clip(frame, 0, 1).astype(np.float32)
    
    # Generate three consecutive frames
    frame_0 = generate_satellite_frame(seed=0)
    frame_1 = generate_satellite_frame(seed=1)  # This will be our "prediction target"
    frame_2 = generate_satellite_frame(seed=2)
    
    print(f"  Frame 0 shape: {frame_0.shape}, range: [{frame_0.min():.3f}, {frame_0.max():.3f}]")
    print(f"  Frame 1 shape: {frame_1.shape}, range: [{frame_1.min():.3f}, {frame_1.max():.3f}]")
    print(f"  Frame 2 shape: {frame_2.shape}, range: [{frame_2.min():.3f}, {frame_2.max():.3f}]")
    
    # 2. Initialize model
    print("\n[2/5] Initializing SatelliteRIFE model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    
    model = SatelliteRIFE(satellite_optimized=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")
    print(f"  Model size: {n_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    # 3. Run inference
    print("\n[3/5] Running frame interpolation (t=0.5)...")
    
    # Convert to tensors
    frame_0_t = torch.from_numpy(frame_0).unsqueeze(0).unsqueeze(0).to(device)
    frame_2_t = torch.from_numpy(frame_2).unsqueeze(0).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        pred_frame = model(frame_0_t, frame_2_t, timestep=0.5)
    
    pred_np = pred_frame[0, 0].cpu().numpy()
    print(f"  Predicted frame shape: {pred_np.shape}")
    print(f"  Predicted frame range: [{pred_np.min():.3f}, {pred_np.max():.3f}]")
    
    # 4. Evaluate quality
    print("\n[4/5] Computing quality metrics...")
    
    calculator = MetricsCalculator(device=device)
    metrics = calculator.compute_all(pred_np, frame_1)
    
    print_metrics(metrics, "Interpolation Quality")
    
    # 5. Visualize results
    print("\n[5/5] Generating visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Satellite Frame Interpolation Results', fontsize=16, fontweight='bold')
    
    # Custom thermal colormap
    from matplotlib.colors import LinearSegmentedColormap
    thermal_cmap = LinearSegmentedColormap.from_list(
        'thermal',
        ['#000080', '#0000ff', '#00ffff', '#00ff00', '#ffff00', '#ff8000', '#ff0000', '#800000']
    )
    
    # Frame 0
    axes[0, 0].imshow(frame_0, cmap=thermal_cmap, vmin=0, vmax=1)
    axes[0, 0].set_title('Frame t=0 (Input)', fontweight='bold')
    axes[0, 0].axis('off')
    
    # Frame 2
    axes[0, 1].imshow(frame_2, cmap=thermal_cmap, vmin=0, vmax=1)
    axes[0, 1].set_title('Frame t=1 (Input)', fontweight='bold')
    axes[0, 1].axis('off')
    
    # Prediction
    axes[0, 2].imshow(pred_np, cmap=thermal_cmap, vmin=0, vmax=1)
    axes[0, 2].set_title('Predicted t=0.5 (AI)', fontweight='bold', color='purple')
    axes[0, 2].axis('off')
    
    # Ground Truth
    axes[1, 0].imshow(frame_1, cmap=thermal_cmap, vmin=0, vmax=1)
    axes[1, 0].set_title('Ground Truth t=0.5', fontweight='bold', color='green')
    axes[1, 0].axis('off')
    
    # Difference
    diff = np.abs(pred_np - frame_1)
    im = axes[1, 1].imshow(diff, cmap='hot', vmin=0, vmax=np.percentile(diff, 95))
    axes[1, 1].set_title('Absolute Difference', fontweight='bold', color='red')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
    
    # Metrics text
    axes[1, 2].axis('off')
    metrics_text = "Quality Metrics\n" + "="*30 + "\n\n"
    for key, value in metrics.items():
        metrics_text += f"{key.upper():10s}: {value:.4f}\n"
    axes[1, 2].text(0.1, 0.5, metrics_text, transform=axes[1, 2].transAxes,
                    fontsize=11, verticalalignment='center',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    output_dir = Path('./outputs')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'demo_result.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Visualization saved to: {output_path}")
    
    # Also demonstrate multi-frame interpolation
    print("\n" + "=" * 70)
    print("Bonus: Multi-Frame Interpolation Demo")
    print("=" * 70)
    
    with torch.no_grad():
        frames_3x = model.generate_multi_frame(frame_0_t, frame_2_t, n_frames=3)
    
    print(f"  Generated {len(frames_3x)} intermediate frames")
    for i, f in enumerate(frames_3x):
        arr = f[0, 0].cpu().numpy()
        print(f"  Frame {i+1}/3: shape={arr.shape}, range=[{arr.min():.3f}, {arr.max():.3f}]")
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Train on real satellite data: python backend/train.py --train-dir ./datasets")
    print("  2. Run inference: python backend/interpolate.py --model ./checkpoints/best.pt --input ./data")
    print("  3. Launch dashboard: cd backend && uvicorn api:app --reload")
    print()
    
    return metrics


if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run demo
    metrics = demo_model_inference()
