# Percepta - Crowd Movement Prediction System

A real-time crowd tracking and movement prediction system using computer vision and deep learning. The system detects individuals from top-down video footage, tracks their movement, predicts future positions using LSTM neural networks, and provides risk analysis for crowd safety.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Training the Model](#training-the-model)
5. [Running the System](#running-the-system)
6. [Controls](#controls)
7. [Module Documentation](#module-documentation)
8. [Troubleshooting](#troubleshooting)

---

## System Overview

Percepta performs the following tasks:

1. **Person Detection**: Uses background subtraction optimized for top-down camera views to detect individuals in video frames.

2. **Object Tracking**: Assigns persistent IDs to detected individuals using centroid-based tracking, maintaining identity across frames.

3. **Data Collection**: Records position history, velocity, acceleration, and other features for each tracked person.

4. **Movement Prediction**: Uses an LSTM neural network to predict future positions up to 10 minutes ahead.

5. **Risk Analysis**: Evaluates crowd density, velocity patterns, and convergence to detect potential safety hazards including stampede conditions.

6. **Visualization**: Displays real-time tracking, predicted trajectories, density heatmaps, and risk indicators.

---

## Architecture

```
crowd_tracker.py          - Core detection and tracking
crowd_data_collector.py   - Movement data collection and feature extraction
synthetic_crowd_generator.py - Generates training data using Social Force Model
crowd_predictor.py        - LSTM/GRU neural network for trajectory prediction
risk_analyzer.py          - Safety risk evaluation and stampede detection
prediction_visualizer.py  - Extended visualization with predictions
crowd_prediction_system.py - Main integration and entry point
```

### How Prediction Works

Since real crowd footage for training is often unavailable, the system uses a **Social Force Model** to generate synthetic training data. This physics-based simulation models:

- Pedestrian desired velocity toward goals
- Repulsion forces between individuals
- Wall/boundary avoidance
- Panic and evacuation behaviors

The LSTM model learns from these simulated trajectories and generalizes to real crowd movement patterns.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- NVIDIA GPU with CUDA support (recommended)
- 4GB+ GPU memory for training

### Step 1: Clean Installation

If you have run this system before, delete existing generated files:

```bash
# Windows
del crowd_predictor_model.pth
del synthetic_training_data.json
del crowd_tracking_data.json

# Linux/Mac
rm -f crowd_predictor_model.pth synthetic_training_data.json crowd_tracking_data.json
```

### Step 2: Install Dependencies

#### For NVIDIA GPU (Recommended)

First, check your CUDA version:

```bash
nvidia-smi
```

Look for "CUDA Version" in the output (e.g., 11.8, 12.1).

Install PyTorch with matching CUDA version:

**For CUDA 11.8:**
```bash
pip install opencv-python numpy scipy
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install opencv-python numpy scipy
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 12.4:**
```bash
pip install opencv-python numpy scipy
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

#### For CPU Only (Slower)

```bash
pip install opencv-python numpy scipy torch torchvision
```

### Step 3: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

Expected output (with GPU):
```
PyTorch: 2.x.x
CUDA Available: True
GPU: NVIDIA GeForce RTX 3050 Laptop GPU
```

---

## Training the Model

Before running the visualization, you must train the prediction model. The training uses synthetically generated crowd data.

### Step 1: Run Training

```bash
python crowd_prediction_system.py --train
```

### What Happens During Training

1. **Synthetic Data Generation** (1-2 minutes)
   - Generates 5 crowd simulations with 20 agents each
   - Simulates normal movement, high-density, panic, and evacuation scenarios
   - Creates realistic trajectories using physics-based Social Force Model

2. **Data Preparation**
   - Extracts sequences of positions and velocities
   - Normalizes coordinates for neural network input

3. **Model Training** (1-5 minutes depending on GPU)
   - Trains LSTM network for 20 epochs
   - Uses mixed precision (FP16) for faster GPU training
   - Displays loss metrics and GPU memory usage per epoch

### Expected Output

```
==================================================
TRAINING PREDICTION MODEL
==================================================

1. Generating synthetic training data...
   (This may take 1-2 minutes)
      Simulating... 100%
   Generated simulation 1/5 (normal, 20 agents)
   Generated simulation 2/5 (high_density, 20 agents)
   Generated simulation 3/5 (panic, 20 agents)
   Generated simulation 4/5 (evacuation, 20 agents)
   Generated simulation 5/5 (normal, 20 agents)

2. Preparing data for training...
   Processing simulation 5/5...
   Training samples: 4250
   Input shape: (4250, 30, 6)
   Output shape: (4250, 30, 2)

3. Training model...
   Using device: cuda
   Creating data loaders...
   Training batches: 106, Validation batches: 27
   Pin memory: True
   Using mixed precision (AMP) for faster training

   Starting training...
   Epoch 1/20 - Train: 0.023456, Val: 0.019876 | GPU: 145MB
   Epoch 2/20 - Train: 0.015432, Val: 0.014567 | GPU: 145MB
   ...
   Epoch 20/20 - Train: 0.002345, Val: 0.002456 | GPU: 145MB

4. Saving model...
Model saved to crowd_predictor_model.pth

==================================================
Training complete!
Model saved to: crowd_predictor_model.pth
==================================================
```

### Training Parameters

To modify training parameters, edit `crowd_prediction_system.py`:

```python
dataset = generator.generate_training_data(
    num_simulations=5,      # More simulations = more diverse data
    agents_per_sim=20,      # More agents = denser crowds
    duration=20.0           # Longer duration = longer trajectories
)
```

---

## Running the System

### Step 1: Configure Video Path

Edit `crowd_prediction_system.py` and set your video path:

```python
VIDEO_PATH = r"C:\path\to\your\video.mp4"  # Windows
# or
VIDEO_PATH = "/path/to/your/video.mp4"     # Linux/Mac
```

### Step 2: Run the System

```bash
python crowd_prediction_system.py
```

### What You Will See

Three windows will open:

1. **Crowd Tracking - Main View**
   - Original video with tracking overlays
   - Green rectangles: Raw detections
   - Colored crosshairs and circles: Tracked individuals with IDs
   - Risk status panel in top-left corner
   - Warning messages when risk is elevated

2. **Dot Matrix**
   - Simplified view showing person positions as dots
   - Movement trails showing recent paths
   - Count of tracked individuals

3. **Prediction and Risk**
   - Predicted future positions (faded dots and lines)
   - Risk indicator panel with gauges
   - Density, velocity, convergence, and stampede probability
   - Hotspot markers for high-risk areas
   - Timeline showing prediction horizon

---

## Controls

| Key | Action |
|-----|--------|
| Q | Quit the application |
| P | Pause/Resume video |
| R | Reset tracker (clear all IDs) |
| T | Toggle prediction visualization |
| H | Toggle density heatmap |
| + | Increase prediction horizon |
| - | Decrease prediction horizon |
| S | Save tracking data to JSON |

---

## Module Documentation

### crowd_tracker.py

Core tracking functionality:

- `PersonDetector`: Detects people using background subtraction or HOG
- `CentroidTracker`: Maintains object IDs across frames using centroid matching
- `DotMatrixVisualizer`: Creates simple dot visualization

### crowd_data_collector.py

Data management:

- `TrackingPoint`: Single position with timestamp
- `ObjectTrack`: Complete trajectory for one person with velocity/acceleration
- `CrowdDataCollector`: Aggregates all tracking data, computes statistics
- `FeatureExtractor`: Normalizes and prepares features for model input

### synthetic_crowd_generator.py

Training data generation:

- `SocialForceModel`: Physics simulation of pedestrian dynamics
- `SyntheticCrowdGenerator`: Creates complete crowd simulations
- `BehaviorMode`: Normal, high-density, panic, evacuation scenarios
- `prepare_pytorch_data()`: Converts trajectories to training tensors

### crowd_predictor.py

Neural network prediction:

- `LSTMPredictor`: LSTM architecture with attention mechanism
- `GRUPredictor`: Lighter GRU alternative
- `CrowdPredictor`: Main interface for training and inference
- Supports GPU acceleration with mixed precision

### risk_analyzer.py

Safety analysis:

- `RiskAnalyzer`: Evaluates multiple risk factors
- `RiskReport`: Complete assessment with recommendations
- `RiskLevel`: SAFE, CAUTION, WARNING, DANGER, CRITICAL
- Detects density thresholds, velocity anomalies, convergence patterns

### prediction_visualizer.py

Advanced visualization:

- `PredictionVisualizer`: Renders predictions and risk data
- `CombinedDisplay`: Multi-panel layout option
- Density heatmaps, movement arrows, hotspot indicators

---

## Troubleshooting

### "CUDA out of memory"

Reduce batch size in `crowd_predictor.py`:
```python
history = predictor.train(X, Y, epochs=20, batch_size=16)  # Reduce from 32
```

### "No GPU detected, using CPU"

1. Verify NVIDIA drivers are installed: `nvidia-smi`
2. Reinstall PyTorch with correct CUDA version (see Installation)
3. Check CUDA toolkit installation

### "Could not open video file"

1. Verify the video path is correct
2. Ensure the video codec is supported (H.264 recommended)
3. Try converting video: `ffmpeg -i input.mp4 -c:v libx264 output.mp4`

### Model loading errors (shape mismatch)

Delete the old model and retrain:
```bash
del crowd_predictor_model.pth
python crowd_prediction_system.py --train
```

### Detection not working well

Adjust detection parameters in `crowd_tracker.py`:
```python
# For larger people in frame (4K video)
min_area = int(500 * self.scale_factor)   # Increase minimum
max_area = int(150000 * self.scale_factor) # Increase maximum
```

### Tracking IDs constantly changing

Increase tracking distance in `crowd_prediction_system.py`:
```python
max_distance = int(120 * self.scale_factor)  # Increase from 80
```

---

## Performance Tips

1. **Use GPU**: Training is 10-20x faster with CUDA
2. **Lower resolution**: Resize 4K video to 1080p for faster processing
3. **Reduce prediction interval**: Edit `self.prediction_interval = 30` for less frequent predictions
4. **Disable heatmap**: Press H to toggle off density heatmap

---

## License

This project is provided as-is for educational and research purposes.

---

## References

- Helbing, D., and Molnar, P. (1995). Social Force Model for Pedestrian Dynamics
- OpenCV Background Subtraction: MOG2 Algorithm
- PyTorch LSTM Documentation
