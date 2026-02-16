"""
Crowd Movement Prediction Model
LSTM-based model for predicting future crowd positions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import os

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("Warning: PyTorch not installed. Using fallback simple predictor.")


class CrowdTrajectoryDataset(Dataset):
    """PyTorch Dataset for crowd trajectories"""
    
    def __init__(self, X: np.ndarray, Y: np.ndarray, labels: np.ndarray = None):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)
        self.labels = torch.LongTensor(labels) if labels is not None else None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.X[idx], self.Y[idx], self.labels[idx]
        return self.X[idx], self.Y[idx]


class LSTMPredictor(nn.Module):
    """
    LSTM model for trajectory prediction.
    
    Architecture:
    - Input: [batch, seq_len, input_features]
    - LSTM layers with dropout
    - Output: [batch, future_steps, 2] (x, y positions)
    """
    
    def __init__(self, 
                 input_size: int = 6,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 output_steps: int = 60,
                 dropout: float = 0.2):
        super(LSTMPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_steps = output_steps
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Decoder to predict future positions
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_steps * 2)
        )
    
    def forward(self, x):
        # x shape: [batch, seq_len, input_size]
        batch_size = x.size(0)
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out shape: [batch, seq_len, hidden_size]
        
        # Attention weights
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Context vector
        context = torch.sum(lstm_out * attn_weights, dim=1)
        # context shape: [batch, hidden_size]
        
        # Decode to future positions
        output = self.decoder(context)
        output = output.view(batch_size, self.output_steps, 2)
        
        return output


class GRUPredictor(nn.Module):
    """
    GRU model for trajectory prediction (lighter than LSTM).
    """
    
    def __init__(self, 
                 input_size: int = 6,
                 hidden_size: int = 96,
                 num_layers: int = 2,
                 output_steps: int = 60,
                 dropout: float = 0.2):
        super(GRUPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_steps = output_steps
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_steps * 2)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        _, h_n = self.gru(x)
        h_n = h_n[-1]  # Take last layer's hidden state
        output = self.fc(h_n)
        output = output.view(batch_size, self.output_steps, 2)
        return output


class SimplePredictor:
    """
    Simple predictor using velocity extrapolation (fallback when no PyTorch).
    """
    
    def __init__(self):
        self.smoothing = 0.3
    
    def predict(self, history: np.ndarray, future_steps: int) -> np.ndarray:
        """
        Predict future positions using velocity extrapolation.
        
        Args:
            history: [seq_len, features] where features = [x, y, vx, vy, ...]
            future_steps: Number of steps to predict
        
        Returns:
            Predicted positions [future_steps, 2]
        """
        if len(history) < 2:
            return np.zeros((future_steps, 2))
        
        # Use last position and average recent velocity
        last_pos = history[-1, :2]
        velocities = history[-10:, 2:4] if len(history) >= 10 else history[:, 2:4]
        avg_velocity = np.mean(velocities, axis=0)
        
        predictions = []
        current_pos = last_pos.copy()
        
        for i in range(future_steps):
            current_pos = current_pos + avg_velocity * 0.033  # dt
            predictions.append(current_pos.copy())
        
        return np.array(predictions)


class CrowdPredictor:
    """
    Main predictor class that handles training and inference.
    Predicts crowd movement up to 10 minutes into the future.
    """
    
    def __init__(self, 
                 model_type: str = "lstm",
                 input_features: int = 6,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 seq_length: int = 30,
                 output_steps: int = 60,
                 device: str = None):
        """
        Initialize the predictor.
        
        Args:
            model_type: "lstm", "gru", or "simple"
            input_features: Number of input features per timestep
            hidden_size: LSTM/GRU hidden size
            num_layers: Number of recurrent layers
            seq_length: Input sequence length
            output_steps: Number of future steps to predict directly
            device: "cuda" or "cpu"
        """
        self.model_type = model_type
        self.input_features = input_features
        self.seq_length = seq_length
        self.output_steps = output_steps
        
        # Frame parameters
        self.frame_width = 1920
        self.frame_height = 1080
        self.fps = 30
        
        if device is None:
            if PYTORCH_AVAILABLE:
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
                    print(f"CUDA version: {torch.version.cuda}")
                else:
                    self.device = torch.device("cpu")
                    print("No GPU detected, using CPU")
            else:
                self.device = "cpu"
        else:
            self.device = torch.device(device) if PYTORCH_AVAILABLE else device
        
        # Initialize model
        if PYTORCH_AVAILABLE and model_type != "simple":
            if model_type == "lstm":
                self.model = LSTMPredictor(
                    input_size=input_features,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    output_steps=output_steps
                ).to(self.device)
            else:  # gru
                self.model = GRUPredictor(
                    input_size=input_features,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    output_steps=output_steps
                ).to(self.device)
            
            self.optimizer = None
            self.criterion = nn.MSELoss()
        else:
            self.model = SimplePredictor()
        
        self.is_trained = False
    
    def set_frame_params(self, width: int, height: int, fps: int = 30):
        """Set frame parameters for coordinate normalization"""
        self.frame_width = width
        self.frame_height = height
        self.fps = fps
    
    def train(self, 
              X: np.ndarray, 
              Y: np.ndarray,
              epochs: int = 50,
              batch_size: int = 64,
              learning_rate: float = 0.001,
              validation_split: float = 0.2,
              verbose: bool = True) -> Dict:
        """
        Train the prediction model.
        
        Args:
            X: Input sequences [num_samples, seq_length, features]
            Y: Target positions [num_samples, future_steps, 2]
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            validation_split: Fraction of data for validation
            verbose: Print training progress
        
        Returns:
            Training history dictionary
        """
        if not PYTORCH_AVAILABLE or self.model_type == "simple":
            print("Training not available for simple predictor (PyTorch not installed)")
            print("The system will use simple velocity extrapolation instead.")
            self.is_trained = True
            return {}
        
        print(f"   Using device: {self.device}", flush=True)
        
        # Split data
        n_samples = len(X)
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)
        
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]
        
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]
        
        # Create data loaders with GPU optimization
        print(f"   Creating data loaders...", flush=True)
        train_dataset = CrowdTrajectoryDataset(X_train, Y_train)
        val_dataset = CrowdTrajectoryDataset(X_val, Y_val)
        
        # Use pin_memory for faster GPU transfer if using CUDA
        use_pin_memory = self.device.type == "cuda"
        num_workers = 0  # Keep at 0 for Windows compatibility
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            pin_memory=use_pin_memory,
            num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            pin_memory=use_pin_memory,
            num_workers=num_workers
        )
        print(f"   Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}", flush=True)
        print(f"   Pin memory: {use_pin_memory}", flush=True)
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float('inf')
        
        # Enable mixed precision for faster GPU training
        use_amp = self.device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        if use_amp:
            print("   Using mixed precision (AMP) for faster training", flush=True)
        
        print("\n   Starting training...", flush=True)
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                
                self.optimizer.zero_grad()
                
                # Use mixed precision if available
                if use_amp:
                    with torch.cuda.amp.autocast():
                        output = self.model(batch_x)
                        loss = self.criterion(output, batch_y)
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    output = self.model(batch_x)
                    loss = self.criterion(output, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)
                    
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            output = self.model(batch_x)
                            loss = self.criterion(output, batch_y)
                    else:
                        output = self.model(batch_x)
                        loss = self.criterion(output, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            
            if verbose:
                gpu_mem = ""
                if self.device.type == "cuda":
                    mem_used = torch.cuda.memory_allocated() / 1024**2
                    gpu_mem = f" | GPU: {mem_used:.0f}MB"
                print(f"   Epoch {epoch + 1}/{epochs} - "
                      f"Train: {train_loss:.6f}, Val: {val_loss:.6f}{gpu_mem}", flush=True)
        
        # Load best model
        self.model.load_state_dict(self.best_state)
        self.is_trained = True
        
        if verbose:
            print(f"\nTraining complete. Best validation loss: {best_val_loss:.6f}")
        
        return history
    
    def predict_single(self, sequence: np.ndarray) -> np.ndarray:
        """
        Predict future positions for a single agent.
        
        Args:
            sequence: [seq_length, features] normalized input sequence
        
        Returns:
            Predicted positions [output_steps, 2] in normalized coordinates
        """
        if not PYTORCH_AVAILABLE or self.model_type == "simple":
            return self.model.predict(sequence, self.output_steps)
        
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            output = self.model(x)
            return output.cpu().numpy()[0]
    
    def predict_future(self, 
                       current_state: Dict,
                       future_seconds: float = 600.0,  # 10 minutes
                       data_collector=None) -> Dict:
        """
        Predict future positions for all tracked objects.
        
        Args:
            current_state: Dictionary from CrowdDataCollector.get_current_state()
            future_seconds: How far into the future to predict
            data_collector: CrowdDataCollector instance for history
        
        Returns:
            Dictionary with predicted positions for each object ID
        """
        predictions = {}
        future_frames = int(future_seconds * self.fps)
        
        for obj_id, state in current_state.items():
            # Get history sequence
            if data_collector and obj_id in data_collector.tracks:
                track = data_collector.tracks[obj_id]
                seq_data = track.get_sequence_data(self.seq_length)
                
                if seq_data is not None:
                    # Normalize
                    seq_normalized = seq_data.copy()
                    seq_normalized[:, 0] /= self.frame_width
                    seq_normalized[:, 1] /= self.frame_height
                    seq_normalized[:, 2] /= self.frame_width
                    seq_normalized[:, 3] /= self.frame_height
                    
                    # Predict
                    pred = self._iterative_predict(seq_normalized, future_frames)
                    
                    # Denormalize
                    pred[:, 0] *= self.frame_width
                    pred[:, 1] *= self.frame_height
                    
                    predictions[obj_id] = pred
                else:
                    # Not enough history, use simple extrapolation
                    predictions[obj_id] = self._simple_extrapolate(state, future_frames)
            else:
                predictions[obj_id] = self._simple_extrapolate(state, future_frames)
        
        return predictions
    
    def _iterative_predict(self, sequence: np.ndarray, total_steps: int) -> np.ndarray:
        """
        Iteratively predict far into the future by feeding predictions back.
        """
        all_predictions = []
        current_seq = sequence.copy()
        
        while len(all_predictions) < total_steps:
            # Predict next chunk
            pred_chunk = self.predict_single(current_seq)
            all_predictions.extend(pred_chunk)
            
            if len(all_predictions) >= total_steps:
                break
            
            # Create new sequence from predictions
            new_points = []
            for i, (x, y) in enumerate(pred_chunk):
                if i > 0:
                    vx = (pred_chunk[i, 0] - pred_chunk[i-1, 0]) * self.fps
                    vy = (pred_chunk[i, 1] - pred_chunk[i-1, 1]) * self.fps
                else:
                    vx = current_seq[-1, 2]
                    vy = current_seq[-1, 3]
                speed = np.sqrt(vx**2 + vy**2)
                direction = np.arctan2(vy, vx)
                new_points.append([x, y, vx, vy, speed, direction])
            
            new_points = np.array(new_points)
            
            # Shift sequence
            if len(new_points) >= self.seq_length:
                current_seq = new_points[-self.seq_length:]
            else:
                current_seq = np.vstack([
                    current_seq[len(new_points):],
                    new_points
                ])
        
        return np.array(all_predictions[:total_steps])
    
    def _simple_extrapolate(self, state: Dict, steps: int) -> np.ndarray:
        """Simple linear extrapolation when model can't be used"""
        predictions = []
        x, y = state["x"], state["y"]
        vx = state.get("vx", 0)
        vy = state.get("vy", 0)
        
        dt = 1.0 / self.fps
        for _ in range(steps):
            x += vx * dt
            y += vy * dt
            # Keep within bounds
            x = np.clip(x, 0, self.frame_width)
            y = np.clip(y, 0, self.frame_height)
            predictions.append([x, y])
        
        return np.array(predictions)
    
    def save_model(self, filepath: str):
        """Save trained model to file"""
        if not PYTORCH_AVAILABLE or self.model_type == "simple":
            print("Cannot save simple predictor")
            return
        
        # Get hidden_size from model
        hidden_size = self.model.hidden_size if hasattr(self.model, 'hidden_size') else 128
        num_layers = self.model.num_layers if hasattr(self.model, 'num_layers') else 2
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'input_features': self.input_features,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'seq_length': self.seq_length,
            'output_steps': self.output_steps,
            'frame_width': self.frame_width,
            'frame_height': self.frame_height,
            'fps': self.fps
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from file"""
        if not PYTORCH_AVAILABLE:
            print("PyTorch not available, cannot load model")
            return
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Get saved configuration
        self.model_type = checkpoint['model_type']
        self.input_features = checkpoint['input_features']
        self.seq_length = checkpoint['seq_length']
        self.output_steps = checkpoint['output_steps']
        self.frame_width = checkpoint.get('frame_width', 1920)
        self.frame_height = checkpoint.get('frame_height', 1080)
        self.fps = checkpoint.get('fps', 30)
        
        # Recreate model with correct architecture from checkpoint
        # Try to infer hidden_size from saved weights if not in checkpoint
        hidden_size = checkpoint.get('hidden_size', None)
        if hidden_size is None:
            # Infer from LSTM weight shape
            state_dict = checkpoint['model_state_dict']
            if 'lstm.weight_hh_l0' in state_dict:
                hidden_size = state_dict['lstm.weight_hh_l0'].shape[1]
            elif 'gru.weight_hh_l0' in state_dict:
                hidden_size = state_dict['gru.weight_hh_l0'].shape[1]
            else:
                hidden_size = 128
        
        num_layers = checkpoint.get('num_layers', 2)
        
        if self.model_type == "lstm":
            self.model = LSTMPredictor(
                input_size=self.input_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_steps=self.output_steps
            ).to(self.device)
        else:
            self.model = GRUPredictor(
                input_size=self.input_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_steps=self.output_steps
            ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")
        print(f"  - Output steps: {self.output_steps}")
        print(f"  - Sequence length: {self.seq_length}")


def train_predictor_from_synthetic(save_path: str = "crowd_predictor_model.pth"):
    """
    Train a predictor using synthetic data.
    Convenience function for quick training.
    """
    from synthetic_crowd_generator import SyntheticCrowdGenerator, prepare_pytorch_data
    
    print("Generating synthetic training data...")
    generator = SyntheticCrowdGenerator(width=1920, height=1080)
    
    dataset = generator.generate_training_data(
        num_simulations=20,
        agents_per_sim=50,
        duration=60.0
    )
    
    print("\nPreparing data for training...")
    X, Y, labels = prepare_pytorch_data(dataset, seq_length=30, future_steps=60)
    print(f"Training samples: {len(X)}")
    
    print("\nInitializing predictor...")
    predictor = CrowdPredictor(
        model_type="lstm",
        input_features=6,
        seq_length=30,
        output_steps=60
    )
    
    print("\nTraining model...")
    history = predictor.train(X, Y, epochs=30, batch_size=64)
    
    predictor.save_model(save_path)
    
    return predictor, history


if __name__ == "__main__":
    # Train a model
    predictor, history = train_predictor_from_synthetic()
    print("\nTraining complete!")
