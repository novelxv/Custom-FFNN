import numpy as np
from tqdm import tqdm
import json

def train(model, X_train, y_train, X_val, y_val, loss_fn, loss_deriv, batch_size=64, learning_rate=0.01, epochs=10, verbose=1):
    num_samples = X_train.shape[1]
    num_batches = num_samples // batch_size

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs): 
        total_train_loss = 0
        if verbose == 1:
            pbar = tqdm(total=num_batches, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        # Mini-batch training
        for i in range(0, num_samples, batch_size):
            X_batch = X_train[:, i:i+batch_size]
            y_batch = y_train[:, i:i+batch_size]
            
            # Forward pass
            y_pred = model.forward(X_batch)
            
            # Compute loss
            loss = loss_fn(y_batch, y_pred)
            total_train_loss += loss
            
            # Backward pass
            model.backward(y_batch, loss_fn, loss_deriv)
            model.update_weights(learning_rate)
            
            if verbose == 1:
                pbar.update(1)
        
        if verbose == 1:
            pbar.close()
        
        # Compute average training loss
        avg_train_loss = total_train_loss / num_batches
        
        # Validation loss
        y_val_pred = model.forward(X_val)
        val_loss = loss_fn(y_val, y_val_pred)
        
        # Store history
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        
        if verbose == 1:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    return history