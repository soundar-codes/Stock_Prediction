# =============================================================================
# MODEL.PY - The LSTM Modeling Core
# =============================================================================
# This module builds, trains, and evaluates the LSTM (Long Short-Term Memory)
# neural network for stock price prediction. LSTM is a type of recurrent
# neural network (RNN) particularly good at learning from time series data.
#
# Student: 711524BCS164 | SL.NO: 54
# =============================================================================

import numpy as np                      # NumPy for numerical computations
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
                                        # Metrics for evaluating model performance
import tensorflow as tf                 # TensorFlow deep learning framework
from tensorflow.keras.models import Sequential  # Keras model architecture (stack of layers)
from tensorflow.keras.layers import LSTM, Dropout, Dense  # Layer types for our network
from tensorflow.keras.optimizers import Adam  # Adam optimizer for training
from tensorflow.keras.callbacks import EarlyStopping  # Stops training when validation stops improving

# Import the data loader and checklist logger
from data_loader import load_and_prepare_data
from checklist import ChecklistLogger


def build_lstm_model(look_back: int = 60):
    """
    Build and compile the LSTM neural network architecture.
    
    This function creates a sequential model with 2 LSTM layers for temporal
    learning, 2 Dropout layers to prevent overfitting, and 2 Dense layers for
    final prediction. The architecture is designed specifically for time series.
    
    Architecture Details:
    - Layer 1: LSTM(128) - Learns long-range temporal dependencies
    - Layer 2: Dropout(0.2) - Prevents overfitting
    - Layer 3: LSTM(64) - Compresses sequence into context vector
    - Layer 4: Dropout(0.2) - Additional regularization
    - Layer 5: Dense(32, relu) - Non-linear feature combination
    - Layer 6: Dense(1) - Final price prediction
    
    Args:
        look_back (int): Number of time steps in input sequence (default: 60)
    
    Returns:
        Sequential: Compiled Keras model ready for training
    """
    print("🧠 Building LSTM model...")
    
    # Create a Sequential model (layers stacked linearly)
    model = Sequential()
    
    # === LAYER 1: First LSTM Layer (128 units, return_sequences=True) ===
    # WHY LSTM? LSTM (Long Short-Term Memory) can remember patterns over long sequences
    # 128 units means 128 "memory cells" learning different patterns
    # return_sequences=True outputs the full sequence to feed into next LSTM layer
    model.add(LSTM(
        units=128,                    # Number of LSTM memory cells
        return_sequences=True,        # Return full sequence (needed for stacking LSTMs)
        input_shape=(look_back, 1)    # Input shape: (time_steps, features)
                                      # We have 60 time steps, 1 feature (Close price)
    ))
    # COMMENT: This layer learns long-range temporal dependencies in stock prices
    # It identifies patterns like "if price rose for 5 days, then..."
    
    # === LAYER 2: First Dropout Layer (0.2 = 20% dropout) ===
    # WHY Dropout? During training, randomly set 20% of neuron outputs to 0
    # This prevents the model from memorizing noise in training data
    model.add(Dropout(0.2))
    # COMMENT: Prevents overfitting (model memorising noise instead of learning patterns)
    
    # === LAYER 3: Second LSTM Layer (64 units, return_sequences=False) ===
    # 64 units (fewer than first layer) compresses the learned information
    # return_sequences=False means we only output the final time step
    model.add(LSTM(
        units=64,                     # Fewer units = compression layer
        return_sequences=False        # Only output final result (not full sequence)
    ))
    # COMMENT: Compresses sequence into a single context vector (summary of 60 days)
    
    # === LAYER 4: Second Dropout Layer (0.2) ===
    # More regularization to keep model generalizable
    model.add(Dropout(0.2))
    # COMMENT: Additional dropout for better generalization
    
    # === LAYER 5: First Dense Layer (32 units, ReLU activation) ===
    # Dense = fully connected layer (every neuron connects to every previous output)
    # ReLU (Rectified Linear Unit): f(x) = max(0, x) - introduces non-linearity
    model.add(Dense(units=32, activation='relu'))
    # COMMENT: Non-linear feature combination - learns complex price relationships
    
    # === LAYER 6: Output Dense Layer (1 unit, linear activation) ===
    # Single output neuron = the predicted stock price
    # Linear activation (default) means output is just the weighted sum
    model.add(Dense(units=1))
    # COMMENT: Final predicted price (single value output - the next day's close price)
    
    # === COMPILE THE MODEL ===
    # Configure how the model learns from data
    optimizer = Adam(learning_rate=0.001)  # Adam: Adaptive learning rate optimizer
                                           # 0.001: Step size for weight updates (small = stable)
    
    model.compile(
        optimizer=optimizer,               # How to update weights based on errors
        loss='mean_squared_error'          # MSE: Average squared difference between predicted and actual
                                           # Squaring penalizes large errors more heavily
    )
    
    print("   → Model architecture created with 6 layers")
    print(f"   → Total parameters: {model.count_params():,}")
    
    return model


def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    """
    Train the LSTM model on the training data.
    
    Uses EarlyStopping to prevent overfitting - training stops if validation
    loss doesn't improve for 5 consecutive epochs. The best weights are restored.
    
    Args:
        model: Compiled Keras model
        X_train: Training input sequences
        y_train: Training target values
        X_test: Validation input sequences
        y_test: Validation target values
        epochs (int): Maximum number of training iterations (default: 50)
        batch_size (int): Number of samples per gradient update (default: 32)
    
    Returns:
        History: Keras History object containing training metrics per epoch
    """
    print("\n🏋️ Training model...")
    print(f"   → Epochs: {epochs}, Batch size: {batch_size}")
    print(f"   → Training samples: {len(X_train)}, Validation samples: {len(X_test)}")
    
    # === EARLY STOPPING CALLBACK ===
    # Monitor validation loss - if it doesn't improve for 5 epochs, stop training
    # restore_best_weights=True: Use the best model, not the last one
    early_stop = EarlyStopping(
        monitor='val_loss',           # Watch validation loss
        patience=5,                   # Wait 5 epochs before stopping
        restore_best_weights=True,    # Keep the best weights, not the last
        verbose=1                     # Print messages when stopping
    )
    
    # === TRAIN THE MODEL ===
    # validation_split=0.1: Use 10% of training data for validation during training
    # callbacks=[early_stop]: Use early stopping to prevent overfitting
    history = model.fit(
        X_train, y_train,
        epochs=epochs,                # Maximum epochs (might stop earlier)
        batch_size=batch_size,        # Process 32 samples at a time
        validation_split=0.1,         # 10% of training data for validation
        callbacks=[early_stop],       # Stop early if validation loss plateaus
        verbose=1                     # Show progress bar
    )
    
    # Print training summary
    final_epoch = len(history.history['loss'])
    final_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    print(f"\n[✓] Training completed at epoch {final_epoch}/{epochs}")
    print(f"    Final training loss: {final_loss:.6f}")
    print(f"    Final validation loss: {final_val_loss:.6f}")
    
    return history


def evaluate_model(model, X_test, y_test, scaler):
    """
    Evaluate the trained model on test data and calculate metrics.
    
    Predictions are inverse-transformed back to original price scale
    using the MinMaxScaler. Three metrics are computed:
    - RMSE: Average error magnitude in original currency
    - MAPE: Percentage error (more intuitive for non-technical users)
    - R²: How well model fits the data (1.0 = perfect fit)
    
    Args:
        model: Trained Keras model
        X_test: Test input sequences
        y_test: Test target values (scaled)
        scaler: Fitted MinMaxScaler for inverse transformation
    
    Returns:
        tuple: (predictions_original_scale, actuals_original_scale, metrics_dict)
    """
    print("\n📊 Evaluating predictions...")
    
    # === STEP 1: Make Predictions ===
    # model.predict outputs scaled predictions (0-1 range)
    predictions_scaled = model.predict(X_test, verbose=0)
    
    # Reshape for inverse_transform (needs 2D array)
    predictions_scaled = predictions_scaled.reshape(-1, 1)
    y_test_reshaped = y_test.reshape(-1, 1)
    
    # === STEP 2: Inverse Transform to Original Scale ===
    # Convert scaled values back to actual dollar/rupee amounts
    # This is necessary because metrics on scaled data are meaningless
    predictions = scaler.inverse_transform(predictions_scaled).flatten()
    actuals = scaler.inverse_transform(y_test_reshaped).flatten()
    
    # === STEP 3: Calculate Evaluation Metrics ===
    
    # RMSE (Root Mean Squared Error)
    # Average magnitude of errors, penalizes large errors heavily
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    
    # MAPE (Mean Absolute Percentage Error)
    # Average percentage error - easy to interpret
    # Example: MAPE of 5% means predictions are off by 5% on average
    mape = mean_absolute_percentage_error(actuals, predictions) * 100
    
    # R² Score (Coefficient of Determination)
    # Measures how well predictions fit the actual data
    # 1.0 = perfect fit, 0 = no better than predicting the mean, negative = worse than mean
    r2 = r2_score(actuals, predictions)
    
    # === STEP 4: Print Results ===
    print("\n" + "=" * 50)
    print("           EVALUATION METRICS")
    print("=" * 50)
    print(f"📏 RMSE  (Root Mean Squared Error): ${rmse:.2f}")
    print(f"   → Average prediction error magnitude")
    print(f"📈 MAPE  (Mean Absolute Percentage): {mape:.2f}%")
    print(f"   → Average percentage deviation from actual")
    print(f"🎯 R²    (R-Squared Score): {r2:.4f}")
    print(f"   → Model explains {r2*100:.2f}% of price variance")
    print("=" * 50)
    
    # Create metrics dictionary for return
    metrics = {
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }
    
    return predictions, actuals, metrics


def run_full_pipeline(ticker: str = "GOOGL", look_back: int = 60, checklist=None):
    """
    Execute the complete modeling pipeline: data → model → training → evaluation.
    
    This is the main entry point that coordinates all steps. It can accept
    an existing checklist logger or create a new one.
    
    Args:
        ticker (str): Stock symbol to predict (default: "GOOGL")
        look_back (int): Days of history to use as input (default: 60)
        checklist (ChecklistLogger): Existing logger or None to create new
    
    Returns:
        dict: Contains model, predictions, actuals, history, metrics, scaler, checklist
    """
    print("\n" + "=" * 60)
    print("   LSTM MODEL PIPELINE - Phase 2: Training & Evaluation")
    print("=" * 60 + "\n")
    
    # Create new checklist if none provided
    if checklist is None:
        checklist = ChecklistLogger()
    
    try:
        # === STEP 1: Load and Prepare Data ===
        X_train, y_train, X_test, y_test, scaler, raw_df, checklist = load_and_prepare_data(
            ticker=ticker, 
            look_back=look_back
        )
        
        # === STEP 2: Build Model Architecture ===
        model = build_lstm_model(look_back=look_back)
        checklist.mark_done("LSTM Model Architecture Defined")
        
        # === STEP 3: Configure Hyperparameters ===
        # These settings are defined in the architecture and training config
        checklist.mark_done("Hyperparameters Configured (lr=0.001, epochs=50, batch=32)")
        
        # === STEP 4: Train Model ===
        history = train_model(
            model, X_train, y_train, X_test, y_test,
            epochs=50, batch_size=32
        )
        checklist.mark_done("Model Trained with Early Stopping")
        
        # === STEP 5: Evaluate Model ===
        predictions, actuals, metrics = evaluate_model(model, X_test, y_test, scaler)
        checklist.mark_done("Evaluation Metrics Computed (RMSE, MAPE, R²)")
        
        print("\n✅ Done! Pipeline completed successfully.")
        
        # Return everything needed for visualization and further analysis
        return {
            'model': model,
            'predictions': predictions,
            'actuals': actuals,
            'history': history,
            'metrics': metrics,
            'scaler': scaler,
            'raw_df': raw_df,
            'checklist': checklist,
            'X_test': X_test,
            'y_test': y_test
        }
        
    except Exception as e:
        # Mark remaining checklist items as failed if error occurs
        checklist.mark_fail("LSTM Model Architecture Defined")
        checklist.mark_fail("Hyperparameters Configured (lr=0.001, epochs=50, batch=32)")
        checklist.mark_fail("Model Trained with Early Stopping")
        checklist.mark_fail("Evaluation Metrics Computed (RMSE, MAPE, R²)")
        
        print(f"\n❌ Pipeline failed: {e}")
        raise


# =============================================================================
# SELF-TEST: Run this module directly to test model building
# =============================================================================
if __name__ == "__main__":
    """
    When this file is run directly, execute the full pipeline.
    WARNING: This will download data and train the model (may take several minutes).
    """
    print("🧪 Testing model.py module (this will take a few minutes)...\n")
    
    try:
        # Run the full pipeline with default settings
        results = run_full_pipeline(ticker="GOOGL", look_back=60)
        
        # Display final checklist
        results['checklist'].print_summary()
        
        print("\n✅ Model test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Model test failed: {e}")
