# ============================================================================
# SPIKE-DRIVEN TRANSFORMER CONSTANTS
# ============================================================================

# === ARCHITECTURE CONSTANTS ===
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
D_FF = 256
TIMESTEPS = 4

# === INITIALIZATION CONSTANTS ===
# Embedding initialization
EMBEDDING_INIT_SCALE = 0.2      
EMBEDDING_INIT_BIAS = 0.0      

FORWARD_MEMBRANE_CLIP_THRE = 5.0

QKV_INIT_SCALE = 0.5          
OUTPUT_INIT_SCALE = 0.3        
FF_INIT_SCALE = 0.5          

# Xavier init multiplier (use 2.0 for ReLU-like activations)
XAVIER_MULTIPLIER = 2.0
XAVIER_MUTLITPLIER_2 = 1.0

# === SPIKE THRESHOLDS ===
SPIKE_THRESH_EMBEDDING = 1.0    
SPIKE_THRESH_Q = 0.3         
SPIKE_THRESH_K = 0.3          
SPIKE_THRESH_V = 0.4        
SPIKE_THRESH_ATTN = 0.3        
SPIKE_THRESH_FF1 = 0.3     
SPIKE_THRESH_FF2 = 0.3

# === SURROGATE GRADIENT PARAMETERS ===
SURROGATE_BETA = 2.0          
SURROGATE_CLIP_VALUE = 10.0

# === OPTIMIZATION CONSTANTS ===
LEARNING_RATE = 0.001  

# Adam optimizer parameters
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8
GRADIENT_CLIP_VALUE = 1.0

# === TRAINING CONSTANTS ===
EPOCHS = 100
SHUFFLE_DATA = True
LOG_INTERVAL = 10

DECAY_FACTOR = 0.9
RESET_VALUE = 0.0