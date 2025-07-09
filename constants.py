# ============================================================================
# SPIKE-DRIVEN TRANSFORMER CONSTANTS
# ============================================================================

# === ARCHITECTURE CONSTANTS ===
"""
MAX_SEQ_LEN = 256
D_MODEL = 256
N_HEADS = 4
N_LAYERS = 4
D_FF = 512
TIMESTEPS = 4
"""
MAX_SEQ_LEN = 64  
D_MODEL = 768      # 3x (was 256)
N_HEADS = 12       # 3x (was 4)
N_LAYERS = 12      # 2x (was 6)
D_FF = 3072        # 3x (was 1024)
TIMESTEPS = 6

# === INITIALIZATION CONSTANTS ===
# Embedding initialization
EMBEDDING_INIT_SCALE = 0.2      
EMBEDDING_INIT_BIAS = 0.0      

FORWARD_MEMBRANE_CLIP_THRE = 5.0

QKV_INIT_SCALE = 0.3          
OUTPUT_INIT_SCALE = 0.3        
FF_INIT_SCALE = 0.3          

# Xavier init multiplier (use 2.0 for ReLU-like activations)
XAVIER_MULTIPLIER = 2.0
XAVIER_MUTLITPLIER_2 = 1.0

# === SPIKE THRESHOLDS ===
SPIKE_THRESH_EMBEDDING = 0.5
SPIKE_THRESH_Q = 0.5
SPIKE_THRESH_K = 0.5  
SPIKE_THRESH_V = 0.6
SPIKE_THRESH_ATTN = 0.5
SPIKE_THRESH_FF1 = 0.5
SPIKE_THRESH_FF2 = 0.6

# === SURROGATE GRADIENT PARAMETERS ===
SURROGATE_BETA = 2.0          
SURROGATE_CLIP_VALUE = 10.0

# === OPTIMIZATION CONSTANTS ===
LEARNING_RATE = 0.00001 

# Adam optimizer parameters
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8
GRADIENT_CLIP_VALUE = 5.0

# === TRAINING CONSTANTS ===
EPOCHS = 1000
SHUFFLE_DATA = True
LOG_INTERVAL = 10

DECAY_FACTOR = 0.9
RESET_VALUE = 0.0

SAVE_FILEPATH = "MODEL"
TRAIN_MODEL = True