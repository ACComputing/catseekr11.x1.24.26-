"""
Pure Python 3.14 implementation of DeepSeek-R1 + BitNet 1.58b Hybrid.
NO DEPENDENCIES (No PyTorch, No NumPy). Pure architectural simulation.
Includes a thread-safe ChatGPT-style GUI and Codebase Cat R1 integration!
"""

import math
import random
import sys
import time
import threading
import tkinter as tk
from tkinter import font

# =============================================================================
# 0. Pure Python Matrix & Vector Math Engine
# =============================================================================

def zeros(shape):
    """Creates a nested list of zeros with the given shape (1D or 2D)."""
    if len(shape) == 1:
        return [0.0] * shape[0]
    return [[0.0] * shape[1] for _ in range(shape[0])]

def rand_matrix(rows, cols, std=0.02):
    """Creates a 2D matrix with random normal-ish initialization."""
    return [[random.gauss(0, std) for _ in range(cols)] for _ in range(rows)]

def vec_add(v1, v2):
    return [a + b for a, b in zip(v1, v2)]

def vec_mul_scalar(v, scalar):
    return [a * scalar for a in v]

def dot_product(v1, v2):
    return sum(a * b for a, b in zip(v1, v2))

def mat_vec_mul(mat, vec):
    """Multiplies a 2D matrix by a 1D vector: y = mat @ vec"""
    return [dot_product(row, vec) for row in mat]

def silu(v):
    """SiLU (Swish) activation function: x * sigmoid(x)"""
    res = []
    for x in v:
        try:
            # Fixed Bug: Added safeguard to prevent OverflowError on large negative numbers
            res.append(x / (1.0 + math.exp(-x)))
        except OverflowError:
            res.append(0.0)
    return res

def softmax(v):
    max_val = max(v) if v else 0
    exps = [math.exp(x - max_val) for x in v]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

def rms_norm(v, weight, eps=1e-6):
    """Root Mean Square Normalization."""
    mean_sq = sum(x * x for x in v) / len(v)
    inv_std = 1.0 / math.sqrt(mean_sq + eps)
    return [(x * inv_std) * w for x, w in zip(v, weight)]

# =============================================================================
# 1. BitNet b1.58 Quantization (Ternary Weights, 8-bit Activations)
# =============================================================================

def quantize_activation(v):
    """Quantize 1D activations to 8-bit."""
    max_abs = max(abs(x) for x in v)
    max_abs = max(max_abs, 1e-5)
    scale = 127.0 / max_abs
    quantized = [max(-128, min(127, round(x * scale))) for x in v]
    return quantized, scale

def quantize_weight(mat):
    """Quantize 2D weights to Ternary {-1, 0, 1}."""
    quantized_mat = []
    gamma_sum = 0
    total_elements = len(mat) * len(mat[0])
    
    for row in mat:
        gamma_sum += sum(abs(x) for x in row)
    
    gamma = gamma_sum / total_elements
    gamma = max(gamma, 1e-5)
    scale = 1.0 / gamma
    
    for row in mat:
        q_row = [max(-1, min(1, round(x * scale))) for x in row]
        quantized_mat.append(q_row)
        
    return quantized_mat, scale

class BitLinear:
    """BitNet 1.58b Linear Layer using Pure Python."""
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = rand_matrix(out_features, in_features)
        
    def forward(self, x):
        x_q, x_scale = quantize_activation(x)
        w_q, w_scale = quantize_weight(self.weight)
        out_q = mat_vec_mul(w_q, x_q)
        dequant_scale = 1.0 / (x_scale * w_scale)
        return vec_mul_scalar(out_q, dequant_scale)

# =============================================================================
# 2. DeepSeek-R1: Multi-Head Latent Attention (MLA)
# =============================================================================

class DeepSeekMLA:
    def __init__(self, dim, num_heads, latent_dim_kv, latent_dim_q):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.w_down_kv = BitLinear(dim, latent_dim_kv)
        self.w_up_k = BitLinear(latent_dim_kv, dim)
        self.w_up_v = BitLinear(latent_dim_kv, dim)
        
        self.w_down_q = BitLinear(dim, latent_dim_q)
        self.w_up_q = BitLinear(latent_dim_q, dim)
        self.w_out = BitLinear(dim, dim)
        
    def apply_rope(self, vec, pos):
        out = [0.0] * len(vec)
        for i in range(0, len(vec), 2):
            freq = 1.0 / (10000 ** (i / len(vec)))
            theta = pos * freq
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            out[i] = vec[i] * cos_t - vec[i+1] * sin_t
            out[i+1] = vec[i+1] * cos_t + vec[i] * sin_t
        return out

    def forward(self, x, pos=0):
        c_kv = self.w_down_kv.forward(x)
        k = self.w_up_k.forward(c_kv)
        v = self.w_up_v.forward(c_kv)
        
        c_q = self.w_down_q.forward(x)
        q = self.w_up_q.forward(c_q)
        
        q = self.apply_rope(q, pos)
        k = self.apply_rope(k, pos)
        
        attn_score = dot_product(q, k) / math.sqrt(self.head_dim)
        attn_weight = softmax([attn_score])[0] 
        
        attn_out = vec_mul_scalar(v, attn_weight)
        return self.w_out.forward(attn_out)

# =============================================================================
# 3. DeepSeek-R1: Mixture of Experts (DeepSeekMoE)
# =============================================================================

class Expert:
    def __init__(self, dim, hidden_dim):
        self.up = BitLinear(dim, hidden_dim)
        self.gate = BitLinear(dim, hidden_dim)
        self.down = BitLinear(hidden_dim, dim)
        
    def forward(self, x):
        up_proj = self.up.forward(x)
        gate_proj = silu(self.gate.forward(x))
        swiglu = [u * g for u, g in zip(up_proj, gate_proj)]
        return self.down.forward(swiglu)

class DeepSeekMoE:
    def __init__(self, dim, num_shared, num_routed, top_k):
        expert_hidden = dim * 2
        self.shared_experts = [Expert(dim, expert_hidden) for _ in range(num_shared)]
        self.routed_experts = [Expert(dim, expert_hidden) for _ in range(num_routed)]
        self.router = BitLinear(dim, num_routed)
        self.top_k = top_k
        
    def forward(self, x):
        shared_out = zeros([len(x)])
        for expert in self.shared_experts:
            e_out = expert.forward(x)
            shared_out = vec_add(shared_out, e_out)
            
        router_logits = self.router.forward(x)
        routing_probs = softmax(router_logits)
        
        indexed_probs = list(enumerate(routing_probs))
        indexed_probs.sort(key=lambda item: item[1], reverse=True)
        top_k_experts = indexed_probs[:self.top_k]
        
        routed_out = zeros([len(x)])
        for expert_idx, prob in top_k_experts:
            e_out = self.routed_experts[expert_idx].forward(x)
            e_out_scaled = vec_mul_scalar(e_out, prob)
            routed_out = vec_add(routed_out, e_out_scaled)
            
        return vec_add(shared_out, routed_out)

# =============================================================================
# 4. Full DeepSeek-R1 + BitNet Hybrid Model Structure
# =============================================================================

class HybridTransformerBlock:
    def __init__(self, dim):
        self.norm_weight_1 = [1.0] * dim
        self.norm_weight_2 = [1.0] * dim
        self.mla = DeepSeekMLA(dim, num_heads=4, latent_dim_kv=dim//4, latent_dim_q=dim//4)
        self.moe = DeepSeekMoE(dim, num_shared=1, num_routed=4, top_k=2)
        
    def forward(self, x, pos):
        x_norm = rms_norm(x, self.norm_weight_1)
        attn_out = self.mla.forward(x_norm, pos)
        x = vec_add(x, attn_out) 
        
        x_norm = rms_norm(x, self.norm_weight_2)
        moe_out = self.moe.forward(x_norm)
        x = vec_add(x, moe_out) 
        
        return x

class DeepSeekBitNetModel:
    def __init__(self, vocab_size=32000, dim=64, num_layers=2):
        self.dim = dim
        self.vocab_size = vocab_size
        self.embed_matrix = rand_matrix(vocab_size, dim) 
        self.layers = [HybridTransformerBlock(dim) for _ in range(num_layers)]
        self.lm_head = BitLinear(dim, vocab_size)
        
    def forward_token(self, token_id, pos):
        x = self.embed_matrix[token_id]
        for layer in self.layers:
            x = layer.forward(x, pos)
        logits = self.lm_head.forward(x)
        return logits

# =============================================================================
# 5. Codebase Cat R1 & ChatGPT-style GUI
# =============================================================================

class CodebaseCatR1:
    @staticmethod
    def get_ascii():
        return """
     /\\_/\\     [ CatSEEK R1 > PR - ACTIVATED ]
    ( o.o )  > "Meow. 1.58b bits loaded. Ready to seek."
     > ^ <
        """
    @staticmethod
    def purr():
        responses = [
            "Routing through Mixture of Experts. Found the best catnip!",
            "Applying Multi-Head Latent Attention. Very latent, much purr.",
            "Quantizing to Ternary {-1, 0, 1}. Saving memory for naps.",
            "RMS Normalizing... Purrfectly balanced, 9 lives strong."
        ]
        return random.choice(responses)

class RedirectText:
    """Fixed Bug: Thread-safe stdout redirection to Tkinter Text widget."""
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        # Using .after() schedules the insert operation on the main Tkinter thread
        self.text_widget.after(0, self._insert_text, string)

    def _insert_text(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
        
    def flush(self):
        pass

class ChatGPTGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ChatGPT - CatSEEK R1 1.58b")
        self.root.geometry("900x600")
        
        # ChatGPT UI Color Palette
        self.bg_sidebar = "#171717"
        self.bg_main = "#212121"
        self.bg_input = "#2f2f2f"
        self.fg_text = "#ececec"
        self.fg_muted = "#9b9b9b"
        
        self.root.configure(bg=self.bg_main)
        
        # --- Sidebar ---
        self.sidebar = tk.Frame(self.root, bg=self.bg_sidebar, width=260)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)
        
        # Top sidebar button (New Chat)
        self.btn_new = tk.Button(self.sidebar, text="+  New Chat", bg=self.bg_sidebar, fg=self.fg_text,
                                 activebackground="#2a2b32", activeforeground=self.fg_text,
                                 font=("Helvetica", 11), relief=tk.FLAT, anchor="w", padx=15, pady=10,
                                 command=self.clear_console)
        self.btn_new.pack(fill=tk.X, padx=10, pady=10)
        
        # Actions in sidebar
        btn_style = {
            "bg": self.bg_sidebar, "fg": self.fg_text, "activebackground": "#2a2b32", 
            "activeforeground": self.fg_text, "font": ("Helvetica", 10), "relief": tk.FLAT, 
            "anchor": "w", "padx": 15, "pady": 8
        }
        
        tk.Label(self.sidebar, text="Today", bg=self.bg_sidebar, fg=self.fg_muted, 
                 font=("Helvetica", 9, "bold"), anchor="w").pack(fill=tk.X, padx=25, pady=(15, 5))
        
        self.btn_run = tk.Button(self.sidebar, text="▶ Run 1.58b Simulation", command=self.start_simulation, **btn_style)
        self.btn_run.pack(fill=tk.X, padx=10, pady=2)
        
        self.btn_cat = tk.Button(self.sidebar, text="🐱 Summon Codebase Cat", command=self.summon_cat, **btn_style)
        self.btn_cat.pack(fill=tk.X, padx=10, pady=2)
        
        # --- Main Chat Area ---
        self.main_area = tk.Frame(self.root, bg=self.bg_main)
        self.main_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Header (Model Selection)
        self.header_frame = tk.Frame(self.main_area, bg=self.bg_main, height=50)
        self.header_frame.pack(side=tk.TOP, fill=tk.X)
        self.header_label = tk.Label(self.header_frame, text="CatSEEK R1 (1.58b)  ▼", bg=self.bg_main, fg=self.fg_text,
                                     font=("Helvetica", 12, "bold"))
        self.header_label.pack(side=tk.LEFT, padx=20, pady=15)
        
        # Text Output (Chat History)
        self.chat_frame = tk.Frame(self.main_area, bg=self.bg_main)
        self.chat_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=40, pady=10)
        
        self.console_font = font.Font(family="Helvetica", size=11)
        self.console = tk.Text(self.chat_frame, bg=self.bg_main, fg=self.fg_text, font=self.console_font, 
                               state=tk.NORMAL, wrap=tk.WORD, relief=tk.FLAT, insertbackground=self.fg_text)
        self.console.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bottom Input Area
        self.input_frame = tk.Frame(self.main_area, bg=self.bg_main)
        self.input_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=80, pady=30)
        
        self.input_box = tk.Entry(self.input_frame, bg=self.bg_input, fg=self.fg_text, font=("Helvetica", 11),
                                  relief=tk.FLAT, insertbackground=self.fg_text)
        self.input_box.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=12, padx=10)
        self.input_box.bind("<Return>", self.handle_input)
        
        self.btn_send = tk.Button(self.input_frame, text="↑", bg="#ececec", fg="#212121",
                                  font=("Helvetica", 12, "bold"), relief=tk.FLAT, command=self.handle_input)
        self.btn_send.pack(side=tk.RIGHT, padx=5, ipady=5, ipadx=10)
        
        # Redirect standard output to the text widget (Thread safe!)
        sys.stdout = RedirectText(self.console)
        self.print_welcome()

    def print_welcome(self):
        print("CatSEEK R1: Hello! I am running the pure Python BitNet-1.58b + DeepSeek-R1 architectural simulation.")
        print("Type 'run' to start the simulation, 'cat' to summon Codebase Cat, or 'clear' to reset this view.\n")

    def handle_input(self, event=None):
        user_text = self.input_box.get().strip()
        if not user_text: return
        
        self.input_box.delete(0, tk.END)
        print(f"\nYou: {user_text}\n")
        
        cmd = user_text.lower()
        if cmd in ["run", "simulate", "start"]:
            self.start_simulation()
        elif cmd in ["cat", "summon cat", "meow"]:
            self.summon_cat()
        elif cmd in ["clear", "new chat"]:
            self.clear_console()
        else:
            print("CatSEEK R1: I am currently observing the 1.58b network. Try typing 'run' or 'cat'.\n")

    def summon_cat(self):
        print("CatSEEK R1:" + CodebaseCatR1.get_ascii())
        print("Codebase Cat R1 is monitoring the 1.58b architecture...\n")

    def clear_console(self):
        self.console.delete(1.0, tk.END)
        self.print_welcome()

    def start_simulation(self):
        self.btn_run.config(state=tk.DISABLED)
        # Run in a background thread to prevent GUI freezing
        threading.Thread(target=self.run_simulation_logic, daemon=True).start()

    def run_simulation_logic(self):
        print("CatSEEK R1: Initializing Pure Python 1.58b Simulation...")
        time.sleep(0.5)
        
        model = DeepSeekBitNetModel(vocab_size=256, dim=32, num_layers=2)
        print("Model initialized: 2 layers, 32 dim, ternary weights {-1, 0, 1}.")
        time.sleep(0.5)
        
        input_tokens = [42, 87, 105]
        print(f"\nSimulating forward pass with input sequence: {input_tokens}\n")
        
        for pos, token in enumerate(input_tokens):
            print(f"  [{pos}] Processing Token {token}...")
            time.sleep(0.6)
            
            # Print feline wisdom
            print(f"  (Cat R1) -> {CodebaseCatR1.purr()}")
            
            logits = model.forward_token(token, pos)
            predicted_token = logits.index(max(logits))
            print(f"  [Output] Next Token Prediction: {predicted_token}\n")
            time.sleep(0.6)
            
        print("Execution Complete. Zero dependencies. Pure Tkinter ChatGPT Theme.\n")
        
        # Re-enable button from main thread (Thread safe!)
        self.root.after(0, lambda: self.btn_run.config(state=tk.NORMAL))

# =============================================================================
# Run the Application
# =============================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatGPTGUI(root)
    root.mainloop()
