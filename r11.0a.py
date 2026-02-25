"""
Pure Python 3.14 DeepSeek-R1 + BitNet 1b/2b Hybrid – FULLY FIXED GUI (CatseekR1V0)
NO DEPENDENCIES • THREAD-SAFE • CODEBASE CAT R1
"""

import math
import random
import sys
import time
import threading
import tkinter as tk
from tkinter import font

# =============================================================================
# 0. Pure Python Math Engine
# =============================================================================
def zeros(shape):
    if len(shape) == 1:
        return [0.0] * shape[0]
    return [[0.0] * shape[1] for _ in range(shape[0])]

def rand_matrix(rows, cols, std=0.02):
    return [[random.gauss(0, std) for _ in range(cols)] for _ in range(rows)]

def vec_add(v1, v2):
    return [a + b for a, b in zip(v1, v2)]

def vec_mul_scalar(v, scalar):
    return [a * scalar for a in v]

def dot_product(v1, v2):
    return sum(a * b for a, b in zip(v1, v2))

def mat_vec_mul(mat, vec):
    return [dot_product(row, vec) for row in mat]

def silu(v):
    res = []
    for x in v:
        try:
            res.append(x / (1.0 + math.exp(-x)))
        except OverflowError:
            res.append(0.0)
    return res

def softmax(v):
    if not v:
        return []
    max_val = max(v)
    exps = [math.exp(x - max_val) for x in v]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

def rms_norm(v, weight, eps=1e-6):
    mean_sq = sum(x * x for x in v) / len(v)
    inv_std = 1.0 / math.sqrt(mean_sq + eps)
    return [(x * inv_std) * w for x, w in zip(v, weight)]

# =============================================================================
# 1. BitNet 1b/2b Quantization
# =============================================================================
def quantize_activation(v):
    max_abs = max(abs(x) for x in v) or 1e-5
    scale = 127.0 / max_abs
    quantized = [max(-128, min(127, round(x * scale))) for x in v]
    return quantized, scale

def quantize_weight(mat, quant_mode='1bit'):
    total = len(mat) * len(mat[0]) or 1
    gamma = sum(abs(x) for row in mat for x in row) / total
    gamma = max(gamma, 1e-5)
    if quant_mode == '1bit':
        q_mat = [[1.0 if x >= 0 else -1.0 for x in row] for row in mat]
        return q_mat, gamma
    elif quant_mode == '2bit':
        levels = [-1.5, -0.5, 0.5, 1.5]
        q_mat = []
        for row in mat:
            q_row = [levels[min(range(4), key=lambda i: abs((x / gamma) - levels[i]))] for x in row]
            q_mat.append(q_row)
        return q_mat, gamma
    raise ValueError("quant_mode must be '1bit' or '2bit'")

class BitLinear:
    def __init__(self, in_features, out_features, quant_mode='1bit'):
        self.in_features = in_features
        self.out_features = out_features
        self.quant_mode = quant_mode
        self.weight = rand_matrix(out_features, in_features)

    def forward(self, x):
        x_q, x_scale = quantize_activation(x)
        w_q, w_scale = quantize_weight(self.weight, self.quant_mode)
        out_q = mat_vec_mul(w_q, x_q)
        dequant_scale = 1.0 / (x_scale * w_scale)
        return vec_mul_scalar(out_q, dequant_scale)

# =============================================================================
# 2-4. DeepSeek MLA, Expert, CatR1XMoE, Hybrid Block, Full Model
# =============================================================================
class DeepSeekMLA:
    def __init__(self, dim, num_heads, latent_dim_kv, latent_dim_q, quant_mode='1bit'):
        self.dim = dim
        self.head_dim = dim // num_heads
        self.quant_mode = quant_mode
        self.w_down_kv = BitLinear(dim, latent_dim_kv, quant_mode)
        self.w_up_k = BitLinear(latent_dim_kv, dim, quant_mode)
        self.w_up_v = BitLinear(latent_dim_kv, dim, quant_mode)
        self.w_down_q = BitLinear(dim, latent_dim_q, quant_mode)
        self.w_up_q = BitLinear(latent_dim_q, dim, quant_mode)
        self.w_out = BitLinear(dim, dim, quant_mode)

    def apply_rope(self, vec, pos):
        out = [0.0] * len(vec)
        for i in range(0, len(vec) - 1, 2):
            freq = 1.0 / (10000 ** (i / len(vec)))
            theta = pos * freq
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            out[i] = vec[i] * cos_t - vec[i + 1] * sin_t
            out[i + 1] = vec[i + 1] * cos_t + vec[i] * sin_t
        if len(vec) % 2 == 1:
            out[-1] = vec[-1]
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

class Expert:
    def __init__(self, dim, hidden_dim, quant_mode='1bit'):
        self.up = BitLinear(dim, hidden_dim, quant_mode)
        self.gate = BitLinear(dim, hidden_dim, quant_mode)
        self.down = BitLinear(hidden_dim, dim, quant_mode)

    def forward(self, x):
        up_proj = self.up.forward(x)
        gate_proj = self.gate.forward(x)
        hidden = [u * g for u, g in zip(up_proj, silu(gate_proj))]
        return self.down.forward(hidden)

class CatR1XMoE:
    def __init__(self, dim, num_shared, num_routed, top_k, quant_mode='1bit'):
        expert_hidden = dim * 2
        self.shared_experts = [Expert(dim, expert_hidden, quant_mode) for _ in range(num_shared)]
        self.routed_experts = [Expert(dim, expert_hidden, quant_mode) for _ in range(num_routed)]
        self.router = BitLinear(dim, num_routed, quant_mode)
        self.top_k = top_k

    def forward(self, x):
        shared_out = [0.0] * len(x)
        for expert in self.shared_experts:
            shared_out = vec_add(shared_out, expert.forward(x))

        route_logits = self.router.forward(x)
        route_probs = softmax(route_logits)

        top_k_indices = sorted(range(len(route_probs)), key=lambda i: route_probs[i], reverse=True)[:self.top_k]
        top_k_probs = [route_probs[i] for i in top_k_indices]
        sum_probs = sum(top_k_probs) or 1e-9
        norm_probs = [p / sum_probs for p in top_k_probs]

        routed_out = [0.0] * len(x)
        for i, idx in enumerate(top_k_indices):
            e_out = self.routed_experts[idx].forward(x)
            weighted = vec_mul_scalar(e_out, norm_probs[i])
            routed_out = vec_add(routed_out, weighted)

        return vec_add(shared_out, routed_out)

class HybridTransformerBlock:
    def __init__(self, dim, quant_mode='1bit'):
        self.norm_weight_1 = [1.0] * dim
        self.norm_weight_2 = [1.0] * dim
        self.mla = DeepSeekMLA(dim, 4, dim//4, dim//4, quant_mode)
        self.moe = CatR1XMoE(dim, 1, 4, 2, quant_mode)

    def forward(self, x, pos):
        x_norm = rms_norm(x, self.norm_weight_1)
        attn_out = self.mla.forward(x_norm, pos)
        x = vec_add(x, attn_out)
        x_norm = rms_norm(x, self.norm_weight_2)
        moe_out = self.moe.forward(x_norm)
        return vec_add(x, moe_out)

class DeepSeekBitNetModel:
    def __init__(self, vocab_size=256, dim=32, num_layers=2, quant_mode='1bit'):
        self.dim = dim
        self.vocab_size = vocab_size
        self.quant_mode = quant_mode
        self.embed_matrix = rand_matrix(vocab_size, dim)
        self.layers = [HybridTransformerBlock(dim, quant_mode) for _ in range(num_layers)]
        self.norm_weight_f = [1.0] * dim
        self.lm_head = BitLinear(dim, vocab_size, quant_mode)

    def forward_token(self, token_id, pos):
        token_id = max(0, min(token_id, self.vocab_size - 1))
        x = self.embed_matrix[token_id]
        for layer in self.layers:
            x = layer.forward(x, pos)
        x = rms_norm(x, self.norm_weight_f)
        return self.lm_head.forward(x)

# =============================================================================
# 5. Codebase Cat R1
# =============================================================================
class CodebaseCatR1:
    @staticmethod
    def get_ascii():
        return """
     /\\_/\\     [ CatSEEK R1 > 1b/2b FULLY FIXED ]
    ( o.o )  > "Meow. All bugs patched. Ready to seek."
     > ^ <
        """
    @staticmethod
    def purr():
        responses = [
            "1-bit routing at lightspeed. Catnip acquired!",
            "2-bit attention locked in. Purrfect precision.",
            "Final RMSNorm applied. 9 lives stabilized.",
            "Top-k normalized. No expert left behind.",
            "All bugs fixed. Now there are zero. Meow."
        ]
        return random.choice(responses)

# =============================================================================
# 6. Thread-Safe Redirect & Full CatseekR1V0-Style GUI
# =============================================================================
class RedirectText:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.after(0, self._insert, string)

    def _insert(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)

    def flush(self):
        pass

class CatseekR1V0GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CatseekR1V0 – CatSEEK R1 1b/2b (Fully Fixed)")
        self.root.geometry("900x650")

        # Colors (dark theme)
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

        self.btn_new = tk.Button(self.sidebar, text="+  New Chat", bg=self.bg_sidebar, fg="#000000",
                                 activebackground="#2a2b32", font=("Helvetica", 11),
                                 relief=tk.FLAT, anchor="w", padx=15, pady=10,
                                 command=self.clear_console)
        self.btn_new.pack(fill=tk.X, padx=10, pady=10)

        btn_style = {"bg": self.bg_sidebar, "fg": "#000000", "activebackground": "#2a2b32",
                     "font": ("Helvetica", 10), "relief": tk.FLAT, "anchor": "w", "padx": 15, "pady": 8}

        tk.Label(self.sidebar, text="Today", bg=self.bg_sidebar, fg=self.fg_muted,
                 font=("Helvetica", 9, "bold"), anchor="w").pack(fill=tk.X, padx=25, pady=(15, 5))

        self.btn_run = tk.Button(self.sidebar, text="▶ Run 1b/2b Simulation",
                                 command=self.start_simulation, **btn_style)
        self.btn_run.pack(fill=tk.X, padx=10, pady=2)

        self.btn_cat = tk.Button(self.sidebar, text="🐱 Summon Codebase Cat",
                                 command=self.summon_cat, **btn_style)
        self.btn_cat.pack(fill=tk.X, padx=10, pady=2)

        # --- Main Area ---
        self.main_area = tk.Frame(self.root, bg=self.bg_main)
        self.main_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.header_frame = tk.Frame(self.main_area, bg=self.bg_main, height=50)
        self.header_frame.pack(side=tk.TOP, fill=tk.X)
        self.header_label = tk.Label(self.header_frame, text="CatseekR1V0 (1b/2b) ▼",
                                     bg=self.bg_main, fg=self.fg_text, font=("Helvetica", 12, "bold"))
        self.header_label.pack(side=tk.LEFT, padx=20, pady=15)

        self.chat_frame = tk.Frame(self.main_area, bg=self.bg_main)
        self.chat_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=40, pady=10)

        self.console_font = font.Font(family="Helvetica", size=11)
        self.console = tk.Text(self.chat_frame, bg=self.bg_main, fg=self.fg_text,
                               font=self.console_font, state=tk.NORMAL, wrap=tk.WORD,
                               relief=tk.FLAT, insertbackground=self.fg_text)
        self.console.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Input area
        self.input_frame = tk.Frame(self.main_area, bg=self.bg_main)
        self.input_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=80, pady=30)

        self.input_box = tk.Entry(self.input_frame, bg=self.bg_input, fg=self.fg_text,
                                  font=("Helvetica", 11), relief=tk.FLAT, insertbackground=self.fg_text)
        self.input_box.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=12, padx=10)
        self.input_box.bind("<Return>", self.handle_input)

        self.btn_send = tk.Button(self.input_frame, text="↑", bg="#ececec", fg="#000000",
                                  font=("Helvetica", 12, "bold"), relief=tk.FLAT,
                                  command=self.handle_input)
        self.btn_send.pack(side=tk.RIGHT, padx=5, ipady=5, ipadx=10)

        # Redirect stdout
        sys.stdout = RedirectText(self.console)
        self.print_welcome()

    def print_welcome(self):
        print("CatseekR1V0 1b/2b: Fully fixed GUI ready. Pure Python 3.14, zero dependencies.")
        print("Type 'run' to simulate forward pass, 'cat' for feline wisdom, or 'clear' to reset.\n")

    def handle_input(self, event=None):
        user_text = self.input_box.get().strip()
        if not user_text:
            return
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
            print("CatseekR1V0: Try 'run', 'cat', or 'clear'.\n")

    def summon_cat(self):
        print("CatseekR1V0:" + CodebaseCatR1.get_ascii())
        print("Codebase Cat R1:", CodebaseCatR1.purr(), "\n")

    def clear_console(self):
        self.console.delete(1.0, tk.END)
        self.print_welcome()

    def start_simulation(self):
        self.btn_run.config(state=tk.DISABLED)
        threading.Thread(target=self.run_simulation_logic, daemon=True).start()

    def run_simulation_logic(self):
        print("CatseekR1V0: Initializing Pure Python 1b/2b Simulation...")
        time.sleep(0.5)
        model = DeepSeekBitNetModel(vocab_size=256, dim=32, num_layers=2, quant_mode='1bit')  # change to '2bit' if desired
        print("Model loaded: 2 layers, dim=32, binary/2-bit weights.")
        time.sleep(0.5)

        input_tokens = [42, 87, 105]
        print(f"\nSimulating forward pass with tokens: {input_tokens}\n")

        for pos, token in enumerate(input_tokens):
            print(f"  [{pos}] Processing Token {token}...")
            time.sleep(0.6)
            print(f"    (Cat R1) -> {CodebaseCatR1.purr()}")
            logits = model.forward_token(token, pos)
            predicted = logits.index(max(logits))
            print(f"    [Output] Next Token Prediction: {predicted}\n")
            time.sleep(0.6)

        print("Execution Complete. Zero dependencies. Pure Python 3.14.\n")
        self.root.after(0, lambda: self.btn_run.config(state=tk.NORMAL))

if __name__ == "__main__":
    random.seed(42)  # reproducible demo
    root = tk.Tk()
    app = CatseekR1V0GUI(root)
    root.mainloop()
