"""
Pure Python 3.14 CATSEEKR0.0B Realistic 4B LLM + o1 Reasoning Chip
NO FILES • NO DEPENDENCIES • THREAD-SAFE GUI • CODEBASE CAT R1
"""

import math
import random
import sys
import time
import threading
import tkinter as tk
from tkinter import font
import builtins

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
# 1. Microsoft BitNet Quantization (1.58bit / 2bit / 4bit)
# =============================================================================
def quantize_activation(v, bit_width=8):
    """
    Symmetric quantization of activations.
    For bit_width = 4: range [-8, 7]; scale = max_abs / 7.
    For bit_width = 8: range [-128, 127]; scale = max_abs / 127.
    """
    max_abs = max(abs(x) for x in v) or 1e-5
    qmax = (1 << (bit_width - 1)) - 1          # e.g., 7 for 4-bit, 127 for 8-bit
    scale = qmax / max_abs
    quantized = [max(-qmax-1, min(qmax, round(x * scale))) for x in v]
    return quantized, scale

def quantize_weight(mat, quant_mode):
    """
    Quantize weight matrix according to the specified mode:
      - '1.58bit': ternary {-γ, 0, +γ}
      - '2bit'   : 4-level (values -1.5γ, -0.5γ, 0.5γ, 1.5γ)
      - '4bit'   : symmetric 4-bit integer (values -8γ/7 … +8γ/7)
    Returns quantized matrix and the shared scale γ.
    """
    total = len(mat) * len(mat[0]) or 1
    gamma = sum(abs(x) for row in mat for x in row) / total
    gamma = max(gamma, 1e-5)

    if quant_mode == '1.58bit':
        q_mat = []
        for row in mat:
            q_row = []
            for x in row:
                if x > 0.5 * gamma:
                    q_row.append(gamma)
                elif x < -0.5 * gamma:
                    q_row.append(-gamma)
                else:
                    q_row.append(0.0)
            q_mat.append(q_row)
        return q_mat, gamma

    elif quant_mode == '2bit':
        levels = [-1.5, -0.5, 0.5, 1.5]
        q_mat = []
        for row in mat:
            q_row = [levels[min(range(4), key=lambda i: abs((x / gamma) - levels[i]))] for x in row]
            q_mat.append(q_row)
        return q_mat, gamma

    elif quant_mode == '4bit':
        # 4-bit symmetric: map each weight to an integer in [-8, 7] / 7 * gamma
        qmax = 7
        q_mat = []
        for row in mat:
            q_row = []
            for x in row:
                # quantize to integer in [-8,7]
                q_val = max(-qmax-1, min(qmax, round(x / gamma * qmax)))
                q_row.append(q_val * gamma / qmax)   # dequantized value
            q_mat.append(q_row)
        return q_mat, gamma

    else:
        raise ValueError("quant_mode must be '1.58bit', '2bit', or '4bit'")

class BitLinear:
    """BitLinear layer with selectable quantization mode."""
    def __init__(self, in_features, out_features, quant_mode='1.58bit'):
        self.in_features = in_features
        self.out_features = out_features
        self.quant_mode = quant_mode
        self.weight = rand_matrix(out_features, in_features)

    def forward(self, x):
        # Activation bit width: 8-bit for ternary/2bit modes, 4-bit for 4bit mode
        act_bit_width = 4 if self.quant_mode == '4bit' else 8
        x_q, x_scale = quantize_activation(x, act_bit_width)
        w_q, w_scale = quantize_weight(self.weight, self.quant_mode)
        out_q = mat_vec_mul(w_q, x_q)
        dequant_scale = 1.0 / (x_scale * w_scale)
        return vec_mul_scalar(out_q, dequant_scale)

# =============================================================================
# 2-4. Realistic 4B-Scale Model (Microsoft BitNet)
# =============================================================================
class DeepSeekMLA:
    def __init__(self, dim, num_heads, latent_dim_kv, latent_dim_q, quant_mode='1.58bit'):
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
    def __init__(self, dim, hidden_dim, quant_mode='1.58bit'):
        self.up = BitLinear(dim, hidden_dim, quant_mode)
        self.gate = BitLinear(dim, hidden_dim, quant_mode)
        self.down = BitLinear(hidden_dim, dim, quant_mode)

    def forward(self, x):
        up_proj = self.up.forward(x)
        gate_proj = self.gate.forward(x)
        hidden = [u * g for u, g in zip(up_proj, silu(gate_proj))]
        return self.down.forward(hidden)

class CatR1XMoE:
    def __init__(self, dim, num_shared, num_routed, top_k, quant_mode='1.58bit'):
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
    def __init__(self, dim, quant_mode='1.58bit'):
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
    def __init__(self, vocab_size=50272, dim=64, num_layers=8, quant_mode='1.58bit'):
        # Realistic 4B‑scale simulation: vocab=50k, 8 layers, dim=64 (for speed)
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
     /\\_/\\     [ CATSEEKR0.0B REALISTIC 4B o1 CHIP ACTIVATED ]
    ( o.o )  > "Meow. 4.1 billion parameters loaded. o1 thinking engaged."
     > ^ <
        """
    @staticmethod
    def purr():
        responses = [
            "o1 Chip allocating realistic 4B test-time compute. Catnip acquired!",
            "4B-scale MoE routing at lightspeed. Purrfect precision.",
            "DeepSeek-R1 + o1 verification loop complete. 9 lives stabilized.",
            "Top-k normalized with hidden CoT. No expert left behind.",
            "Realistic 4B simulation green. Zero bugs. Meow."
        ]
        return random.choice(responses)

# =============================================================================
# 6. Thread-Safe Redirect & CATSEEKR0.0B GUI (enhanced with code interpreter)
# =============================================================================
class RedirectText:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.after(0, self._insert, string)

    def _insert(self, string):
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
        self.text_widget.config(state=tk.DISABLED)

    def flush(self):
        pass

class CATSEEKR0_0BGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CATSEEKR0.0B – Code Interpreter & Sandbox (BitNet 1.58/2/4-bit + o1)")
        self.root.geometry("900x720")

        self.bg_sidebar = "#171717"
        self.bg_main = "#212121"
        self.bg_input = "#2f2f2f"
        self.fg_text = "#ececec"
        self.fg_muted = "#9b9b9b"

        self.root.configure(bg=self.bg_main)

        # Language mode: 'auto' (detect) or forced 'en'/'zh'
        self.language_mode = tk.StringVar(value="auto")
        # Quantization mode selector
        self.quant_mode = tk.StringVar(value="1.58bit")

        # Sidebar
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

        self.btn_run = tk.Button(self.sidebar, text="▶ Run Demo Generation",
                                 command=lambda: self.start_simulation("Tell me about BitNet."), **btn_style)
        self.btn_run.pack(fill=tk.X, padx=10, pady=2)

        self.btn_cat = tk.Button(self.sidebar, text="🐱 Summon Codebase Cat",
                                 command=self.summon_cat, **btn_style)
        self.btn_cat.pack(fill=tk.X, padx=10, pady=2)

        # Quantization mode selection
        q_frame = tk.Frame(self.sidebar, bg=self.bg_sidebar)
        q_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(q_frame, text="Quantization:", bg=self.bg_sidebar, fg=self.fg_muted,
                 font=("Helvetica", 9)).pack(anchor="w")
        tk.Radiobutton(q_frame, text="1.58-bit (ternary)", variable=self.quant_mode, value="1.58bit",
                       bg=self.bg_sidebar, fg=self.fg_text, selectcolor=self.bg_sidebar,
                       activebackground=self.bg_sidebar).pack(anchor="w")
        tk.Radiobutton(q_frame, text="2-bit", variable=self.quant_mode, value="2bit",
                       bg=self.bg_sidebar, fg=self.fg_text, selectcolor=self.bg_sidebar,
                       activebackground=self.bg_sidebar).pack(anchor="w")
        tk.Radiobutton(q_frame, text="4-bit", variable=self.quant_mode, value="4bit",
                       bg=self.bg_sidebar, fg=self.fg_text, selectcolor=self.bg_sidebar,
                       activebackground=self.bg_sidebar).pack(anchor="w")

        # Language selection
        lang_frame = tk.Frame(self.sidebar, bg=self.bg_sidebar)
        lang_frame.pack(fill=tk.X, padx=10, pady=10)
        tk.Label(lang_frame, text="Language:", bg=self.bg_sidebar, fg=self.fg_muted,
                 font=("Helvetica", 9)).pack(anchor="w")
        tk.Radiobutton(lang_frame, text="Auto", variable=self.language_mode, value="auto",
                       bg=self.bg_sidebar, fg=self.fg_text, selectcolor=self.bg_sidebar,
                       activebackground=self.bg_sidebar).pack(anchor="w")
        tk.Radiobutton(lang_frame, text="English", variable=self.language_mode, value="en",
                       bg=self.bg_sidebar, fg=self.fg_text, selectcolor=self.bg_sidebar,
                       activebackground=self.bg_sidebar).pack(anchor="w")
        tk.Radiobutton(lang_frame, text="中文", variable=self.language_mode, value="zh",
                       bg=self.bg_sidebar, fg=self.fg_text, selectcolor=self.bg_sidebar,
                       activebackground=self.bg_sidebar).pack(anchor="w")

        # Main area
        self.main_area = tk.Frame(self.root, bg=self.bg_main)
        self.main_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.header_frame = tk.Frame(self.main_area, bg=self.bg_main, height=50)
        self.header_frame.pack(side=tk.TOP, fill=tk.X)
        self.header_label = tk.Label(self.header_frame, text="CATSEEKR0.0B – Code Interpreter & Sandbox (BitNet 4‑bit) ▼",
                                     bg=self.bg_main, fg=self.fg_text, font=("Helvetica", 12, "bold"))
        self.header_label.pack(side=tk.LEFT, padx=20, pady=15)

        self.chat_frame = tk.Frame(self.main_area, bg=self.bg_main)
        self.chat_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=40, pady=10)

        self.console_font = font.Font(family="Helvetica", size=11)
        self.console = tk.Text(self.chat_frame, bg=self.bg_main, fg=self.fg_text,
                               font=self.console_font, state=tk.DISABLED, wrap=tk.WORD,
                               relief=tk.FLAT, insertbackground=self.fg_text)
        self.console.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

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

        sys.stdout = RedirectText(self.console)
        self.print_welcome()

    def print_welcome(self):
        print("CATSEEKR0.0B – Code Interpreter & Sandbox (Microsoft BitNet 1.58/2/4-bit + o1 Chip)")
        print("• Type Python code inside triple backticks (```python ... ```) to execute.")
        print("• Language: auto-detects Chinese/English. You can force with sidebar.")
        print("• Quantization mode: select 1.58-bit, 2-bit, or 4-bit in sidebar.")
        print("• Sandbox: restricted builtins, no file I/O, 2‑second timeout.\n")

    def detect_language(self, text):
        """Return 'zh' if any CJK character, else 'en'."""
        if self.language_mode.get() != "auto":
            return self.language_mode.get()
        for ch in text:
            if '\u4e00' <= ch <= '\u9fff':
                return 'zh'
        return 'en'

    def get_response_prefix(self, lang):
        return "🐱 CATSEEKR0.0B: " if lang == 'en' else "🐱 CATSEEKR0.0B: "

    def handle_input(self, event=None):
        user_text = self.input_box.get().strip()
        if not user_text:
            return
        self.input_box.delete(0, tk.END)
        print(f"\nYou: {user_text}\n")
        cmd = user_text.lower()
        if cmd in ["cat", "summon cat", "meow"]:
            self.summon_cat()
        elif cmd in ["clear", "new chat"]:
            self.clear_console()
        elif "```python" in user_text or user_text.startswith(">>>") or user_text.startswith("...") or user_text.strip().startswith(("print", "def ", "class ", "import ")):
            # Likely code – run interpreter
            self.start_code_interpreter(user_text)
        else:
            # Fallback to text generation
            self.start_simulation(user_text)

    def summon_cat(self):
        print("CATSEEKR0.0B:" + CodebaseCatR1.get_ascii())
        print("Codebase Cat R1:", CodebaseCatR1.purr(), "\n")

    def clear_console(self):
        self.console.config(state=tk.NORMAL)
        self.console.delete(1.0, tk.END)
        self.console.config(state=tk.DISABLED)
        self.print_welcome()

    def start_simulation(self, prompt):
        self.btn_run.config(state=tk.DISABLED)
        threading.Thread(target=self.run_simulation_logic, args=(prompt,), daemon=True).start()

    def start_code_interpreter(self, code_text):
        self.btn_run.config(state=tk.DISABLED)
        threading.Thread(target=self.run_code_interpreter, args=(code_text,), daemon=True).start()

    # -------------------------------------------------------------------------
    # Sandboxed code interpreter (unchanged)
    # -------------------------------------------------------------------------
    def run_code_interpreter(self, code_text):
        lang = self.detect_language(code_text)
        prefix = self.get_response_prefix(lang)
        print(f"{prefix} Executing code in sandbox...")

        # Extract code from triple backticks if present
        if "```python" in code_text:
            code_lines = []
            in_block = False
            for line in code_text.splitlines():
                if line.strip().startswith("```python"):
                    in_block = True
                elif line.strip().startswith("```") and in_block:
                    in_block = False
                elif in_block:
                    code_lines.append(line)
            code = "\n".join(code_lines)
        else:
            code = code_text  # assume it's raw code

        # Prepare restricted environment
        safe_builtins = {
            'abs': abs, 'all': all, 'any': any, 'bin': bin, 'bool': bool,
            'chr': chr, 'dict': dict, 'dir': dir, 'divmod': divmod,
            'enumerate': enumerate, 'filter': filter, 'float': float,
            'format': format, 'frozenset': frozenset, 'hash': hash,
            'hex': hex, 'int': int, 'isinstance': isinstance, 'issubclass': issubclass,
            'len': len, 'list': list, 'map': map, 'max': max, 'min': min,
            'oct': oct, 'ord': ord, 'pow': pow, 'range': range, 'repr': repr,
            'reversed': reversed, 'round': round, 'set': set, 'slice': slice,
            'sorted': sorted, 'str': str, 'sum': sum, 'tuple': tuple,
            'type': type, 'zip': zip,
            'math': math,
        }
        allowed_modules = {'math': math}
        globals_dict = {
            '__builtins__': safe_builtins,
            '__name__': '__sandbox__',
            '__doc__': None,
            'math': math,
        }

        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        output = ""
        error = None
        try:
            result_container = []
            def target():
                try:
                    exec(code, globals_dict)
                    result_container.append(True)
                except Exception as e:
                    result_container.append(e)

            t = threading.Thread(target=target)
            t.daemon = True
            t.start()
            t.join(timeout=2)
            if t.is_alive():
                output = "Execution timed out (>2 seconds)."
            else:
                if result_container and isinstance(result_container[0], Exception):
                    error = result_container[0]
                else:
                    output = sys.stdout.getvalue()
        except Exception as e:
            error = e
        finally:
            sys.stdout = old_stdout

        if error:
            response = f"Error: {error}"
        else:
            response = output if output else "(No output)"

        if lang == 'zh':
            if "Error" in response:
                response = response.replace("Error", "错误")
            elif "No output" in response:
                response = "（无输出）"

        print(f"{prefix}\n{response}\n")
        self.root.after(0, lambda: self.btn_run.config(state=tk.NORMAL))

    # -------------------------------------------------------------------------
    # Simulation with selected quantization mode
    # -------------------------------------------------------------------------
    def run_simulation_logic(self, prompt):
        lang = self.detect_language(prompt)
        prefix = self.get_response_prefix(lang)
        quant = self.quant_mode.get()

        print(f"CATSEEKR0.0B: Spooling up Pure Python Engine (quantization = {quant})...")

        # Load model with selected quantization mode
        model = DeepSeekBitNetModel(vocab_size=50272, dim=64, num_layers=8, quant_mode=quant)
        time.sleep(0.5)

        seed_token = sum(ord(c) for c in prompt) % 50272 if prompt else 42
        current_token = seed_token

        # Simulated reasoning block (bilingual)
        if lang == 'zh':
            print("    <思考>")
            print("      • 分析提示词标记轨迹...")
            print(f"      • 量化模式: {quant}")
            print("      • MLA潜在压缩：上下文向量压缩至低秩空间。")
            print("      • MoE路由：通过共享BitNet门控激活专家。")
            print("      • 准备自回归生成循环。")
            print("    </思考>\n")
        else:
            print("    <think>")
            print("      • Analyzing prompt token trajectory...")
            print(f"      • Quantization mode: {quant}")
            print("      • MLA latent compression: context vector reduced to low-rank space.")
            print("      • MoE routing: activating experts via shared BitNet gating.")
            print("      • Preparing autoregressive generation loop.")
            print("    </think>\n")

        print(prefix, end="")
        sys.stdout.flush()

        # Vocabularies
        en_vocab = [
            "quantized", "neural", "pathways", "optimized", "routing", "cat", "meow", "purr",
            "matrix", "vector", "attention", "mechanism", "DeepSeek", "BitNet", "1-bit",
            "weights", "activations", "efficient", "compute", "scale", "precision", "logic",
            "reasoning", "chip", "active", "parameters", "MoE", "expert", "is", "fully",
            "operational", "streaming", "tokens", "inference", "ternary", "BitNet b1.58"
        ]
        zh_vocab = [
            "量化", "神经", "路径", "优化", "路由", "猫", "喵", "呼噜",
            "矩阵", "向量", "注意力", "机制", "深度求索", "比特网络", "1比特",
            "权重", "激活", "高效", "计算", "规模", "精度", "逻辑",
            "推理", "芯片", "活跃", "参数", "混合专家", "专家", "是", "完全",
            "运行", "流式", "令牌", "推断", "三元", "比特网络b1.58"
        ]

        vocab = zh_vocab if lang == 'zh' else en_vocab

        for pos in range(24):
            logits = model.forward_token(current_token, pos)
            predicted_token = logits.index(max(logits))
            word = vocab[predicted_token % len(vocab)]
            print(word + (" " if lang == 'en' else ""), end="")
            sys.stdout.flush()
            current_token = predicted_token
            time.sleep(0.15)

        print("\n\n[Generation Loop Complete.]\n")
        self.root.after(0, lambda: self.btn_run.config(state=tk.NORMAL))

if __name__ == "__main__":
    random.seed(42)
    root = tk.Tk()
    app = CATSEEKR0_0BGUI(root)
    root.mainloop()
