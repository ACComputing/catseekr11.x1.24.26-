"""
Pure Python 3.14 CATSEEKR1.0B – BitNet b1.58 + DeepSeek-R1 MLA Simulator
NO FILES • NO DEPENDENCIES • 600x400 THREAD-SAFE GUI
Inspired by DeepSeek-R1 (arXiv:2501.12948) & BitNet b1.58
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
    return [x / (1.0 + math.exp(-x)) if abs(x) < 700 else 0.0 for x in v]

def softmax(v):
    if not v:
        return []
    max_val = max(v)
    exps = [math.exp(x - max_val) for x in v]
    s = sum(exps)
    return [e / s for e in exps]

def rms_norm(v, weight, eps=1e-6):
    mean_sq = sum(x * x for x in v) / len(v)
    inv_std = 1.0 / math.sqrt(mean_sq + eps)
    return [(x * inv_std) * w for x, w in zip(v, weight)]

# =============================================================================
# 1. BitNet b1.58 Quantization (ternary weights: -γ, 0, +γ)
# =============================================================================
def quantize_activation(v, bit_width=8):
    max_abs = max(abs(x) for x in v) or 1e-5
    qmax = (1 << (bit_width - 1)) - 1
    scale = qmax / max_abs
    quantized = [max(-qmax-1, min(qmax, round(x * scale))) for x in v]
    return quantized, scale

def quantize_weight(mat, quant_mode):
    total = len(mat) * len(mat[0]) or 1
    gamma = max(1e-5, sum(abs(x) for row in mat for x in row) / total)

    if quant_mode == '1.58bit':  # true ternary {-γ, 0, +γ}
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
            q_row = [levels[min(range(4), key=lambda i: abs((x/gamma)-levels[i]))] * gamma for x in row]
            q_mat.append(q_row)
        return q_mat, gamma

    elif quant_mode == '4bit':
        qmax = 7
        q_mat = []
        for row in mat:
            q_row = []
            for x in row:
                q_val = max(-qmax-1, min(qmax, round(x / gamma * qmax)))
                q_row.append(q_val * gamma / qmax)
            q_mat.append(q_row)
        return q_mat, gamma

    raise ValueError("quant_mode must be '1.58bit', '2bit', or '4bit'")

class BitLinear:
    def __init__(self, in_features, out_features, quant_mode='1.58bit'):
        self.quant_mode = quant_mode
        self.weight = rand_matrix(out_features, in_features)

    def forward(self, x):
        act_bit = 4 if self.quant_mode == '4bit' else 8
        x_q, x_scale = quantize_activation(x, act_bit)
        w_q, w_scale = quantize_weight(self.weight, self.quant_mode)
        out_q = mat_vec_mul(w_q, x_q)
        return vec_mul_scalar(out_q, 1.0 / (x_scale * w_scale))

# =============================================================================
# 2-4. DeepSeek-R1 Style Model (MLA + DeepSeekMoE)
# =============================================================================
class DeepSeekMLA:
    def __init__(self, dim, num_heads, latent_dim_kv, latent_dim_q, quant_mode='1.58bit'):
        self.dim = dim
        self.head_dim = dim // num_heads
        self.w_down_kv = BitLinear(dim, latent_dim_kv, quant_mode)
        self.w_up_k = BitLinear(latent_dim_kv, dim, quant_mode)
        self.w_up_v = BitLinear(latent_dim_kv, dim, quant_mode)
        self.w_down_q = BitLinear(dim, latent_dim_q, quant_mode)
        self.w_up_q = BitLinear(latent_dim_q, dim, quant_mode)
        self.w_out = BitLinear(dim, dim, quant_mode)

    def apply_rope(self, vec, pos):
        out = [0.0] * len(vec)
        for i in range(0, len(vec)-1, 2):
            freq = 1.0 / (10000 ** (i / len(vec)))
            theta = pos * freq
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            out[i] = vec[i] * cos_t - vec[i+1] * sin_t
            out[i+1] = vec[i+1] * cos_t + vec[i] * sin_t
        if len(vec) % 2:
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
        for e in self.shared_experts:
            shared_out = vec_add(shared_out, e.forward(x))

        route_probs = softmax(self.router.forward(x))
        top_k_idx = sorted(range(len(route_probs)), key=lambda i: route_probs[i], reverse=True)[:self.top_k]
        top_k_p = [route_probs[i] for i in top_k_idx]
        norm_p = [p / (sum(top_k_p) or 1e-9) for p in top_k_p]

        routed_out = [0.0] * len(x)
        for i, idx in enumerate(top_k_idx):
            routed_out = vec_add(routed_out, vec_mul_scalar(self.routed_experts[idx].forward(x), norm_p[i]))
        return vec_add(shared_out, routed_out)

class HybridTransformerBlock:
    def __init__(self, dim, quant_mode='1.58bit'):
        self.norm_weight_1 = [1.0] * dim
        self.norm_weight_2 = [1.0] * dim
        self.mla = DeepSeekMLA(dim, 4, dim//4, dim//4, quant_mode)
        self.moe = CatR1XMoE(dim, 1, 4, 2, quant_mode)

    def forward(self, x, pos):
        x = vec_add(x, self.mla.forward(rms_norm(x, self.norm_weight_1), pos))
        return vec_add(x, self.moe.forward(rms_norm(x, self.norm_weight_2)))

class DeepSeekBitNetModel:
    def __init__(self, vocab_size=50272, dim=64, num_layers=8, quant_mode='1.58bit'):
        self.embed_matrix = rand_matrix(vocab_size, dim)
        self.layers = [HybridTransformerBlock(dim, quant_mode) for _ in range(num_layers)]
        self.norm_weight_f = [1.0] * dim
        self.lm_head = BitLinear(dim, vocab_size, quant_mode)

    def forward_token(self, token_id, pos):
        x = self.embed_matrix[max(0, min(token_id, len(self.embed_matrix)-1))]
        for layer in self.layers:
            x = layer.forward(x, pos)
        return self.lm_head.forward(rms_norm(x, self.norm_weight_f))

# =============================================================================
# 5. Codebase Cat R1 – with R1 paper aha moment
# =============================================================================
class CodebaseCatR1:
    @staticmethod
    def get_ascii():
        return """     /\\_/\\     [ CATSEEKR1.0B – BitNet b1.58 + DeepSeek-R1 ACTIVATED ]
    ( o.o )  > "Meow. 1.58-bit ternary loaded. R1-style reasoning engaged."
     > ^ <"""
    @staticmethod
    def purr():
        return random.choice([
            "DeepSeek-R1 RL pipeline engaged – aha moment incoming!",
            "BitNet b1.58 ternary weights + MLA KV compression = purrfect efficiency.",
            "Cold-start reflection complete. Self-verification loop active."
        ])

# =============================================================================
# 6. GUI (compact 600x400)
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

class CATSEEKR1_0BGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CATSEEKR1.0B – BitNet b1.58 + DeepSeek-R1 Simulator")
        self.root.geometry("600x400")

        self.bg_sidebar = "#171717"
        self.bg_main = "#212121"
        self.bg_input = "#2f2f2f"
        self.fg_text = "#ececec"

        self.root.configure(bg=self.bg_main)
        self.language_mode = tk.StringVar(value="auto")
        self.quant_mode = tk.StringVar(value="1.58bit")

        # Compact sidebar (width 180)
        self.sidebar = tk.Frame(self.root, bg=self.bg_sidebar, width=180)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        self.btn_new = tk.Button(self.sidebar, text="+ New Chat", bg=self.bg_sidebar, fg="#000000",
                                 font=("Helvetica", 10), relief=tk.FLAT, command=self.clear_console)
        self.btn_new.pack(fill=tk.X, padx=8, pady=6)

        tk.Label(self.sidebar, text="Today", bg=self.bg_sidebar, fg="#9b9b9b",
                 font=("Helvetica", 9, "bold")).pack(fill=tk.X, padx=15, pady=(10,3))

        self.btn_run = tk.Button(self.sidebar, text="▶ Run R1 Demo",
                                 command=lambda: self.start_simulation("Tell me a hard math riddle."),
                                 bg=self.bg_sidebar, fg="#000000", relief=tk.FLAT)
        self.btn_run.pack(fill=tk.X, padx=8, pady=2)

        self.btn_cat = tk.Button(self.sidebar, text="🐱 Summon Cat", command=self.summon_cat,
                                 bg=self.bg_sidebar, fg="#000000", relief=tk.FLAT)
        self.btn_cat.pack(fill=tk.X, padx=8, pady=2)

        # Quant selector
        qf = tk.Frame(self.sidebar, bg=self.bg_sidebar)
        qf.pack(fill=tk.X, padx=8, pady=4)
        tk.Label(qf, text="Quant:", bg=self.bg_sidebar, fg="#9b9b9b", font=("Helvetica", 9)).pack(anchor="w")
        for mode in ["1.58bit", "2bit", "4bit"]:
            tk.Radiobutton(qf, text=mode, variable=self.quant_mode, value=mode,
                           bg=self.bg_sidebar, fg=self.fg_text, selectcolor=self.bg_sidebar).pack(anchor="w")

        # Language
        lf = tk.Frame(self.sidebar, bg=self.bg_sidebar)
        lf.pack(fill=tk.X, padx=8, pady=4)
        tk.Label(lf, text="Lang:", bg=self.bg_sidebar, fg="#9b9b9b", font=("Helvetica", 9)).pack(anchor="w")
        for txt, val in [("Auto", "auto"), ("EN", "en"), ("中文", "zh")]:
            tk.Radiobutton(lf, text=txt, variable=self.language_mode, value=val,
                           bg=self.bg_sidebar, fg=self.fg_text, selectcolor=self.bg_sidebar).pack(anchor="w")

        # Main area
        main = tk.Frame(self.root, bg=self.bg_main)
        main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(main, text="CATSEEKR1.0B – BitNet b1.58 + DeepSeek-R1", bg=self.bg_main, fg=self.fg_text,
                 font=("Helvetica", 11, "bold")).pack(pady=8)

        self.console = tk.Text(main, bg=self.bg_main, fg=self.fg_text, font=("Helvetica", 10), state=tk.DISABLED, wrap=tk.WORD)
        self.console.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Input
        inf = tk.Frame(main, bg=self.bg_main)
        inf.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=8)
        self.input_box = tk.Entry(inf, bg=self.bg_input, fg=self.fg_text, font=("Helvetica", 10), relief=tk.FLAT)
        self.input_box.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8)
        self.input_box.bind("<Return>", self.handle_input)
        tk.Button(inf, text="↑", bg="#ececec", fg="#000000", font=("Helvetica", 12, "bold"),
                  relief=tk.FLAT, command=self.handle_input).pack(side=tk.RIGHT, padx=4)

        sys.stdout = RedirectText(self.console)
        self.print_welcome()

    def print_welcome(self):
        print("CATSEEKR1.0B – Pure Python BitNet b1.58 + DeepSeek-R1 Simulator")
        print("• 600×400 compact GUI • Default 1.58-bit ternary quantization")
        print("• R1-style <think>…</think><answer>…</answer> + self-reflection")
        print("• Based on arXiv:2501.12948 (DeepSeek-R1) & BitNet papers\n")

    def detect_language(self, text):
        if self.language_mode.get() != "auto":
            return self.language_mode.get()
        for ch in text:
            if '\u4e00' <= ch <= '\u9fff':
                return 'zh'
        return 'en'

    def get_response_prefix(self, lang):
        return "🐱 CATSEEKR1.0B: "

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
        elif "```python" in user_text or user_text.startswith(">>>") or user_text.strip().startswith(("print", "def ", "class ", "import ")):
            self.start_code_interpreter(user_text)
        else:
            self.start_simulation(user_text)

    def summon_cat(self):
        print("CATSEEKR1.0B:" + CodebaseCatR1.get_ascii())
        print("Codebase Cat R1:", CodebaseCatR1.purr(), "\n")

    def clear_console(self):
        self.console.config(state=tk.NORMAL)
        self.console.delete(1.0, tk.END)
        self.console.config(state=tk.DISABLED)
        self.print_welcome()

    def start_simulation(self, prompt):
        self.btn_run.config(state=tk.DISABLED)
        self.btn_cat.config(state=tk.DISABLED)
        threading.Thread(target=self.run_simulation_logic, args=(prompt,), daemon=True).start()

    def start_code_interpreter(self, code_text):
        self.btn_run.config(state=tk.DISABLED)
        self.btn_cat.config(state=tk.DISABLED)
        threading.Thread(target=self.run_code_interpreter, args=(code_text,), daemon=True).start()

    def run_code_interpreter(self, code_text):
        lang = self.detect_language(code_text)
        prefix = self.get_response_prefix(lang)
        print(f"{prefix} Executing code in sandbox...")

        # Extract code from triple backticks
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
            code = code_text

        # Restricted environment
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
        self.root.after(0, lambda: self.btn_cat.config(state=tk.NORMAL))

    def run_simulation_logic(self, prompt):
        lang = self.detect_language(prompt)
        prefix = self.get_response_prefix(lang)
        quant = self.quant_mode.get()
        print(f"CATSEEKR1.0B: Loading {quant} DeepSeek-R1 style engine...")
        model = DeepSeekBitNetModel(quant_mode=quant)
        time.sleep(0.3)

        # Simulated reasoning block (R1 style)
        if lang == 'zh':
            print(prefix + "<思考>")
            print("  • 分析提示词标记轨迹...")
            print(f"  • 量化模式: {quant}")
            print("  • MLA潜在压缩: 上下文向量压缩至低秩空间。")
            print("  • MoE路由: 激活共享专家和路由专家。")
            print("  • 准备自回归生成，可能触发R1‑Zero式自省。")
            print("</思考>")
        else:
            print(prefix + "<think>")
            print("  • Analyzing prompt token trajectory...")
            print(f"  • Quantization mode: {quant}")
            print("  • MLA latent compression active.")
            print("  • MoE routing: shared + routed experts engaged.")
            print("  • Preparing autoregressive generation – R1‑Zero 'aha moments' may occur.")
            print("</think>")

        print(prefix + "<answer>", end="")
        sys.stdout.flush()

        # Simplified generation with aha moments
        seed_token = sum(ord(c) for c in prompt) % 50272 if prompt else 42
        current_token = seed_token
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
        generated = []

        for pos in range(18):
            logits = model.forward_token(current_token, pos)
            predicted_token = logits.index(max(logits))
            word = vocab[predicted_token % len(vocab)]
            generated.append(word)

            # Simulate aha moment (approx. every 6 tokens)
            if pos > 3 and pos % 6 == 0 and random.random() < 0.4:
                print("\n")
                if lang == 'zh':
                    print("    [💡 啊哈！模型暂停反思...]")
                    if random.random() < 0.3:
                        old_word = generated[-1]
                        new_word = random.choice(vocab)
                        generated[-1] = new_word
                        print(f"    [自校正: '{old_word}' → '{new_word}']")
                else:
                    print("    [💡 Aha! Model reflects...]")
                    if random.random() < 0.3:
                        old_word = generated[-1]
                        new_word = random.choice(vocab)
                        generated[-1] = new_word
                        print(f"    [Self‑correction: '{old_word}' → '{new_word}']")
                print(prefix + "".join(generated), end="")
                sys.stdout.flush()
            else:
                print(word + (" " if lang == 'en' else ""), end="")
                sys.stdout.flush()
                current_token = predicted_token
                time.sleep(0.12)

        print("\n</answer>\n[Generation complete – R1 self-evolution simulated]\n")
        self.root.after(0, lambda: self.btn_run.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.btn_cat.config(state=tk.NORMAL))

if __name__ == "__main__":
    random.seed(42)
    root = tk.Tk()
    app = CATSEEKR1_0BGUI(root)
    root.mainloop()
