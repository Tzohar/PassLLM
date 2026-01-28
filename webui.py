import os
import gradio as gr
import sys
import time
import re
import math
import json
import subprocess
import threading
import importlib
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent  # passllm/
src_path = project_root / "src"
models_dir = project_root / "models"
config_file_path = src_path / "config.py"
target_file_path = project_root / "target.jsonl"
app_script_path = project_root / "app.py"

file_lock = threading.Lock()

## --- CONFIG IMPORT & FACTORY DEFAULTS ---

missing_files = []
if not app_script_path.exists():
    missing_files.append("app.py (Inference Engine)")
if not config_file_path.exists():
    missing_files.append("src/config.py (Configuration)")

if missing_files:
    print("❌ FATAL ERROR: Critical files are missing!")
    for f in missing_files:
        print(f"   - Missing: {f}")
    sys.exit(1)

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    import config 
except ImportError:
    print(f"❌ FATAL ERROR: Could not import src/config.py. Ensure the file exists at {config_file_path}")
    sys.exit(1)
except SyntaxError as e:
    print(f"❌ FATAL ERROR: Syntax error in 'src/config.py':")
    print(f"   Line {e.lineno}: {e.text.strip()}")
    sys.exit(1)

if not hasattr(config, 'Config'):
    print("❌ FATAL ERROR: 'class Config' not found in src/config.py")
    sys.exit(1)

FACTORY_DEFAULTS = {}
for key in dir(config):
    if key.startswith("DEFAULT_"):
        target_setting = key.replace("DEFAULT_", "")
        if hasattr(config.Config, target_setting):
            FACTORY_DEFAULTS[target_setting] = getattr(config, key)

if not FACTORY_DEFAULTS:
    print("⚠️ WARNING: No 'DEFAULT_' variables found in config. Factory Reset will be disabled.")
else:
    print(f"✅ Loaded {len(FACTORY_DEFAULTS)} factory default settings.")
print(f"✅ Config loaded from {config_file_path}")

# --- CONFIG READ/WRITE LOGIC ---

def write_config_to_disk(key, value):
    if not config_file_path.exists(): 
        return
    
    if isinstance(value, str):
        safe_val = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        val_str = f'"{safe_val}"'
    elif isinstance(value, bool):
        val_str = str(value)
    elif isinstance(value, float) and math.isinf(value):
        val_str = "float('-inf')" if value < 0 else "float('inf')"
    else:
        val_str = str(value)

    with file_lock:
        temp_path = config_file_path.with_suffix(".tmp")
        try:
            with open(config_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Improved Regex:
            # ^(\s*)    -> Capture indentation at start of line (Multi-line mode)
            # KEY =     -> Match the key and equals sign
            # .+?       -> Match the value non-greedily
            # (?=...)   -> Stop at a comment (#) or end of line
            pattern = r"^(\s*)(" + re.escape(key) + r"\s*=\s*)(.+?)(?=\s*(?:#|$))"
            
            new_content = re.sub(
                pattern, 
                lambda m: m.group(1) + m.group(2) + val_str, 
                content, 
                flags=re.MULTILINE
            )
            
            if new_content == content:
                return

            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
                f.flush()            
                os.fsync(f.fileno()) 
            
            retries = 3
            while retries > 0:
                try:
                    os.replace(str(temp_path), str(config_file_path))
                    break 
                except PermissionError:
                    retries -= 1
                    time.sleep(0.1) 
                    
                    if retries == 0:
                        try:
                            if os.path.exists(config_file_path):
                                os.remove(str(config_file_path))
                            os.rename(str(temp_path), str(config_file_path))
                        except Exception as final_e:
                            print(f"❌ Critical Save Error: {final_e}")
            
        except Exception as e:
            print(f"❌ File Error during config save: {e}")
            if temp_path.exists():
                try: os.remove(str(temp_path))
                except: pass

def update_setting(key, value):
    if hasattr(config, 'Config'):
        current_val = getattr(config.Config, key, None)
        
        if current_val is not None:
            try:
                if isinstance(current_val, bool):
                    if isinstance(value, str):
                        value = value.lower() in ("true", "1", "yes", "on")
                    else:
                        value = bool(value)
                
                elif isinstance(current_val, int):
                    value = int(float(value)) 
                elif isinstance(current_val, float):
                    value = float(value)
                
                elif isinstance(current_val, str):
                    value = str(value)

            except (ValueError, TypeError) as e:
                print(f"⚠️ Type casting failed for {key}: {e}. Keeping original type.")
                pass
            
        setattr(config.Config, key, value)
    
    write_config_to_disk(key, value)

def handle_ban_toggle(key, is_banned, slider_val):
    val = float('-inf') if is_banned else float(slider_val)
    update_setting(key, val)
    return gr.update(interactive=not is_banned)

def handle_slider_change(key, is_banned, slider_val):
    if not is_banned: update_setting(key, float(slider_val))

def get_current_config_values():
    C = config.Config

    def get_val(key, default_var_name, hard_fallback):
        if hasattr(C, key):
            return getattr(C, key)
        if hasattr(config, default_var_name):
            return getattr(config, default_var_name)
        return hard_fallback

    def is_inf(val): return math.isinf(val) and val < 0
    def safe_val(val): return 0.0 if math.isinf(val) else val

    return (
        # Generation Params
        get_val("MIN_PASSWORD_LENGTH", "DEFAULT_MIN_PASSWORD_LENGTH", 6), 
        get_val("MAX_PASSWORD_LENGTH", "DEFAULT_MAX_PASSWORD_LENGTH", 16),
        get_val("EPSILON_END_PROB", "DEFAULT_EPSILON_END_PROB", 0.3), 
        get_val("INFERENCE_BATCH_SIZE", "DEFAULT_INFERENCE_BATCH_SIZE", 32), 
        "Standard", # Default Beam Schedule

        # Bias Values (Sliders)
        safe_val(get_val("VOCAB_BIAS_UPPER", "DEFAULT_VOCAB_BIAS_UPPER", 0.0)), 
        safe_val(get_val("VOCAB_BIAS_LOWER", "DEFAULT_VOCAB_BIAS_LOWER", 0.0)),
        safe_val(get_val("VOCAB_BIAS_DIGITS", "DEFAULT_VOCAB_BIAS_DIGITS", -1.0)), 
        safe_val(get_val("VOCAB_BIAS_SYMBOLS", "DEFAULT_VOCAB_BIAS_SYMBOLS", -1.0)),

        # Ban States (Checkboxes)
        is_inf(get_val("VOCAB_BIAS_UPPER", "DEFAULT_VOCAB_BIAS_UPPER", 0.0)), 
        is_inf(get_val("VOCAB_BIAS_LOWER", "DEFAULT_VOCAB_BIAS_LOWER", 0.0)),
        is_inf(get_val("VOCAB_BIAS_DIGITS", "DEFAULT_VOCAB_BIAS_DIGITS", -1.0)), 
        is_inf(get_val("VOCAB_BIAS_SYMBOLS", "DEFAULT_VOCAB_BIAS_SYMBOLS", -1.0)),

        # Vocabulary Lists
        get_val("VOCAB_WHITELIST", "DEFAULT_VOCAB_WHITELIST", ""), 
        get_val("VOCAB_BLACKLIST", "DEFAULT_VOCAB_BLACKLIST", " \t\r\n"),

        # Hardware & LoRA
        get_val("DEVICE", "DEFAULT_DEVICE", "cuda"), 
        get_val("TORCH_DTYPE", "DEFAULT_TORCH_DTYPE", "float16"),
        get_val("USE_4BIT", "DEFAULT_USE_4BIT", True), 
        get_val("LORA_R", "DEFAULT_LORA_R", 16), 
        get_val("LORA_ALPHA", "DEFAULT_LORA_ALPHA", 32)
    )

def reset_to_factory():
    for key, val in FACTORY_DEFAULTS.items(): update_setting(key, val)
    return get_current_config_values()

def reload_config_from_disk():
    try:
        importlib.reload(config)
        return get_current_config_values()
    except Exception as e:
        print(f"❌ Error reloading config: {e}")
        return get_current_config_values()
    
# --- PII JSON LOGIC ---

BLANK_TEMPLATE = {
    "name": "", "birth_year": "", "birth_month": "", "birth_day": "",
    "username": "", "email": "", "address": "", "phone": "",
    "country": "", "sister_pw": ""
}

pii_cache = None

def read_pii_file():
    global pii_cache
    
    if pii_cache is not None:
        return pii_cache.copy()

    if not target_file_path.exists(): 
        pii_cache = BLANK_TEMPLATE.copy()
        return pii_cache

    try:
        with file_lock:
            with open(target_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                full_data = BLANK_TEMPLATE.copy()
                full_data.update(data)
                
                pii_cache = full_data
                return full_data
    except (json.JSONDecodeError, Exception) as e:
        print(f"⚠️ Error reading PII (resetting to blank): {e}")
        return BLANK_TEMPLATE.copy()

def save_pii_data(data):
    global pii_cache
    
    pii_cache = data
    
    temp_path = target_file_path.with_suffix(".tmp")
    try:
        with file_lock:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2) 
                f.flush()
                os.fsync(f.fileno())
            
            retries = 3
            while retries > 0:
                try:
                    os.replace(str(temp_path), str(target_file_path))
                    break
                except PermissionError:
                    retries -= 1
                    time.sleep(0.1)
                    if retries == 0:
                        if target_file_path.exists(): os.remove(str(target_file_path))
                        os.rename(str(temp_path), str(target_file_path))

    except Exception as e:
        print(f"❌ Error saving PII: {e}")
        if temp_path.exists():
            try: os.remove(str(temp_path))
            except: pass

def perform_full_reset():
    save_pii_data(BLANK_TEMPLATE)
    print("✅ PII Reset.")
    return [""] * 8

def update_pii_field(key, value):
    data = read_pii_file()
    
    new_val = str(value)
    if data.get(key) != new_val:
        data[key] = new_val
        save_pii_data(data)

# --- FORMATTERS ---

def fmt_month(val):
    if not val: return ""
    digits = ''.join(filter(str.isdigit, str(val)))
    if not digits: return ""
    num = max(1, min(12, int(digits)))
    return f"{num:02d}"

def fmt_day(val):
    if not val: return ""
    digits = ''.join(filter(str.isdigit, str(val)))
    if not digits: return ""
    num = max(1, min(31, int(digits)))
    return f"{num:02d}"

def fmt_year(val):
    if not val: return ""
    digits = ''.join(filter(str.isdigit, str(val)))
    if not digits: return ""

    if len(digits) == 2:
        y = int(digits)
        prefix = "20" if y < 40 else "19"
        digits = prefix + digits
        
    return digits[:4] #

def fmt_list_commas(val):
    if not val: return ""
    items = [x.strip() for x in str(val).split(',') if x.strip()]
    return ", ".join(items)

def load_pii_to_ui():
    data = read_pii_file()
    return (
        data.get("name", ""), 
        data.get("username", ""), 
        data.get("email", ""),
        data.get("phone", ""), 
        data.get("birth_year", ""), 
        data.get("birth_month", ""),
        data.get("birth_day", ""), 
        data.get("sister_pw", "")
    )

# --- EXECUTION CONTROL ---

current_process = None
should_stop = False
process_lock = threading.Lock()

def run_inference_process(schedule_mode, model_name=None):
    global current_process, should_stop
    
    if not process_lock.acquire(blocking=False):
        yield "⚠️ Process is already running.", [], "Busy"
        return

    should_stop = False
    
    cmd = [sys.executable, "-u", "app.py", "--file", "target.jsonl"]
    
    if schedule_mode == "Fast": cmd.append("--fast")
    elif schedule_mode == "Superfast": cmd.append("--superfast")
    elif schedule_mode == "Deep Search": cmd.append("--deep")
    
    if model_name and "No models" not in model_name:
        model_path = models_dir / model_name
        if model_path.exists():
            cmd.extend(["--weights", str(model_path)])

    print(f"DEBUG: Executing: {' '.join(cmd)}")
    yield f"🚀 Launching Engine...\nCommand: {' '.join(cmd)}\n", [], "Starting..."

    try:
        current_process = subprocess.Popen(
            cmd,
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, 
            text=True,
            bufsize=1, 
            env=os.environ.copy()
        )

        log_lines = [] 
        results_data = [] 
        final_file_path = None
        
        stream_pattern = re.compile(r"(\d+\.\d+%)\s+\|\s+(.*?)(?:\s+\(.*\))?$")
        
        file_pattern = re.compile(r"Saving \d+ candidates to:\s*(.+)")

        for line in iter(current_process.stdout.readline, ""):
            if should_stop:
                current_process.terminate()
                log_lines.append("\n🛑 Stopped by user.")
                yield "".join(log_lines[-2000:]), results_data, "Stopped"
                return
            
            print(line, end="", flush=True)
            log_lines.append(line)
            if len(log_lines) > 2000: log_lines = log_lines[-2000:]

            match = stream_pattern.search(line)
            if match:
                conf = match.group(1)
                pwd = match.group(2).strip()
                results_data.append([len(results_data)+1, pwd, conf])

            path_match = file_pattern.search(line)
            if path_match:
                raw_path = path_match.group(1).strip()
                final_file_path = Path(raw_path)

            if len(log_lines) % 5 == 0:
                yield "".join(log_lines[-2000:]), results_data, "Running..."

        current_process.wait()
        
        if current_process.returncode == 0:
            
            if final_file_path and final_file_path.exists():
                try:
                    log_lines.append(f"\n📂 Reading results from: {final_file_path.name}...")
                    yield "".join(log_lines[-2000:]), results_data, "Reading File..."
                    
                    with open(final_file_path, 'r', encoding='utf-8') as f:
                        file_data = json.load(f)
                        
                        results_data = [] 
                        
                        items = file_data if isinstance(file_data, list) else file_data.get("candidates", [])
                        
                        for idx, item in enumerate(items):
                            pwd = item.get("password") or item.get("candidate", "Unknown")
                            conf = item.get("confidence") or item.get("probability", "0%")
                            
                            results_data.append([idx + 1, pwd, str(conf)])
                            
                    log_lines.append("\n✅ Loaded clean results from file.")
                    
                except Exception as file_err:
                    log_lines.append(f"\n⚠️ Could not read result file: {file_err}")
            
            yield "".join(log_lines[-2000:]), results_data, "Completed"
            
        else:
            yield "".join(log_lines[-2000:]) + f"\n❌ Exited (Code {current_process.returncode})", results_data, "Failed"

    except Exception as e:
        yield f"❌ Error: {str(e)}", [], "Error"
    
    finally:
        if process_lock.locked():
            process_lock.release()
        current_process = None

def stop_inference():
    global current_process, should_stop
    should_stop = True
    if current_process:
        try:
            current_process.terminate()
            current_process.wait(timeout=1)
        except:
            current_process.kill()
    return "🛑 Stopping...", "Stopped"


# --- MISC HELPERS ---
import os

def scan_models():
    models_dir.mkdir(parents=True, exist_ok=True)
    valid_extensions = {".pth", ".pt", ".bin", ".safetensors", ".gguf"}
    options = []
    
    if models_dir.exists():
        for item in models_dir.iterdir():
            if item.name.startswith("."):
                continue
                
            if item.is_dir():
                options.append(item.name)
            elif item.suffix.lower() in valid_extensions:
                options.append(item.name)
    
    return sorted(options) if options else ["No models found"]

def load_model_sim(model_name):
    if not model_name or "No models" in model_name:
        yield "⚠️ Invalid Selection"
        return

    try:
        safe_path = (models_dir / model_name).resolve()
        
        if not safe_path.is_relative_to(models_dir.resolve()):
            yield "❌ Security Alert: Access Denied"
            return
            
        if not safe_path.exists():
            yield "❌ Error: Model file missing"
            return

        if not os.access(safe_path, os.R_OK):
            yield "❌ Error: File locked or unreadable"
            return

        yield f"⏳ Verifying {model_name}..."
        time.sleep(0.5) 
        
        if safe_path.is_dir():
            yield f"✅ Ready: {model_name} (Model Folder)"
        else:
            size_mb = safe_path.stat().st_size / (1024 * 1024)
            
            if size_mb < 1:
                yield f"⚠️ Warning: Model seems empty ({size_mb:.2f} MB)"
            elif size_mb < 500:
                yield f"✅ Ready: {model_name} (LoRA/Adapter - {size_mb:.0f} MB)"
            else:
                yield f"✅ Ready: {model_name} (Full Model - {size_mb:.0f} MB)"
            
    except Exception as e:
        yield f"❌ Error: {str(e)}"

# --- UI CONSTRUCTION ---

CUSTOM_CSS = """
/* 1. Main Layout Tweaks */
.center-row { 
    align-items: center !important; 
    gap: 12px !important; 
}

/* 2. Fix Checkbox Alignment */
.ban-checkbox { 
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    margin-top: 0 !important; 
    height: 100% !important; 
}

/* 3. Headers & Typography */
h3 { margin-bottom: 0.5rem !important; }

.section-header {
    font-size: 1.15rem !important; 
    font-weight: 700 !important;
    letter-spacing: 0.5px;
    margin: 10px 0 5px 0 !important;
    padding-bottom: 4px;
    border-bottom: 2px solid var(--border-color-primary); /* Use Gradio variable */
    /* Color removed: Will now be White in Dark Mode / Black in Light Mode */
}

.tight-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
    margin-top: 8px;
    min-height: 40px; 
}

.tight-header .gradio-html {
    min-height: unset !important;
    padding: 0 !important;
}

/* 4. Compact PII Rows */
.pii-row {
    margin-bottom: 0 !important;
    gap: 8px !important; 
}

.pii-row > .form { 
    gap: 0 !important; 
}

/* 5. Log Window Scroll Fix */
#component-log-accordion > .label-wrap { 
    border: none !important; 
}
"""

def create_ui():
    theme = gr.themes.Default(
        primary_hue="blue", secondary_hue="slate", neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"]
    )

    with gr.Blocks(title="PassLLM Inference Engine") as app:
        
        # HEADER
        with gr.Row(variant="panel", elem_classes="header-row"):
            with gr.Column(scale=1):
                with gr.Row():
                    model_options = scan_models()
                    model_selector = gr.Dropdown(choices=model_options, value=model_options[0] if model_options else None, label="📚 Model Library", container=False, scale=3, interactive=True)
                    load_btn = gr.Button("Load Model", variant="primary", scale=1)
        
        status_bar = gr.Markdown(value="*System Ready. Select a local model to begin.*")
        gr.HTML("<hr style='border-top: 1px solid #e5e7eb; margin: 10px 0;'>")

        with gr.Row():
            
            # --- LEFT: CONFIGURATION ---
            with gr.Column(scale=1, variant="panel"):
                with gr.Group():
                    gr.HTML("<div class='section-header' title='Global actions to manage configuration.'>🛠️ Configuration Actions</div>")
                    with gr.Row(variant="compact"):
                        reload_btn = gr.Button("📂 Read from Config", size="sm", variant="secondary", elem_id="read_config_btn")
                        reset_btn = gr.Button("↺ Factory Reset", size="sm", variant="stop")

                with gr.Accordion("⚡ Generation Parameters", open=True):
                    gr.HTML("<h4 title='Hard limits for password length.'>📏 Length Constraints</h4>")
                    with gr.Group():
                        with gr.Row():
                            min_len = gr.Number(value=config.Config.MIN_PASSWORD_LENGTH, label="Min Length", precision=0, minimum=1, step=1, info="Candidates shorter than this are discarded.")
                            max_len = gr.Number(value=config.Config.MAX_PASSWORD_LENGTH, label="Max Length", precision=0, minimum=1, step=1, info="AI stops generating at this count.")
                    
                    gr.HTML("<h4 title='How the AI explores possibilities.'>🧠 Search Strategy</h4>")
                    epsilon = gr.Slider(0.0, 1.0, value=config.Config.EPSILON_END_PROB, step=0.01, label="Stop Probability (ε)", info="Confidence threshold. Higher = Stricter.")
                    batch_size = gr.Slider(1, 128, value=config.Config.INFERENCE_BATCH_SIZE, step=1, label="Batch Size", info="Parallel candidates per step.")
                    schedule_dropdown = gr.Dropdown(choices=["Standard", "Fast", "Superfast", "Deep Search"], value="Standard", label="Beam Search Schedule", interactive=True, info="Search width.")

                with gr.Accordion("🔤 Vocabulary Constraints", open=True, elem_id="voca_acc"):
                    gr.HTML("<p title='Influence char likelihood.' style='margin-bottom: 10px;'><i>Adjust bias or strict-ban categories.</i></p>")
                    with gr.Row(elem_classes="center-row"):
                        bias_upper = gr.Slider(-10, 10, value=getattr(config.Config, "VOCAB_BIAS_UPPER", 0.0), label="🔠 Upper Bias", scale=4, info="Boost Uppercase.")
                        chk_upper = gr.Checkbox(label="🚫 Ban", value=False, scale=1, min_width=20, elem_classes="ban-checkbox", info="Prohibit Uppercase.")
                    with gr.Row(elem_classes="center-row"):
                        bias_lower = gr.Slider(-10, 10, value=getattr(config.Config, "VOCAB_BIAS_LOWER", 0.0), label="🔡 Lower Bias", scale=4, info="Boost Lowercase.")
                        chk_lower = gr.Checkbox(label="🚫 Ban", value=False, scale=1, min_width=20, elem_classes="ban-checkbox", info="Prohibit Lowercase.")
                    with gr.Row(elem_classes="center-row"):
                        bias_digits = gr.Slider(-10, 10, value=config.Config.VOCAB_BIAS_DIGITS, label="🔢 Digit Bias", scale=4, info="Boost Digits.")
                        chk_digits = gr.Checkbox(label="🚫 Ban", value=False, scale=1, min_width=20, elem_classes="ban-checkbox", info="Prohibit Digits.")
                    with gr.Row(elem_classes="center-row"):
                        bias_symbols = gr.Slider(-10, 10, value=config.Config.VOCAB_BIAS_SYMBOLS, label="🔣 Symbol Bias", scale=4, info="Boost Symbols.")
                        chk_symbols = gr.Checkbox(label="🚫 Ban", value=False, scale=1, min_width=20, elem_classes="ban-checkbox", info="Prohibit Symbols.")
                    with gr.Row():
                        whitelist = gr.Textbox(value=config.Config.VOCAB_WHITELIST, label="✅ Whitelist", placeholder="@!#", lines=1, info="Always allow.")
                        blacklist = gr.Textbox(value=config.Config.VOCAB_BLACKLIST, label="🚫 Blacklist", placeholder="\\n \\t", lines=1, info="Never allow.")

                with gr.Accordion("🖥️ Hardware Acceleration", open=True, elem_id="hard_acc"):
                    device = gr.Radio(["cuda", "cpu", "dml"], label="Compute Device", value=config.Config.DEVICE)s
                    dtype = gr.Dropdown(choices=["float16", "bfloat16", "float32"], label="Torch Datatype", value=str(config.Config.TORCH_DTYPE).split('.')[-1] if 'torch' in str(config.Config.TORCH_DTYPE) else str(config.Config.TORCH_DTYPE))
                    use_4bit = gr.Checkbox(label="4-Bit Quantization", value=config.Config.USE_4BIT)

                with gr.Accordion("🧬 LoRA Adapters", open=True, elem_id="lora_acc"):
                    gr.HTML("<p style='margin-bottom: 10px;'><i>Fine-tuning weights for the inference engine.</i></p>")
                    with gr.Row():
                        lora_r = gr.Number(value=config.Config.LORA_R, label="Rank (r)", precision=0)
                        lora_alpha = gr.Number(value=config.Config.LORA_ALPHA, label="Alpha", precision=0)

            # --- RIGHT: PII INPUTS ---
            with gr.Column(scale=2):
                
                with gr.Group():
                    with gr.Row(elem_classes="tight-header"):
                         gr.HTML("<div class='section-header' title='Personal info used to customize the attack.'>🎯 Target Profile (PII)</div>")

                    # Identity
                    with gr.Row(elem_classes="pii-row"):
                        inp_name = gr.Textbox(label="👤 Full Name", placeholder="e.g. John Doe")
                        inp_username = gr.Textbox(label="👤 Username", placeholder="e.g. jdoe99")
                    
                    # Contact
                    with gr.Row(elem_classes="pii-row"):
                        inp_email = gr.Textbox(label="📧 Email", placeholder="e.g. john@example.com")
                        inp_phone = gr.Textbox(label="📞 Phone", placeholder="e.g. 555-0199")

                    # Dates
                    with gr.Row(elem_classes="pii-row"):
                        inp_year = gr.Textbox(label="📅 Year", placeholder="YYYY")
                        inp_month = gr.Textbox(label="📅 Month", placeholder="MM")
                        inp_day = gr.Textbox(label="📅 Day", placeholder="DD")
                    
                    inp_sister = gr.Textbox(label="🔑 Sister Passwords", placeholder="pass1, pass2, pass3", info="Comma separated previous passwords.")

                with gr.Row(variant="panel"):
                    pii_load_btn = gr.Button("📂 Read target.jsonl", variant="secondary", scale=1, elem_id="pii_load_btn") # MOVED HERE
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary", scale=1)
                    gen_btn = gr.Button("🔮 Generate", variant="primary", scale=2)
                    stop_btn = gr.Button("🛑 Stop", variant="stop", scale=1)

                result_status_markdown = gr.Markdown("### 📋 Results", visible=True)
                run_status = gr.Label(value="Ready", label="Status", color="gray", show_label=False)

                output_table = gr.Dataframe(
                    headers=["Rank", "Candidate", "Confidence"], 
                    datatype=["number", "str", "str"], 
                    interactive=False,
                )

                with gr.Accordion("View Raw Logs", open=False):
                    console_log = gr.Markdown(value="*Ready...*")

        # --- EVENT WIRING ---
        load_btn.click(load_model_sim, inputs=[model_selector], outputs=[status_bar])

        min_len.change(lambda x: update_setting("MIN_PASSWORD_LENGTH", int(x)), inputs=[min_len])
        max_len.change(lambda x: update_setting("MAX_PASSWORD_LENGTH", int(x)), inputs=[max_len])
        epsilon.change(lambda x: update_setting("EPSILON_END_PROB", float(x)), inputs=[epsilon])
        batch_size.change(lambda x: update_setting("INFERENCE_BATCH_SIZE", int(x)), inputs=[batch_size])

        chk_upper.change(fn=lambda b, v: handle_ban_toggle("VOCAB_BIAS_UPPER", b, v), inputs=[chk_upper, bias_upper], outputs=[bias_upper])
        chk_lower.change(fn=lambda b, v: handle_ban_toggle("VOCAB_BIAS_LOWER", b, v), inputs=[chk_lower, bias_lower], outputs=[bias_lower])
        chk_digits.change(fn=lambda b, v: handle_ban_toggle("VOCAB_BIAS_DIGITS", b, v), inputs=[chk_digits, bias_digits], outputs=[bias_digits])
        chk_symbols.change(fn=lambda b, v: handle_ban_toggle("VOCAB_BIAS_SYMBOLS", b, v), inputs=[chk_symbols, bias_symbols], outputs=[bias_symbols])
        bias_upper.change(fn=lambda b, v: handle_slider_change("VOCAB_BIAS_UPPER", b, v), inputs=[chk_upper, bias_upper])
        bias_lower.change(fn=lambda b, v: handle_slider_change("VOCAB_BIAS_LOWER", b, v), inputs=[chk_lower, bias_lower])
        bias_digits.change(fn=lambda b, v: handle_slider_change("VOCAB_BIAS_DIGITS", b, v), inputs=[chk_digits, bias_digits])
        bias_symbols.change(fn=lambda b, v: handle_slider_change("VOCAB_BIAS_SYMBOLS", b, v), inputs=[chk_symbols, bias_symbols])

        whitelist.change(lambda x: update_setting("VOCAB_WHITELIST", str(x)), inputs=[whitelist])
        blacklist.change(lambda x: update_setting("VOCAB_BLACKLIST", str(x)), inputs=[blacklist])
        device.change(lambda x: update_setting("DEVICE", str(x)), inputs=[device])
        dtype.change(lambda x: update_setting("TORCH_DTYPE", str(x)), inputs=[dtype])
        use_4bit.change(lambda x: update_setting("USE_4BIT", bool(x)), inputs=[use_4bit])
        lora_r.change(lambda x: update_setting("LORA_R", int(x)), inputs=[lora_r])
        lora_alpha.change(lambda x: update_setting("LORA_ALPHA", int(x)), inputs=[lora_alpha])

        # Update Reset/Reload output list 
        config_outputs = [
            min_len, max_len, epsilon, batch_size, schedule_dropdown, 
            bias_upper, bias_lower, bias_digits, bias_symbols, 
            chk_upper, chk_lower, chk_digits, chk_symbols, 
            whitelist, blacklist, device, dtype, use_4bit, lora_r, lora_alpha
        ]

        app.load(fn=load_pii_to_ui, inputs=None, outputs=[inp_name, inp_username, inp_email, inp_phone, inp_year, inp_month, inp_day, inp_sister])
        app.load(fn=reload_config_from_disk, inputs=None, outputs=config_outputs)

        reset_btn.click(fn=reset_to_factory, inputs=[], outputs=config_outputs)
        reload_btn.click(fn=reload_config_from_disk, inputs=[], outputs=config_outputs)
        
        # --- PII WIRING ---
        inp_name.change(lambda x: update_pii_field("name", x), inputs=[inp_name])
        inp_username.change(lambda x: update_pii_field("username", x), inputs=[inp_username])
        inp_email.change(lambda x: update_pii_field("email", x), inputs=[inp_email])
        inp_phone.change(lambda x: update_pii_field("phone", x), inputs=[inp_phone])
        inp_year.change(lambda x: update_pii_field("birth_year", x), inputs=[inp_year])
        inp_month.change(lambda x: update_pii_field("birth_month", x), inputs=[inp_month])
        inp_day.change(lambda x: update_pii_field("birth_day", x), inputs=[inp_day])
        inp_sister.change(lambda x: update_pii_field("sister_pw", x), inputs=[inp_sister])

        inp_month.blur(fn=fmt_month, inputs=[inp_month], outputs=[inp_month])
        inp_day.blur(fn=fmt_day, inputs=[inp_day], outputs=[inp_day])
        inp_year.blur(fn=fmt_year, inputs=[inp_year], outputs=[inp_year])
        inp_sister.blur(fn=fmt_list_commas, inputs=[inp_sister], outputs=[inp_sister])
        
        pii_load_btn.click(fn=load_pii_to_ui, inputs=[], outputs=[inp_name, inp_username, inp_email, inp_phone, inp_year, inp_month, inp_day, inp_sister])
        
        clear_btn.click(
            fn=perform_full_reset, 
            inputs=[], 
            outputs=[inp_name, inp_username, inp_email, inp_phone, inp_year, inp_month, inp_day, inp_sister]
        )

        # --- EXECUTION WIRING ---
        gen_btn.click(
            fn=run_inference_process,
            inputs=[schedule_dropdown, model_selector],
            outputs=[console_log, output_table, run_status]
        )
        
        stop_btn.click(
            fn=stop_inference,
            inputs=[],
            outputs=[console_log, run_status]
        )
        

    return app, theme

if __name__ == "__main__":
    app, theme = create_ui()


    app.launch(theme=theme, css=CUSTOM_CSS)
