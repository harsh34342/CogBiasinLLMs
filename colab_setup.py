#!/usr/bin/env python3
"""
colab_setup.py
--------------
One-shot setup for running MACBOS on Google Colab (or any fresh clone).

Run once, then use main.py normally:

    !python colab_setup.py

    # smoke test
    !python main.py --quick --bias framing --mitigation baseline \\
                    --model Qwen/Qwen2.5-0.5B-Instruct

    # full run
    !python main.py --model Qwen/Qwen2.5-1.5B-Instruct --mitigation all

What it does:
  1. Creates agents/, biases/, data/ subpackages with __init__.py
  2. Copies source files from the flat repo root into each subpackage
  3. Patches max_new_tokens kwarg collision in debiasing_agent / evaluator_agent
  4. Adds Qwen (ChatML) chat-format support to base_agent
"""

import os
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))


def ensure_package(pkg):
    path = os.path.join(ROOT, pkg)
    os.makedirs(path, exist_ok=True)
    init = os.path.join(path, "__init__.py")
    if not os.path.exists(init):
        open(init, "w").close()
    print(f"  [ok] {pkg}/__init__.py")


def copy_file(src_name, dst_rel):
    src = os.path.join(ROOT, src_name)
    dst = os.path.join(ROOT, dst_rel)
    if not os.path.exists(src):
        print(f"  [WARN] source not found: {src_name}")
        return False
    shutil.copy2(src, dst)
    print(f"  [ok] {src_name} → {dst_rel}")
    return True


def patch_file(path_rel, old, new, description):
    path = os.path.join(ROOT, path_rel)
    with open(path) as f:
        src = f.read()
    if old not in src:
        print(f"  [skip] {description} — already applied or pattern not found")
        return
    with open(path, "w") as f:
        f.write(src.replace(old, new, 1))
    print(f"  [ok] {description}")


# ── Step 1: create subpackages ────────────────────────────────────────────────
print("1. Creating subpackages...")
for pkg in ("agents", "biases", "data"):
    ensure_package(pkg)

# ── Step 2: copy files into subpackages ───────────────────────────────────────
print("\n2. Copying files...")
for src, dst in [
    ("base_agent.py",       "agents/base_agent.py"),
    ("decision_agent.py",   "agents/decision_agent.py"),
    ("debiasing_agent.py",  "agents/debiasing_agent.py"),
    ("evaluator_agent.py",  "agents/evaluator_agent.py"),
    ("anchoring.py",        "biases/anchoring.py"),
    ("framing.py",          "biases/framing.py"),
    ("student_profiles.py", "data/student_profiles.py"),
]:
    copy_file(src, dst)

# ── Step 3: patch max_new_tokens kwarg collision ───────────────────────────────
# Both DebiasingAgent and EvaluatorAgent hardcode max_new_tokens in their
# __init__ AND pass **kwargs (which also contains max_new_tokens from the
# caller) to super().__init__() → TypeError: got multiple values for argument.
# Fix: pop it from kwargs before the super() call.
print("\n3. Patching max_new_tokens kwarg collision...")
_KW_OLD = (
    '    def __init__(self, model_name: str, **kwargs):\n'
    '        super().__init__('
)
_KW_NEW = (
    '    def __init__(self, model_name: str, **kwargs):\n'
    "        kwargs.pop('max_new_tokens', None)\n"
    '        super().__init__('
)
for agent_file in ("agents/debiasing_agent.py", "agents/evaluator_agent.py"):
    patch_file(agent_file, _KW_OLD, _KW_NEW,
               f"max_new_tokens kwarg fix in {agent_file}")

# ── Step 4: add Qwen / ChatML chat format to base_agent ───────────────────────
# Qwen2.5-Instruct models use the ChatML token format.
# We insert the handler just before the Falcon block so it takes priority.
print("\n4. Patching Qwen ChatML format in agents/base_agent.py...")
_QWEN_OLD = '        # Falcon instruct format'
_QWEN_NEW = (
    '        # Qwen / ChatML format\n'
    '        if "qwen" in model_lower:\n'
    '            return (\n'
    '                f"<|im_start|>system\\n{system}\\n<|im_end|>\\n"\n'
    '                f"<|im_start|>user\\n{user}\\n<|im_end|>\\n"\n'
    '                f"<|im_start|>assistant\\n"\n'
    '            )\n'
    '\n'
    '        # Falcon instruct format'
)
patch_file("agents/base_agent.py", _QWEN_OLD, _QWEN_NEW,
           "Qwen ChatML format in agents/base_agent.py")

# ── Done ──────────────────────────────────────────────────────────────────────
print("""
Setup complete.

Smoke test:
  !python main.py --quick --bias framing --mitigation baseline \\
                  --model Qwen/Qwen2.5-0.5B-Instruct

Full run:
  !python main.py --model Qwen/Qwen2.5-1.5B-Instruct --mitigation all
""")
