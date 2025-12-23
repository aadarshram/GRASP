
import os
import sys

def patch_deepspeed():
    # Path constructed relative to user specific venv structure knowing the path from previous tools
    target_file = "/home/nightfury/Desktop/GRASP/.venv/lib/python3.13/site-packages/deepspeed/runtime/zero/linear.py"
    
    if not os.path.exists(target_file):
        print(f"File not found: {target_file}")
        return

    with open(target_file, 'r') as f:
        content = f.read()

    # The block to replace
    old_block = """try:
    autocast_custom_fwd = get_accelerator().amp().custom_fwd
    autocast_custom_bwd = get_accelerator().amp().custom_bwd
except (ImportError, AttributeError) as exp:
    autocast_custom_fwd = noop_decorator
    autocast_custom_bwd = noop_decorator"""

    new_block = """try:
    import torch
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'custom_fwd'):
        from functools import partial
        autocast_custom_fwd = partial(torch.amp.custom_fwd, device_type='cuda')
        autocast_custom_bwd = partial(torch.amp.custom_bwd, device_type='cuda')
    else:
        autocast_custom_fwd = get_accelerator().amp().custom_fwd
        autocast_custom_bwd = get_accelerator().amp().custom_bwd
except (ImportError, AttributeError) as exp:
    autocast_custom_fwd = noop_decorator
    autocast_custom_bwd = noop_decorator"""

    if old_block in content:
        print("Patching file...")
        new_content = content.replace(old_block, new_block)
        with open(target_file, 'w') as f:
            f.write(new_content)
        print("Patch applied successfully.")
    elif new_block in content:
        print("Patch already applied.")
    else:
        print("Could not find the exact code block to replace. It might differ or be already modified.")
        # Debug: print snippet
        start_idx = content.find("try:")
        if start_idx != -1:
             print("Found snippet start:", content[start_idx:start_idx+200])

if __name__ == "__main__":
    patch_deepspeed()
