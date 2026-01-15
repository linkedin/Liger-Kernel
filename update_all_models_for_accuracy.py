#!/usr/bin/env python3
"""
Script to update all model files to support accuracy tracking.

This script updates the lce_forward function in all model files to:
1. Import the unpack_liger_loss_with_accuracy helper
2. Use the helper to extract accuracy from LigerForCausalLMLoss results
"""

import re

from pathlib import Path


def update_model_file(filepath):
    """Update a single model file with the accuracy tracking pattern."""
    print(f"Processing {filepath.name}...")

    with open(filepath, "r") as f:
        content = f.read()

    original_content = content

    # Step 1: Add import for helper function if not already present
    if "unpack_liger_loss_with_accuracy" not in content:
        # Find the LigerForCausalLMLoss import line and add helper
        content = re.sub(
            r"from liger_kernel\.transformers\.model\.loss_utils import LigerForCausalLMLoss",
            r"from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss, unpack_liger_loss_with_accuracy",
            content,
        )

    # Step 2: Find and update the pattern where we call LigerForCausalLMLoss
    # Pattern 1: Direct assignment (loss = LigerForCausalLMLoss(...))
    pattern1 = r"(\s+)(loss\s*=\s*LigerForCausalLMLoss\()"
    if re.search(pattern1, content):
        content = re.sub(pattern1, r"\1result = LigerForCausalLMLoss(", content)

        # Now we need to add the unpack call after the LigerForCausalLMLoss block
        # Find the closing of LigerForCausalLMLoss call and add unpack line
        # This is tricky - we need to find where the function call ends
        # Look for the pattern of the function call and its closing
        pattern_call = r"(result = LigerForCausalLMLoss\([^)]+\))"
        match = re.search(pattern_call, content, re.DOTALL)

        if match:
            # Insert unpack call after the assignment
            insert_pos = match.end()
            indent = "        "  # Standard 8-space indent
            unpack_line = f"\n{indent}# Unpack loss and potentially add accuracy to output dict\n{indent}loss = unpack_liger_loss_with_accuracy(result, output)"
            content = content[:insert_pos] + unpack_line + content[insert_pos:]

    # Step 3: Update the output creation to happen BEFORE calling LigerForCausalLMLoss
    # Find where we create CausalLMOutputWithPast and move it earlier
    # This is complex, so let's do a pattern match for the return statement

    # Pattern: return CausalLMOutputWithPast(loss=loss, ...)
    return_pattern = r"(\s+)return CausalLMOutputWithPast\(\s*loss=loss,\s*logits=logits,\s*past_key_values=outputs\.past_key_values,\s*hidden_states=outputs\.hidden_states,\s*attentions=outputs\.attentions,?\s*\)"

    if re.search(return_pattern, content, re.DOTALL):
        # Replace return statement with output assignment
        content = re.sub(
            return_pattern,
            r"""\1# Update the output with final loss and logits
\1output["loss"] = loss
\1output["logits"] = logits

\1return output""",
            content,
            flags=re.DOTALL,
        )

        # Now we need to create the output dict earlier, right after determining skip_logits
        # Find the line "if skip_logits:" and insert output creation before it
        skip_logits_pattern = r"(\s+)(if skip_logits is None:.*?skip_logits = .*?\n)(.*?)(if skip_logits:)"

        def add_output_creation(match):
            indent = match.group(1)
            skip_none_block = match.group(2)
            between = match.group(3)
            if_skip = match.group(4)

            output_creation = f"""{indent}# Create output first so we can pass it to unpack helper
{indent}output = CausalLMOutputWithPast(
{indent}    loss=None,  # Will be set below
{indent}    logits=None,  # Will be set below if not skipping logits
{indent}    past_key_values=outputs.past_key_values,
{indent}    hidden_states=outputs.hidden_states,
{indent}    attentions=outputs.attentions,
{indent})

"""
            return f"{indent}{skip_none_block}{between}{output_creation}{if_skip}"

        content = re.sub(skip_logits_pattern, add_output_creation, content, flags=re.DOTALL)

    # Step 4: Fix the non-return_dict path
    non_return_pattern = (
        r"(\s+)output = \(logits,\) \+ outputs\[1:\]\s*\n\s+return \(loss,\) \+ output if loss is not None else output"
    )
    if re.search(non_return_pattern, content):
        content = re.sub(
            non_return_pattern,
            r"""\1output_tuple = (logits,) + outputs[1:]
\1return (loss,) + output_tuple if loss is not None else output_tuple""",
            content,
        )

    # Only write if content changed
    if content != original_content:
        with open(filepath, "w") as f:
            f.write(content)
        print(f"  ✓ Updated {filepath.name}")
        return True
    else:
        print(f"  - No changes needed for {filepath.name}")
        return False


def main():
    # Find all model Python files
    models_dir = Path(__file__).parent / "src" / "liger_kernel" / "transformers" / "model"

    # List of model files to update (excluding ones we already did manually)
    model_files = [
        "gemma.py",
        "gemma2.py",
        "gemma3.py",
        "qwen2.py",
        "qwen3.py",
        "qwen3_moe.py",
        "phi3.py",
        "mixtral.py",
        "llama4.py",
        "smollm3.py",
        "qwen2_vl.py",
        "qwen2_5_vl.py",
        "olmo2.py",
        "falcon_h1.py",
        "glm4.py",
        "glm4v.py",
        "glm4v_moe.py",
        "paligemma.py",
        "llava.py",
        "mllama.py",
        "internvl.py",
    ]

    updated_count = 0
    for model_file in model_files:
        filepath = models_dir / model_file
        if filepath.exists():
            if update_model_file(filepath):
                updated_count += 1
        else:
            print(f"  ! File not found: {model_file}")

    print(f"\n✅ Updated {updated_count} model files")


if __name__ == "__main__":
    main()
