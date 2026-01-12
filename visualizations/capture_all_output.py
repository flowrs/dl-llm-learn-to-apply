"""
Capture all visualization outputs to files.
Runs each visualization non-interactively and saves output.
"""

import sys
import os
from io import StringIO
from contextlib import redirect_stdout

# Add visualizations directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Create output directory
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualization_outputs")
os.makedirs(output_dir, exist_ok=True)

def mock_pause(msg="Press Enter to continue..."):
    """Replace pause with a separator line"""
    print(f"\n{'='*60}\n")

def mock_input(prompt=""):
    """Mock input to return empty string"""
    print(prompt)
    return ""

def mock_clear():
    """Mock clear screen"""
    print("\n" + "="*70 + "\n")

def capture_visualization(module_name, func_name, output_file):
    """Capture output from a visualization function"""
    try:
        # Import the module
        module = __import__(module_name)

        # Replace pause and input functions
        if hasattr(module, 'pause'):
            module.pause = mock_pause
        if hasattr(module, 'clear_screen'):
            module.clear_screen = mock_clear

        # Also patch builtins for any direct input() calls
        original_input = __builtins__['input'] if isinstance(__builtins__, dict) else __builtins__.input
        if isinstance(__builtins__, dict):
            __builtins__['input'] = mock_input
        else:
            __builtins__.input = mock_input

        # Get the function
        func = getattr(module, func_name)

        # Capture stdout
        output = StringIO()
        with redirect_stdout(output):
            try:
                func()
            except (EOFError, KeyboardInterrupt):
                pass

        # Restore input
        if isinstance(__builtins__, dict):
            __builtins__['input'] = original_input
        else:
            __builtins__.input = original_input

        return output.getvalue()
    except Exception as e:
        return f"Error capturing {module_name}.{func_name}: {str(e)}\n"

def main():
    print("Capturing all visualization outputs...")
    print(f"Output directory: {output_dir}\n")

    # Define all visualizations to capture
    visualizations = {
        "week_01_02": {
            "module": "week_01_02_visualizer",
            "functions": [
                ("visualize_neuron", "01_neuron"),
                ("visualize_forward_pass", "02_forward_pass"),
                ("visualize_gradient_descent", "03_gradient_descent"),
                ("visualize_backprop", "04_backpropagation"),
                ("visualize_activations", "05_activations"),
                ("visualize_loss_functions", "06_loss_functions"),
            ]
        },
        "week_03_04": {
            "module": "week_03_04_visualizer",
            "functions": [
                ("visualize_convolution", "01_convolution"),
                ("visualize_padding_stride", "02_padding_stride"),
                ("visualize_pooling", "03_pooling"),
                ("visualize_cnn_architecture", "04_cnn_architecture"),
                ("visualize_batch_norm", "05_batch_norm"),
                ("visualize_dropout", "06_dropout"),
                ("visualize_famous_architectures", "07_famous_architectures"),
            ]
        },
        "week_05_06": {
            "module": "week_05_06_visualizer",
            "functions": [
                ("visualize_self_supervised", "01_self_supervised"),
                ("visualize_contrastive", "02_contrastive"),
                ("visualize_rnn", "03_rnn"),
                ("visualize_vanishing_gradient", "04_vanishing_gradient"),
                ("visualize_lstm", "05_lstm"),
                ("visualize_gru", "06_gru"),
                ("visualize_seq2seq", "07_seq2seq"),
            ]
        },
        "week_07": {
            "module": "week_07_visualizer",
            "functions": [
                ("visualize_attention_intuition", "01_attention_intuition"),
                ("visualize_self_attention", "02_self_attention"),
                ("visualize_multi_head", "03_multi_head"),
                ("visualize_transformer", "04_transformer"),
                ("visualize_causal_masking", "05_causal_masking"),
                ("visualize_positional_encoding", "06_positional_encoding"),
                ("visualize_kv_cache", "07_kv_cache"),
            ]
        },
        "week_08_10": {
            "module": "week_08_10_visualizer",
            "functions": [
                ("visualize_gen_vs_disc", "01_gen_vs_disc"),
                ("visualize_vae", "02_vae"),
                ("visualize_gan", "03_gan"),
                ("visualize_diffusion", "04_diffusion"),
                ("visualize_interpretability", "05_interpretability"),
                ("visualize_ethics", "06_ethics"),
                ("visualize_deployment", "07_deployment"),
            ]
        },
        "week_11": {
            "module": "week_11_visualizer",
            "functions": [
                ("visualize_tokenization", "01_tokenization"),
                ("visualize_pretraining", "02_pretraining"),
                ("visualize_scaling_laws", "03_scaling_laws"),
                ("visualize_finetuning", "04_finetuning"),
                ("visualize_rlhf", "05_rlhf"),
                ("visualize_prompting", "06_prompting"),
                ("visualize_rag", "07_rag"),
            ]
        },
    }

    all_output = []

    for week_name, week_data in visualizations.items():
        module_name = week_data["module"]
        week_output = []
        week_output.append("=" * 70)
        week_output.append(f"  {week_name.upper().replace('_', ' ')} VISUALIZATIONS")
        week_output.append("=" * 70)
        week_output.append("")

        print(f"Processing {week_name}...")

        for func_name, file_suffix in week_data["functions"]:
            print(f"  - {func_name}")

            # Need to reimport module fresh each time
            if module_name in sys.modules:
                del sys.modules[module_name]

            output = capture_visualization(module_name, func_name, file_suffix)
            week_output.append(output)
            week_output.append("\n" + "-" * 70 + "\n")

        # Save individual week file
        week_file = os.path.join(output_dir, f"{week_name}_output.txt")
        with open(week_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(week_output))
        print(f"  Saved to {week_file}")

        all_output.extend(week_output)
        all_output.append("\n\n")

    # Save combined file
    combined_file = os.path.join(output_dir, "all_visualizations_output.txt")
    with open(combined_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(all_output))
    print(f"\nCombined output saved to {combined_file}")

    print("\nDone! All outputs captured.")

if __name__ == "__main__":
    main()
