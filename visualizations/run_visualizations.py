"""
Deep Learning Course - Interactive Visualizations
Main launcher for all week visualizers.
Run: python run_visualizations.py
"""

import os
import sys
import importlib.util

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def load_module(filepath):
    """Dynamically load a Python module from filepath."""
    spec = importlib.util.spec_from_file_location("module", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def print_banner():
    clear_screen()
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║     ██████╗ ███████╗███████╗██████╗     ██╗     ███████╗ █████╗   ║
║     ██╔══██╗██╔════╝██╔════╝██╔══██╗    ██║     ██╔════╝██╔══██╗  ║
║     ██║  ██║█████╗  █████╗  ██████╔╝    ██║     █████╗  ███████║  ║
║     ██║  ██║██╔══╝  ██╔══╝  ██╔═══╝     ██║     ██╔══╝  ██╔══██║  ║
║     ██████╔╝███████╗███████╗██║         ███████╗███████╗██║  ██║  ║
║     ╚═════╝ ╚══════╝╚══════╝╚═╝         ╚══════╝╚══════╝╚═╝  ╚═╝  ║
║                                                                   ║
║           Interactive Deep Learning Course Visualizations          ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    weeks = [
        ("1", "Week 1-2: Foundations", "week_01_02_visualizer.py",
         ["Neuron/Perceptron", "Forward Pass", "Gradient Descent",
          "Backpropagation", "Activation Functions", "Loss Functions"]),

        ("2", "Week 3-4: CNNs & Training", "week_03_04_visualizer.py",
         ["Convolution Operation", "Padding & Stride", "Pooling",
          "CNN Architecture", "Batch Normalization", "Dropout", "Famous Architectures"]),

        ("3", "Week 5-6: SSL & RNNs", "week_05_06_visualizer.py",
         ["Self-Supervised Learning", "Contrastive Learning", "RNN Basics",
          "Vanishing Gradients", "LSTM", "GRU", "Seq2Seq"]),

        ("4", "Week 7: Attention & Transformers", "week_07_visualizer.py",
         ["Attention Intuition", "Self-Attention", "Multi-Head Attention",
          "Transformer Architecture", "Causal Masking", "Positional Encoding", "KV Cache"]),

        ("5", "Week 8-10: Advanced Topics", "week_08_10_visualizer.py",
         ["Generative vs Discriminative", "VAE", "GAN", "Diffusion Models",
          "Interpretability", "Ethics & Bias", "Deployment"]),

        ("6", "Week 11: Large Language Models", "week_11_visualizer.py",
         ["Tokenization (BPE)", "Pre-training", "Scaling Laws",
          "LoRA Fine-tuning", "RLHF", "Prompting Techniques", "RAG"]),
    ]

    while True:
        print_banner()
        print("Select a week to explore:\n")

        for num, title, _, topics in weeks:
            print(f"    [{num}] {title}")
            for topic in topics[:3]:
                print(f"        - {topic}")
            if len(topics) > 3:
                print(f"        ... and {len(topics)-3} more")
            print()

        print("    [Q] Quit\n")

        choice = input("Enter choice: ").strip().upper()

        if choice == 'Q':
            print("\nHappy learning!")
            break

        # Find matching week
        week_found = None
        for num, title, script, _ in weeks:
            if choice == num:
                week_found = (title, script)
                break

        if week_found:
            title, script = week_found
            script_path = os.path.join(script_dir, script)

            if os.path.exists(script_path):
                print(f"\nLaunching {title}...")
                try:
                    module = load_module(script_path)
                    module.main()
                except Exception as e:
                    print(f"\nError running visualizer: {e}")
                    input("Press Enter to continue...")
            else:
                print(f"\nError: {script} not found!")
                input("Press Enter to continue...")
        else:
            print("\nInvalid choice. Please try again.")
            import time
            time.sleep(1)

if __name__ == "__main__":
    main()
