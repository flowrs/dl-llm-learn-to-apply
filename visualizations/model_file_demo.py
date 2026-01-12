"""
Demonstration: What do ML models actually look like as files?
"""

import struct
import json
import os

def demo_model_contents():
    print("""
============================================================
          WHAT IS A MODEL FILE? (Software Perspective)
============================================================

A model is NOT a program - it's a DATA FILE containing:

    +--------------------------------------------------+
    |                   MODEL FILE                     |
    +--------------------------------------------------+
    |                                                  |
    |   WEIGHTS (99% of the file)                      |
    |   +-----------------------------------------+    |
    |   | layer1.weight: [0.023, -0.891, 0.445,   |    |
    |   |                 0.112, -0.667, 0.234,   |    |
    |   |                 ... millions more ...]  |    |
    |   | layer1.bias:   [0.001, -0.003, 0.002]   |    |
    |   | layer2.weight: [0.156, 0.892, -0.234,   |    |
    |   |                 ... millions more ...]  |    |
    |   | ...                                     |    |
    |   +-----------------------------------------+    |
    |                                                  |
    |   METADATA (small)                               |
    |   +-----------------------------------------+    |
    |   | format_version: 2                       |    |
    |   | pytorch_version: "2.0"                  |    |
    |   +-----------------------------------------+    |
    |                                                  |
    +--------------------------------------------------+

The MODEL FILE is useless without:
    1. Architecture code (Python) - defines layer structure
    2. Framework (PyTorch/TensorFlow) - runs the computation
""")
    input("\nPress Enter to see common file formats...")

    print("""
============================================================
                    COMMON MODEL FORMATS
============================================================

FORMAT          EXTENSION       USED BY
------          ---------       -------
PyTorch         .pt, .pth       PyTorch models
SafeTensors     .safetensors    HuggingFace (safer)
GGUF            .gguf           llama.cpp, Ollama
HDF5            .h5             Keras/TensorFlow
Checkpoint      .ckpt           TensorFlow
ONNX            .onnx           Cross-platform
Pickle          .pkl            Legacy Python

FILE SIZES (typical):
----------------------------------------------
Model Type              Parameters      Size
----------------------------------------------
Small CNN (MNIST)       ~100K           ~400 KB
ResNet-50               ~25M            ~100 MB
BERT-base               ~110M           ~440 MB
GPT-2                   ~1.5B           ~6 GB
Llama-2-7B              ~7B             ~14 GB (fp16)
Llama-2-70B             ~70B            ~140 GB (fp16)
GPT-4 (estimated)       ~1.7T           ~3.4 TB (fp16)
----------------------------------------------

Why so big? Each parameter = 4 bytes (fp32) or 2 bytes (fp16)
    7 billion params x 2 bytes = 14 GB
""")
    input("\nPress Enter to see inside a PyTorch model...")

    print("""
============================================================
              INSIDE A PYTORCH MODEL (.pt file)
============================================================

When you save a PyTorch model:

    torch.save(model.state_dict(), "model.pt")

It creates a file containing a Python dictionary:

    {
        "layer1.weight": tensor([[ 0.0231, -0.8912,  0.4451, ...],
                                 [ 0.1123, -0.6671,  0.2341, ...],
                                 ...]),  # Shape: [256, 784]

        "layer1.bias":   tensor([0.0012, -0.0034, 0.0021, ...]),
                                 # Shape: [256]

        "layer2.weight": tensor([[ 0.1561,  0.8923, -0.2341, ...],
                                 ...]),  # Shape: [128, 256]

        "layer2.bias":   tensor([...]),  # Shape: [128]

        "output.weight": tensor([...]),  # Shape: [10, 128]
        "output.bias":   tensor([...]),  # Shape: [10]
    }

This is serialized using Python's pickle + compression.
""")
    input("\nPress Enter to see a real example...")

    # Create a tiny "model" to demonstrate
    print("""
============================================================
                 REAL EXAMPLE: TINY MODEL
============================================================

Let's create a tiny 3-layer network and look at its file:
""")

    # Simulate model weights
    import random
    random.seed(42)

    # Tiny model: 4 inputs -> 3 hidden -> 2 outputs
    weights = {
        "layer1.weight": [[random.gauss(0, 0.5) for _ in range(4)] for _ in range(3)],
        "layer1.bias": [random.gauss(0, 0.1) for _ in range(3)],
        "layer2.weight": [[random.gauss(0, 0.5) for _ in range(3)] for _ in range(2)],
        "layer2.bias": [random.gauss(0, 0.1) for _ in range(2)],
    }

    print("Model architecture: 4 inputs -> 3 hidden -> 2 outputs")
    print("\nWeights dictionary contents:")
    print("-" * 50)

    total_params = 0
    for name, values in weights.items():
        if isinstance(values[0], list):
            shape = f"[{len(values)}, {len(values[0])}]"
            params = len(values) * len(values[0])
        else:
            shape = f"[{len(values)}]"
            params = len(values)
        total_params += params
        print(f"  {name:20s} shape: {shape:10s} ({params} params)")

    print("-" * 50)
    print(f"  Total parameters: {total_params}")
    print(f"  File size (fp32): {total_params * 4} bytes")

    print("\nActual weight values (first layer):")
    print("-" * 50)
    print("layer1.weight =")
    for i, row in enumerate(weights["layer1.weight"]):
        print(f"  [{', '.join(f'{v:7.4f}' for v in row)}]")
    print(f"\nlayer1.bias = [{', '.join(f'{v:7.4f}' for v in weights['layer1.bias'])}]")

    input("\nPress Enter to see the binary representation...")

    print("""
============================================================
                 BINARY FILE STRUCTURE
============================================================

When saved to disk, weights become raw bytes:

FLOAT32 ENCODING (IEEE 754):
    Value: 0.4503
    Binary: 01111110 11100110 10101110 00011110
    Hex: 3E E6 AE 1E
    Bytes: [0x3E, 0xE6, 0xAE, 0x1E]

A model file is essentially:

    Offset    Content
    ------    -------
    0x0000    [Header/Magic bytes]
    0x0010    [Metadata length]
    0x0014    [Metadata: layer names, shapes, dtypes]
    0x0100    [Weight data starts]
    0x0100    3E E6 AE 1E  <- layer1.weight[0][0] = 0.4503
    0x0104    BF 23 D7 0A  <- layer1.weight[0][1] = -0.6401
    0x0108    3E 8F 5C 29  <- layer1.weight[0][2] = 0.2800
    ...       ... millions/billions more floats ...
    EOF

HexDump of a tiny weight file:
""")

    # Create actual binary representation
    demo_file = os.path.join(os.path.dirname(__file__), "demo_weights.bin")
    with open(demo_file, 'wb') as f:
        # Write header
        f.write(b'WEIGHTS\x00')  # Magic bytes
        f.write(struct.pack('I', total_params))  # Number of params
        # Write weights as float32
        for name, values in weights.items():
            if isinstance(values[0], list):
                for row in values:
                    for v in row:
                        f.write(struct.pack('f', v))
            else:
                for v in values:
                    f.write(struct.pack('f', v))

    # Read and display hex
    with open(demo_file, 'rb') as f:
        data = f.read()

    print("    Offset  | Hex                                      | ASCII")
    print("    --------|------------------------------------------|----------")
    for i in range(0, min(128, len(data)), 16):
        hex_str = ' '.join(f'{b:02X}' for b in data[i:i+16])
        ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in data[i:i+16])
        print(f"    {i:04X}    | {hex_str:40s} | {ascii_str}")

    file_size = os.path.getsize(demo_file)
    print(f"\n    File size: {file_size} bytes")
    print(f"    Created: {demo_file}")

    # Cleanup
    os.remove(demo_file)

    input("\nPress Enter to see HuggingFace model structure...")

    print("""
============================================================
            HUGGINGFACE MODEL STRUCTURE (LLMs)
============================================================

When you download a model from HuggingFace, you get:

    my-model/
    |
    +-- config.json           <- Architecture definition (TEXT)
    |   {
    |     "model_type": "llama",
    |     "hidden_size": 4096,
    |     "num_hidden_layers": 32,
    |     "num_attention_heads": 32,
    |     "vocab_size": 32000,
    |     ...
    |   }
    |
    +-- model.safetensors     <- Weights (BINARY, huge!)
    |   or model-00001-of-00003.safetensors (sharded)
    |   or pytorch_model.bin
    |
    +-- tokenizer.json        <- Vocabulary (TEXT)
    |   {
    |     "model": {"vocab": {"hello": 1234, ...}},
    |     "merges": ["h e", "he llo", ...]
    |   }
    |
    +-- tokenizer_config.json <- Tokenizer settings (TEXT)
    |
    +-- generation_config.json <- Generation defaults (TEXT)

The BINARY weight file is 99%+ of the total size.
Everything else is small JSON/text config files.
""")
    input("\nPress Enter to see how models are loaded...")

    print("""
============================================================
              HOW MODELS ARE LOADED AND RUN
============================================================

STEP 1: Load architecture (from code or config)
-----------------------------------------------
    # Python code defines the structure:

    class MyModel(nn.Module):
        def __init__(self):
            self.layer1 = nn.Linear(784, 256)
            self.layer2 = nn.Linear(256, 128)
            self.output = nn.Linear(128, 10)

        def forward(self, x):
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
            return self.output(x)

    model = MyModel()  # Empty model (random weights)

STEP 2: Load weights from file
-----------------------------------------------
    # Load saved weights into the model:

    state_dict = torch.load("model.pt")
    model.load_state_dict(state_dict)

    # Now model has trained weights!

STEP 3: Run inference
-----------------------------------------------
    # Feed data through the model:

    input_data = torch.tensor([...])
    output = model(input_data)  # Matrix multiplications!

THE KEY INSIGHT:
+----------------------------------------------------------+
|                                                          |
|   Model File (.pt)     +    Architecture Code    =  AI   |
|   (just numbers)            (defines structure)          |
|                                                          |
|   Like a brain's        +   Like a brain's       =  Mind |
|   synaptic weights          physical structure           |
|                                                          |
+----------------------------------------------------------+
""")
    input("\nPress Enter to see GGUF format (for local LLMs)...")

    print("""
============================================================
                GGUF FORMAT (llama.cpp/Ollama)
============================================================

GGUF = GPT-Generated Unified Format
Used by: llama.cpp, Ollama, LM Studio, etc.

GGUF is special because it's SELF-CONTAINED:

    +------------------------------------------------+
    |                  GGUF FILE                     |
    +------------------------------------------------+
    |  HEADER                                        |
    |    - Magic: "GGUF"                             |
    |    - Version: 3                                |
    |    - Tensor count                              |
    |    - Metadata count                            |
    +------------------------------------------------+
    |  METADATA (architecture is INSIDE the file!)   |
    |    - model.type = "llama"                      |
    |    - llama.context_length = 4096               |
    |    - llama.embedding_length = 4096             |
    |    - llama.block_count = 32                    |
    |    - llama.attention.head_count = 32           |
    |    - tokenizer.model = [BPE vocab...]          |
    +------------------------------------------------+
    |  TENSOR INFO                                   |
    |    - "token_embd.weight" @ offset X, Q4_K_M    |
    |    - "blk.0.attn_q.weight" @ offset Y, Q4_K_M  |
    |    - ...                                       |
    +------------------------------------------------+
    |  TENSOR DATA (bulk of the file)                |
    |    [quantized weight bytes...]                 |
    +------------------------------------------------+

QUANTIZATION (smaller files, faster inference):
-----------------------------------------------
    Format      Bits/Weight    7B Model Size
    ------      -----------    -------------
    FP32        32 bits        28 GB
    FP16        16 bits        14 GB
    Q8_0        8 bits         7 GB
    Q4_K_M      ~4.5 bits      4 GB    <- Common choice
    Q2_K        ~2.5 bits      2.5 GB  <- Lower quality

That's why you see files like:
    llama-2-7b-chat.Q4_K_M.gguf  (4.08 GB)
""")

    print("""
============================================================
                        SUMMARY
============================================================

Q: What IS a model file?
A: A binary file containing billions of floating-point numbers
   (weights) that were learned during training.

Q: Is it a program?
A: NO. It's data. You need separate code to use it.

Q: What's inside?
A: 99% weights (numbers), 1% metadata (layer names, shapes).

Q: How big are they?
A: Tiny CNN: ~1 MB
   ResNet-50: ~100 MB
   GPT-2: ~6 GB
   Llama-7B: ~14 GB (fp16) or ~4 GB (quantized)
   Llama-70B: ~140 GB (fp16)

Q: How do you run them?
A: Load the file + architecture code into a framework
   (PyTorch, TensorFlow, llama.cpp) and feed it data.

============================================================
""")

if __name__ == "__main__":
    demo_model_contents()
