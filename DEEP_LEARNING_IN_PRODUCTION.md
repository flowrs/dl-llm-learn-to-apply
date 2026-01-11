# Deep Learning in Production: From MVP to Scale

## How Real Systems Are Built, Evolved, and Operated

This document takes you inside actual production ML systems. We'll trace the journey
from first prototype to mature production system, showing architecture decisions,
data flows, and the ongoing work that keeps these systems running.

```
THE PRODUCTION ML REALITY
=========================

What students think ML is:

    Data ──► Train Model ──► Deploy ──► Done!
                                         ↑
                                    (one time)

What production ML actually is:

    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │   ┌─────┐    ┌───────┐    ┌──────┐    ┌────────┐    ┌──────┐   │
    │   │Data │───►│Feature│───►│Train │───►│Validate│───►│Deploy│   │
    │   │Pipe │    │Engine │    │      │    │        │    │      │   │
    │   └─────┘    └───────┘    └──────┘    └────────┘    └──────┘   │
    │      ▲                                                   │      │
    │      │         ┌──────────┐    ┌──────────┐              │      │
    │      │         │ Monitor  │◄───│ Serve    │◄─────────────┘      │
    │      │         └──────────┘    └──────────┘                     │
    │      │              │                                           │
    │      └──────────────┴───────── (continuous loop) ───────────────┘
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

    Code: ~10% of the work
    Data, infrastructure, monitoring: ~90% of the work
```

---

# Application 1: E-Commerce Visual Search & Recommendations

## The Business Problem

Users want to find products by image ("I saw this dress, find similar ones")
and get personalized recommendations based on visual preferences.

```
USER JOURNEY
============

User uploads photo          System finds similar products
      │                              │
      ▼                              ▼
┌───────────┐                 ┌─────────────────────────┐
│           │                 │ ┌───┐ ┌───┐ ┌───┐ ┌───┐ │
│    📷     │  ──────────►    │ │👗 │ │👗 │ │👗 │ │👗 │ │
│           │                 │ └───┘ └───┘ └───┘ └───┘ │
└───────────┘                 │  $49   $65   $52   $78  │
                              └─────────────────────────┘
```

---

## Phase 1: The MVP (Week 1-4)

**Goal**: Prove the concept works with minimal infrastructure.

```
MVP ARCHITECTURE
================

                    ┌─────────────────────────────────┐
                    │         MONOLITH API            │
                    │         (Flask/FastAPI)         │
                    │                                 │
User ──► Nginx ──►  │  ┌─────────────────────────┐   │
                    │  │ /search endpoint         │   │
                    │  │                          │   │
                    │  │  1. Load image           │   │
                    │  │  2. ResNet extract       │   │
                    │  │  3. Brute-force search   │   │
                    │  │  4. Return top-10        │   │
                    │  └─────────────────────────┘   │
                    │                                 │
                    │  ┌─────────────────────────┐   │
                    │  │ Product embeddings      │   │
                    │  │ (loaded in memory)      │   │
                    │  │ ~50K products = 200MB   │   │
                    │  └─────────────────────────┘   │
                    └─────────────────────────────────┘

Tech stack:
- Python + FastAPI
- PyTorch + torchvision (pretrained ResNet50)
- NumPy for similarity search
- Single EC2 instance or laptop
```

**MVP Code Structure:**

```python
# app.py - The entire MVP in one file
from fastapi import FastAPI, UploadFile
import torch
from torchvision import models, transforms
import numpy as np
from PIL import Image

app = FastAPI()

# Load model once at startup
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classifier
model.eval()

# Load product embeddings (pre-computed offline)
product_embeddings = np.load("embeddings.npy")  # Shape: (50000, 2048)
product_ids = np.load("product_ids.npy")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

@app.post("/search")
async def visual_search(image: UploadFile):
    # 1. Extract embedding
    img = Image.open(image.file).convert("RGB")
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        embedding = model(x).squeeze().numpy()

    # 2. Brute-force similarity search
    similarities = np.dot(product_embeddings, embedding)
    top_indices = np.argsort(similarities)[-10:][::-1]

    # 3. Return results
    return {"products": product_ids[top_indices].tolist()}
```

**What Works:**
- Functional visual search in days
- Proves concept to stakeholders
- Handles ~10 requests/second

**What Doesn't Scale:**
- Brute-force search is O(n) - gets slow with millions of products
- Single point of failure
- No monitoring, no versioning
- Cold start takes 30+ seconds

---

## Phase 2: First Production System (Month 2-4)

**Trigger**: MVP proved value, now need to handle real traffic.

```
PRODUCTION V1 ARCHITECTURE
==========================

                                    ┌─────────────────┐
                                    │   CDN (images)  │
                                    │   CloudFront    │
                                    └────────┬────────┘
                                             │
         ┌───────────────────────────────────┼───────────────┐
         │                                   │               │
         │  ┌────────────────────────────────┼────────┐      │
         │  │           KUBERNETES CLUSTER   │        │      │
         │  │                                ▼        │      │
         │  │  ┌──────────────────────────────────┐   │      │
User ────┼──┼─►│        Load Balancer (ALB)       │   │      │
         │  │  └──────────────────────────────────┘   │      │
         │  │              │                          │      │
         │  │     ┌────────┴────────┐                 │      │
         │  │     ▼                 ▼                 │      │
         │  │  ┌──────┐         ┌──────┐              │      │
         │  │  │ API  │         │ API  │  (3 replicas)│      │
         │  │  │ Pod  │         │ Pod  │              │      │
         │  │  └──┬───┘         └──┬───┘              │      │
         │  │     │                │                  │      │
         │  │     └───────┬────────┘                  │      │
         │  │             ▼                           │      │
         │  │  ┌─────────────────────┐                │      │
         │  │  │   Vector Database   │ (Milvus/Pinecone)     │
         │  │  │   1M+ embeddings    │                │      │
         │  │  │   ANN search <10ms  │                │      │
         │  │  └─────────────────────┘                │      │
         │  │             │                           │      │
         │  └─────────────┼───────────────────────────┘      │
         │                │                                  │
         │                ▼                                  │
         │  ┌─────────────────────────┐                      │
         │  │      Redis Cache        │                      │
         │  │  (popular queries)      │                      │
         │  └─────────────────────────┘                      │
         │                                                   │
         └───────────────────────────────────────────────────┘

New components:
- Vector database for fast ANN search (O(log n) vs O(n))
- Kubernetes for scaling API pods
- Redis for caching frequent queries
- CDN for image delivery
```

**Key Changes from MVP:**

```
WHAT CHANGED AND WHY
====================

1. VECTOR DATABASE (Milvus/Pinecone/Weaviate)

   MVP: Brute-force search
        for each query:
            for each of 1M products:
                compute similarity  → O(n) = 1M operations

   Now: Approximate Nearest Neighbor (ANN)
        HNSW index / IVF index
        O(log n) = ~20 operations

   Latency: 500ms → 5ms


2. SEPARATED MODEL SERVING

   MVP: Model loaded in each API instance
        - 3 replicas = 3 copies of model in memory
        - Cold start = reload model each time

   Now: Dedicated inference service
        - Single model instance
        - GPU utilization
        - TorchServe or Triton


3. CACHING LAYER

   Cache frequent/recent queries:

   Query hash → cached results

   Hit rate typically 30-50% for visual search
   (users often search similar things)
```

**Data Pipeline for Embeddings:**

```
BATCH EMBEDDING PIPELINE
========================

Product Catalog              Embedding Pipeline              Vector DB
    │                              │                            │
    ▼                              ▼                            ▼
┌─────────┐      ┌─────────────────────────────┐      ┌─────────────┐
│New/     │      │                             │      │             │
│Updated  │─────►│  1. Pull from S3            │─────►│   Index     │
│Products │      │  2. Download images         │      │   Update    │
│(daily)  │      │  3. Batch inference (GPU)   │      │             │
└─────────┘      │  4. Upload embeddings       │      └─────────────┘
                 │                             │
                 │  Airflow DAG / Prefect flow │
                 └─────────────────────────────┘

Runs daily at 2 AM:
- Process ~10K new/updated products
- Takes ~30 minutes on single GPU
- Incremental update to index
```

---

## Phase 3: Mature System (Month 6-12)

**Triggers**:
- Need personalization (not just visual similarity)
- A/B testing capabilities
- Real-time model updates
- Multi-region deployment

```
MATURE ARCHITECTURE
===================

                                    ┌─────────────────────────────────────┐
                                    │           GLOBAL CDN               │
                                    └─────────────────┬───────────────────┘
                                                      │
                    ┌─────────────────────────────────┼─────────────────────────────────┐
                    │                                 │                                 │
                    ▼                                 ▼                                 ▼
            ┌───────────────┐                 ┌───────────────┐                 ┌───────────────┐
            │   US-EAST     │                 │   EU-WEST     │                 │   APAC        │
            │   Region      │                 │   Region      │                 │   Region      │
            └───────┬───────┘                 └───────┬───────┘                 └───────┬───────┘
                    │                                 │                                 │
         ┌──────────┴──────────┐                      │                                 │
         │                     │                      │                                 │
         ▼                     ▼                      ▼                                 ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐              ┌─────────────────┐
│  API Gateway    │   │ Feature Store   │   │  Replicated     │              │ ...             │
│  (Kong/AWS)     │   │ (Feast/Tecton)  │   │  Vector DB      │              │                 │
└────────┬────────┘   └────────┬────────┘   └─────────────────┘              └─────────────────┘
         │                     │
         ▼                     │
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                        INFERENCE LAYER                                          │
│                                                                                                 │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                      │
│   │   Visual    │    │   Ranker    │    │   Personal- │    │   Business  │                      │
│   │   Encoder   │    │   Model     │    │   ization   │    │   Rules     │                      │
│   │   (ResNet)  │    │   (XGBoost) │    │   (RecSys)  │    │   Engine    │                      │
│   └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘                      │
│         │                  │                  │                  │                              │
│         └──────────────────┴──────────────────┴──────────────────┘                              │
│                                        │                                                        │
│                                        ▼                                                        │
│                              ┌─────────────────┐                                                │
│                              │    Ensemble     │                                                │
│                              │    Combiner     │                                                │
│                              └─────────────────┘                                                │
│                                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                        MLOPS LAYER                                              │
│                                                                                                 │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                      │
│   │   Model     │    │   A/B Test  │    │   Feature   │    │   Drift     │                      │
│   │   Registry  │    │   Platform  │    │   Store     │    │   Monitor   │                      │
│   │  (MLflow)   │    │  (custom)   │    │  (Feast)    │    │ (Evidently) │                      │
│   └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘                      │
│                                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘


DATA FLOW FOR A SINGLE REQUEST
==============================

User Query                                                      Response
    │                                                              ▲
    ▼                                                              │
┌───────────┐                                                      │
│ 1. Parse  │ Extract image, user_id, context                     │
└─────┬─────┘                                                      │
      │                                                            │
      ▼                                                            │
┌───────────┐                                                      │
│ 2. Cache  │ ──Hit──► Return cached ─────────────────────────────►│
│   Check   │                                                      │
└─────┬─────┘                                                      │
      │ Miss                                                       │
      ▼                                                            │
┌───────────┐   ┌───────────────────────────────────────┐          │
│ 3. Encode │   │ Feature Store (parallel fetch)        │          │
│   Image   │   │ - User history (last 50 views)        │          │
└─────┬─────┘   │ - User preferences (learned profile)  │          │
      │         │ - Category affinity scores            │          │
      │         └───────────────────┬───────────────────┘          │
      │                             │                              │
      ▼                             ▼                              │
┌─────────────────────────────────────────────────────┐            │
│ 4. Candidate Retrieval                               │            │
│    - Visual similarity: top 1000                     │            │
│    - Personalization: top 500                        │            │
│    - Category boost: top 200                         │            │
│    - Merge & dedupe: ~1500 candidates                │            │
└─────────────────────────────────────────────────────┘            │
                        │                                          │
                        ▼                                          │
┌─────────────────────────────────────────────────────┐            │
│ 5. Ranking                                           │            │
│    Input: 1500 candidates + user features            │            │
│    Model: XGBoost ranker (trained on click data)     │            │
│    Output: ranked list with scores                   │            │
└─────────────────────────────────────────────────────┘            │
                        │                                          │
                        ▼                                          │
┌─────────────────────────────────────────────────────┐            │
│ 6. Business Rules                                    │            │
│    - Boost in-stock items                            │            │
│    - Demote recently viewed                          │            │
│    - Ensure category diversity                       │            │
│    - Apply margin optimization                       │            │
└─────────────────────────────────────────────────────┘            │
                        │                                          │
                        ▼                                          │
┌─────────────────────────────────────────────────────┐            │
│ 7. Return top 20 + log for training                 │────────────►
└─────────────────────────────────────────────────────┘
```

---

## OSS vs Cloud Decision Map

```
OPEN SOURCE vs CLOUD SERVICES
=============================

Component           OSS Option              Cloud Option          When to Use Cloud
─────────           ──────────              ────────────          ─────────────────
Vector DB           Milvus, Weaviate,       Pinecone,             - < 10M vectors
                    Qdrant                   Vertex Matching       - Need managed ops
                                             Engine                - Fast time-to-market

Model Serving       TorchServe,             SageMaker,            - Need autoscaling
                    Triton, BentoML         Vertex AI,            - GPU management pain
                                             Azure ML              - Compliance requirements

Feature Store       Feast                   Tecton, Databricks    - Real-time features
                                             Feature Store         - Large team

Model Registry      MLflow                  Weights & Biases,     - Enterprise features
                                             Neptune               - Collaboration

Orchestration       Airflow, Prefect        AWS Step Functions,   - Simple workflows
                                             Managed Airflow       - Ops overhead concern

Monitoring          Prometheus + Grafana,   Datadog, New Relic,   - Correlation features
                    Evidently               Arize                 - Out-of-box ML metrics


DECISION FRAMEWORK:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Team Size < 5 engineers?                                      │
│        │                                                        │
│        ├── Yes ──► Prefer cloud (less ops burden)               │
│        │                                                        │
│        └── No ──► Consider OSS (more control, lower cost)       │
│                                                                 │
│   Latency critical (< 50ms p99)?                                │
│        │                                                        │
│        ├── Yes ──► Self-hosted (control over deployment)        │
│        │                                                        │
│        └── No ──► Cloud OK                                      │
│                                                                 │
│   Data sensitivity / compliance?                                │
│        │                                                        │
│        ├── High ──► Self-hosted or private cloud                │
│        │                                                        │
│        └── Low ──► Public cloud OK                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘


COST COMPARISON (1M products, 100 QPS):

                        OSS (Self-Hosted)         Cloud Managed
                        ─────────────────         ─────────────
Vector DB               $500/mo (3 nodes)         $800/mo (Pinecone)
GPU Inference           $1500/mo (2 A10)          $2000/mo (SageMaker)
Orchestration           $200/mo (Airflow)         $400/mo (MWAA)
Monitoring              $100/mo (Prometheus)      $500/mo (Datadog)
                        ─────────────────         ─────────────
Total                   $2300/mo                  $3700/mo
Engineering hours       40 hrs/mo ops             10 hrs/mo ops

True cost includes engineer time!
At $150/hr, 30 extra hours = $4500 "hidden" cost for OSS
```

---

## Ongoing Operations

```
WEEKLY OPERATIONS WORKFLOW
==========================

┌─────────────────────────────────────────────────────────────────────────┐
│                           MONDAY                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Review Weekend Metrics                                           │   │
│  │ - Latency p50/p95/p99                                            │   │
│  │ - Error rates by endpoint                                        │   │
│  │ - Model prediction distribution                                  │   │
│  │ - Cache hit rates                                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         TUESDAY-WEDNESDAY                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Model Development                                                │   │
│  │ - Experiment with new architectures                              │   │
│  │ - Feature engineering                                            │   │
│  │ - Retrain on latest data                                         │   │
│  │ - Offline evaluation (A/B simulation)                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           THURSDAY                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Model Validation & Staging                                       │   │
│  │ - Shadow mode deployment                                         │   │
│  │ - Compare predictions: new vs current                            │   │
│  │ - Check for regressions on key segments                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            FRIDAY                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Gradual Rollout                                                  │   │
│  │ - 5% traffic to new model                                        │   │
│  │ - Monitor business metrics (CTR, conversion)                     │   │
│  │ - If good: increase to 20% over weekend                          │   │
│  │ - Full rollout next week if metrics hold                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘


AUTOMATED DAILY JOBS
====================

2:00 AM  ─► Embedding pipeline for new products
3:00 AM  ─► Index update and replication
4:00 AM  ─► Data quality checks
5:00 AM  ─► Training data preparation (click logs)
6:00 AM  ─► Drift detection reports
8:00 AM  ─► Daily metrics email to stakeholders
```

---

# Application 2: Manufacturing Defect Detection

## The Business Problem

Detect defects on production line at 100+ items/minute with <0.1% false negatives
(can't let bad products ship).

```
PRODUCTION LINE CONTEXT
=======================

    Camera 1        Camera 2        Camera 3
        │               │               │
        ▼               ▼               ▼
   ┌────────┐      ┌────────┐      ┌────────┐
   │        │      │        │      │        │
   │  Top   │      │ Side   │      │ Bottom │
   │  View  │      │ View   │      │ View   │
   │        │      │        │      │        │
   └────────┘      └────────┘      └────────┘
        │               │               │
        └───────────────┼───────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │   Inference     │
              │   Edge Device   │
              │   (< 50ms)      │
              └─────────────────┘
                        │
            ┌───────────┴───────────┐
            │                       │
            ▼                       ▼
      ┌──────────┐           ┌──────────┐
      │   PASS   │           │  REJECT  │
      │   ──►    │           │   ──► 🗑 │
      └──────────┘           └──────────┘
```

---

## Phase 1: MVP with Edge Constraints

```
EDGE MVP ARCHITECTURE
=====================

┌─────────────────────────────────────────────────────────────┐
│                      EDGE DEVICE                            │
│                   (Jetson Xavier NX)                        │
│                                                             │
│   ┌──────────────────────────────────────────────────┐      │
│   │                 GStreamer Pipeline                │      │
│   │                                                   │      │
│   │  Camera ──► Decode ──► Resize ──► Model ──► Logic │      │
│   │    │                              │         │     │      │
│   │    │        (hardware             │         │     │      │
│   │    │         accelerated)         │         ▼     │      │
│   │    │                              │    ┌────────┐ │      │
│   │    │                              │    │ PLC    │ │      │
│   │    │                              │    │ Signal │ │      │
│   │    │                              │    └────────┘ │      │
│   │    │                              │         │     │      │
│   │    ▼                              ▼         ▼     │      │
│   │  30 FPS                       TensorRT    Reject  │      │
│   │                               INT8        gate    │      │
│   └──────────────────────────────────────────────────┘      │
│                                                             │
│   Local storage: Last 1000 images + predictions             │
│   Sync to cloud: Every 15 minutes (batch upload)            │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Constraints:
- 50ms latency budget (20 FPS minimum)
- Must work offline (factory floor network issues)
- 10W power budget
- Model must fit in 4GB memory

Model choice:
- MobileNetV3 + custom head
- TensorRT INT8 quantization
- Input: 224x224
- Inference: ~15ms on Xavier NX
```

**Model Optimization for Edge:**

```python
# Convert PyTorch model to TensorRT

import torch
import tensorrt as trt

# 1. Export to ONNX
model = MobileNetV3Defect()
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "defect_model.onnx",
    opset_version=13,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
)

# 2. Build TensorRT engine with INT8 calibration
# (Using trtexec command line)
"""
trtexec --onnx=defect_model.onnx \
        --saveEngine=defect_model.trt \
        --int8 \
        --calib=calibration_images/ \
        --workspace=1024
"""

# 3. Inference code on edge device
import pycuda.driver as cuda
import pycuda.autoinit

class TRTInference:
    def __init__(self, engine_path):
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(trt.Logger()).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        # Allocate buffers...

    def infer(self, image):
        # Copy to GPU, run, copy back
        # Returns: (is_defect, confidence, defect_type)
        pass
```

---

## Phase 2: Cloud Backend + Active Learning

```
HYBRID ARCHITECTURE
===================

FACTORY FLOOR                           CLOUD
─────────────                           ─────

┌──────────────────┐              ┌─────────────────────────────────┐
│   Edge Device 1  │              │                                 │
│   ┌────────────┐ │    Sync      │   ┌───────────────────────────┐ │
│   │ Inference  │ │───(15min)───►│   │      Data Lake (S3)       │ │
│   │ + Logging  │ │              │   │  - All images             │ │
│   └────────────┘ │              │   │  - Predictions            │ │
├──────────────────┤              │   │  - Operator corrections   │ │
│   Edge Device 2  │              │   └───────────────────────────┘ │
│   ┌────────────┐ │              │               │                 │
│   │ Inference  │ │───(15min)───►│               ▼                 │
│   │ + Logging  │ │              │   ┌───────────────────────────┐ │
│   └────────────┘ │              │   │    Active Learning        │ │
├──────────────────┤              │   │    Pipeline               │ │
│   ...            │              │   │                           │ │
├──────────────────┤              │   │  1. Find uncertain preds  │ │
│   Edge Device N  │              │   │  2. Find misclassified    │ │
└──────────────────┘              │   │  3. Find novel defects    │ │
        ▲                         │   │  4. Queue for labeling    │ │
        │                         │   └───────────────────────────┘ │
        │                         │               │                 │
        │                         │               ▼                 │
        │                         │   ┌───────────────────────────┐ │
        │   Model                 │   │    Labeling Interface     │ │
        │   Update                │   │    (Label Studio)         │ │
        │   (weekly)              │   │                           │ │
        │                         │   │    Quality team reviews   │ │
        │                         │   │    uncertain cases        │ │
        │                         │   └───────────────────────────┘ │
        │                         │               │                 │
        │                         │               ▼                 │
        │                         │   ┌───────────────────────────┐ │
        │                         │   │    Training Pipeline      │ │
        │                         │   │                           │ │
        │                         │   │  - Automated retraining   │ │
        │                         │   │  - Validation on holdout  │ │
        │◄────────────────────────│   │  - A/B test on 1 device   │ │
                                  │   │  - Gradual rollout        │ │
                                  │   └───────────────────────────┘ │
                                  │                                 │
                                  └─────────────────────────────────┘


ACTIVE LEARNING STRATEGY
========================

Model predictions:

    Confidence
        │
    1.0 │  ●●●●●●●●●●●●●●●●●●●●●  High confidence (ignore)
        │
    0.8 │  ●●●●●  Uncertain zone (label these!)
        │
    0.5 │  ●●
        │
    0.2 │  ●●●●●●●  Uncertain zone (label these!)
        │
    0.0 │  ●●●●●●●●●●●●●●●●●●●●●  High confidence (ignore)
        │
        └──────────────────────────────────────────

Priority for labeling:
1. Predictions near decision boundary (0.4-0.6)
2. Operator overrides (model wrong)
3. New defect patterns (anomaly detection)
4. Random sample (maintain distribution)

Weekly labeling budget: ~500 images
Result: Model improves continuously without massive labeling effort
```

---

## Phase 3: Multi-Factory Deployment

```
MULTI-SITE ARCHITECTURE
=======================

                              ┌─────────────────────────────────────┐
                              │           CENTRAL CLOUD             │
                              │                                     │
                              │  ┌─────────────────────────────┐    │
                              │  │     Global Model Registry   │    │
                              │  │     - Base model            │    │
                              │  │     - Factory-specific      │    │
                              │  │       fine-tuned models     │    │
                              │  └─────────────────────────────┘    │
                              │                                     │
                              │  ┌─────────────────────────────┐    │
                              │  │     Federated Learning      │    │
                              │  │     Coordinator             │    │
                              │  │     - Aggregate gradients   │    │
                              │  │     - Privacy preserved     │    │
                              │  └─────────────────────────────┘    │
                              │                                     │
                              │  ┌─────────────────────────────┐    │
                              │  │     Central Dashboard       │    │
                              │  │     - All factories view    │    │
                              │  │     - Cross-site analytics  │    │
                              │  └─────────────────────────────┘    │
                              │                                     │
                              └──────────────┬──────────────────────┘
                                             │
              ┌──────────────────────────────┼──────────────────────────────┐
              │                              │                              │
              ▼                              ▼                              ▼
    ┌──────────────────┐           ┌──────────────────┐           ┌──────────────────┐
    │   FACTORY A      │           │   FACTORY B      │           │   FACTORY C      │
    │   (Detroit)      │           │   (Shanghai)     │           │   (Munich)       │
    │                  │           │                  │           │                  │
    │ ┌──────────────┐ │           │ ┌──────────────┐ │           │ ┌──────────────┐ │
    │ │ Local        │ │           │ │ Local        │ │           │ │ Local        │ │
    │ │ Edge Cluster │ │           │ │ Edge Cluster │ │           │ │ Edge Cluster │ │
    │ │ (5 lines)    │ │           │ │ (8 lines)    │ │           │ │ (3 lines)    │ │
    │ └──────────────┘ │           │ └──────────────┘ │           │ └──────────────┘ │
    │                  │           │                  │           │                  │
    │ ┌──────────────┐ │           │ ┌──────────────┐ │           │ ┌──────────────┐ │
    │ │ Local        │ │           │ │ Local        │ │           │ │ Local        │ │
    │ │ Training     │ │           │ │ Training     │ │           │ │ Training     │ │
    │ │ (fine-tune)  │ │           │ │ (fine-tune)  │ │           │ │ (fine-tune)  │ │
    │ └──────────────┘ │           │ └──────────────┘ │           │ └──────────────┘ │
    │                  │           │                  │           │                  │
    └──────────────────┘           └──────────────────┘           └──────────────────┘

    Different products,            Different lighting,            Different cameras,
    different defects              local regulations              local operators


WHY FEDERATED + LOCAL FINE-TUNING:

1. Data privacy: Manufacturing data can't leave factory
2. Bandwidth: Terabytes of images daily, can't upload all
3. Latency: Must work even if cloud connection fails
4. Customization: Each factory has unique conditions

Workflow:
1. Train base model on pooled representative data
2. Deploy to all factories
3. Each factory fine-tunes on local data
4. Gradients (not data) sent to cloud
5. Cloud aggregates and improves base model
6. New base model distributed
```

---

# Application 3: Document Processing (Invoice Extraction)

## The Business Problem

Extract structured data from invoices in various formats: vendor, amount, line items,
dates, tax, etc. Handle 50+ different vendor formats.

```
THE DOCUMENT UNDERSTANDING CHALLENGE
====================================

Input (varied formats):                    Output (structured):

┌─────────────────────────┐               {
│ INVOICE                 │                 "vendor": "Acme Corp",
│ Acme Corp               │                 "invoice_no": "INV-2024-001",
│ Invoice #: INV-2024-001 │   ──────►       "date": "2024-01-15",
│ Date: Jan 15, 2024      │                 "total": 1250.00,
│                         │                 "line_items": [
│ Widget A     $500.00    │                   {"desc": "Widget A", "amt": 500},
│ Widget B     $750.00    │                   {"desc": "Widget B", "amt": 750}
│ ─────────────────────   │                 ],
│ Total:     $1,250.00    │                 "tax": 0.00
└─────────────────────────┘               }

Challenge: Every vendor has different format!
```

---

## Phase 1: Template-Based MVP

```
TEMPLATE MATCHING MVP
=====================

For each known vendor, define extraction rules:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   Vendor Template Registry                                              │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │ "acme_corp": {                                                  │   │
│   │   "identifier": "Acme Corp" in first 3 lines,                   │   │
│   │   "invoice_no": regex r"Invoice #:\s*(\S+)",                    │   │
│   │   "date": regex r"Date:\s*(.+)" -> parse_date(),                │   │
│   │   "total": regex r"Total:\s*\$?([\d,]+\.?\d*)",                 │   │
│   │   "line_items": table_between("Item", "Total")                  │   │
│   │ }                                                               │   │
│   ├─────────────────────────────────────────────────────────────────┤   │
│   │ "globex_inc": {                                                 │   │
│   │   "identifier": "Globex" in header,                             │   │
│   │   "invoice_no": regex r"Inv\.?\s*#?\s*(\d+)",                   │   │
│   │   ...                                                           │   │
│   │ }                                                               │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

Pipeline:

PDF ──► OCR ──► Identify Vendor ──► Apply Template ──► Structured Data
         │           │
      Tesseract   String matching
      or Cloud    on first N lines


LIMITATIONS:

1. New vendor = new template (manual work)
2. Vendor changes format = template breaks
3. Poor OCR = regex fails
4. Tables are hard to parse reliably
```

---

## Phase 2: ML-Enhanced Extraction

```
HYBRID ML + RULES ARCHITECTURE
==============================

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │                         Document Ingestion                          │    │
│   │                                                                     │    │
│   │   PDF/Image ──► Cloud Vision API ──► Text + Bounding Boxes         │    │
│   │                 (or Tesseract)        + Confidence scores          │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                   │                                         │
│                                   ▼                                         │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │                      Document Classification                        │    │
│   │                                                                     │    │
│   │   LayoutLMv3 or Donut model                                         │    │
│   │   Input: Document image + OCR text                                  │    │
│   │   Output: Document type (invoice, receipt, PO, etc.)                │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                   │                                         │
│                                   ▼                                         │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │                      Entity Extraction (NER)                        │    │
│   │                                                                     │    │
│   │   LayoutLMv3 fine-tuned for invoice entities                        │    │
│   │   Entities: VENDOR, INVOICE_NO, DATE, TOTAL, LINE_ITEM, TAX         │    │
│   │                                                                     │    │
│   │   Input:  Document image + OCR with positions                       │    │
│   │   Output: Labeled tokens with entity types                          │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                   │                                         │
│                                   ▼                                         │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │                      Table Extraction                               │    │
│   │                                                                     │    │
│   │   Table Transformer or rule-based table detection                   │    │
│   │   Extract line items as structured rows                             │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                   │                                         │
│                                   ▼                                         │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │                      Post-Processing & Validation                   │    │
│   │                                                                     │    │
│   │   - Date normalization (various formats → ISO)                      │    │
│   │   - Currency parsing                                                │    │
│   │   - Cross-validation (line items sum = total?)                      │    │
│   │   - Confidence thresholding                                         │    │
│   │   - Business rules (vendor in approved list?)                       │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                   │                                         │
│                                   ▼                                         │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │                      Human-in-the-Loop                              │    │
│   │                                                                     │    │
│   │   If confidence < threshold:                                        │    │
│   │     Route to human reviewer                                         │    │
│   │     Human corrections feed back to training                         │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


MODEL CHOICE: LayoutLMv3
========================

Why LayoutLMv3 for documents:

Traditional NER:     LayoutLM:

"Total: $500"        "Total: $500" at position (x=100, y=450)
     │                      │
     ▼                      ▼
[BERT sees           [LayoutLM sees text
 only text]           + 2D position
                      + image features]

LayoutLM understands:
- "Total" at bottom right usually = grand total
- Numbers aligned in columns = table
- Large text at top = header/vendor name
```

**Training Pipeline:**

```python
# Fine-tuning LayoutLMv3 for invoice extraction

from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

# Labels for invoice entities
labels = ["O", "B-VENDOR", "I-VENDOR", "B-INVOICE_NO", "I-INVOICE_NO",
          "B-DATE", "I-DATE", "B-TOTAL", "I-TOTAL", "B-LINE_ITEM", "I-LINE_ITEM"]

model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=len(labels)
)
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")

def prepare_example(image, words, boxes, word_labels):
    """Convert annotated document to model input"""
    encoding = processor(
        image,
        words,
        boxes=boxes,
        word_labels=word_labels,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    return encoding

# Training loop
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./invoice-layoutlm",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
```

---

## Phase 3: End-to-End Production System

```
PRODUCTION DOCUMENT PROCESSING SYSTEM
=====================================

                                        ┌─────────────────────────┐
                                        │   Email Inboxes         │
                                        │   Shared Drives         │
                                        │   API Upload            │
                                        │   Scanner Integration   │
                                        └───────────┬─────────────┘
                                                    │
                                                    ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                                    INGESTION LAYER                                    │
│                                                                                       │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐           │
│   │   Email     │    │   Watch     │    │   REST      │    │   Batch     │           │
│   │   Poller    │    │   Folders   │    │   API       │    │   Import    │           │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘           │
│          │                  │                  │                  │                  │
│          └──────────────────┴──────────────────┴──────────────────┘                  │
│                                        │                                             │
│                                        ▼                                             │
│                              ┌─────────────────┐                                     │
│                              │  Message Queue  │                                     │
│                              │  (SQS / Kafka)  │                                     │
│                              └─────────────────┘                                     │
│                                                                                       │
└───────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                                   PROCESSING LAYER                                    │
│                                                                                       │
│   ┌───────────────────────────────────────────────────────────────────────────────┐   │
│   │                          Document Processing Workers                          │   │
│   │                          (Kubernetes / AWS Lambda)                            │   │
│   │                                                                               │   │
│   │    ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │   │
│   │    │ Worker 1 │   │ Worker 2 │   │ Worker 3 │   │ Worker 4 │   │ Worker N │  │   │
│   │    └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘  │   │
│   │         │              │              │              │              │         │   │
│   │         └──────────────┴──────────────┴──────────────┴──────────────┘         │   │
│   │                                       │                                       │   │
│   │                                       ▼                                       │   │
│   │    ┌─────────────────────────────────────────────────────────────────────┐    │   │
│   │    │                        Processing Pipeline                          │    │   │
│   │    │                                                                     │    │   │
│   │    │   1. PDF → Images (pdf2image)                                       │    │   │
│   │    │   2. OCR (Google Vision / AWS Textract / Azure Form Recognizer)     │    │   │
│   │    │   3. Document Classification (LayoutLMv3)                           │    │   │
│   │    │   4. Entity Extraction (LayoutLMv3 fine-tuned)                      │    │   │
│   │    │   5. Table Extraction (Table Transformer)                           │    │   │
│   │    │   6. Post-processing & Validation                                   │    │   │
│   │    │   7. Confidence scoring                                             │    │   │
│   │    └─────────────────────────────────────────────────────────────────────┘    │   │
│   └───────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                       │
└───────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                          ┌─────────────┴─────────────┐
                          │                           │
                          ▼                           ▼
              ┌─────────────────────┐    ┌─────────────────────┐
              │   High Confidence   │    │   Low Confidence    │
              │   (> 0.9)           │    │   (< 0.9)           │
              └──────────┬──────────┘    └──────────┬──────────┘
                         │                          │
                         ▼                          ▼
              ┌─────────────────────┐    ┌─────────────────────┐
              │   Auto-processed    │    │   Human Review      │
              │   → ERP System      │    │   Queue             │
              └─────────────────────┘    └──────────┬──────────┘
                                                    │
                                                    ▼
                                        ┌─────────────────────┐
                                        │   Review Interface  │
                                        │   (Web App)         │
                                        │                     │
                                        │   - Show document   │
                                        │   - Highlight       │
                                        │     extractions     │
                                        │   - Allow edits     │
                                        │   - Approve/Reject  │
                                        └──────────┬──────────┘
                                                   │
                                        ┌──────────┴──────────┐
                                        │                     │
                                        ▼                     ▼
                              ┌─────────────────┐   ┌─────────────────┐
                              │  Corrections    │   │   Approved      │
                              │  → Training     │   │   → ERP System  │
                              │    Data         │   │                 │
                              └─────────────────┘   └─────────────────┘


HUMAN-IN-THE-LOOP METRICS
=========================

Week 1:     70% auto-processed,  30% human review
Week 4:     80% auto-processed,  20% human review
Week 12:    92% auto-processed,   8% human review
Week 24:    96% auto-processed,   4% human review

Human corrections continuously improve the model!
```

---

## OSS vs Cloud for Document Processing

```
DOCUMENT AI: BUILD vs BUY
=========================

                    OSS Stack                    Cloud AI Services
                    ─────────                    ─────────────────
OCR                 Tesseract, EasyOCR           Google Vision, AWS Textract,
                                                 Azure Form Recognizer

Document AI         LayoutLMv3 + custom          AWS Textract, Google
                    training                     Document AI, Azure
                                                 Form Recognizer

Tables              Table Transformer,           Built into cloud services
                    Camelot, Tabula

Cost                $0.001/page (compute)        $0.01-0.05/page

Accuracy            High with training           High out-of-box
                    (~95% after fine-tune)       (~90% zero-shot)

Effort              High (training, tuning)      Low (API call)


DECISION MATRIX:

Volume < 10K docs/month AND standard formats?
    → Use cloud service (not worth building)

Volume > 100K docs/month OR custom formats?
    → Build with OSS (cost savings + customization)

Sensitive documents (medical, legal, financial)?
    → Self-hosted OSS (data sovereignty)


HYBRID APPROACH (Common):

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Use Cloud OCR (best quality)                                  │
│                    │                                            │
│                    ▼                                            │
│   Self-hosted ML models (customization + cost control)          │
│                    │                                            │
│                    ▼                                            │
│   Custom post-processing (business logic)                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

# Application 4: Real-Time Fraud Detection

## The Business Problem

Detect fraudulent transactions in real-time (<100ms) while minimizing false positives
(blocking legitimate customers is costly).

```
THE FRAUD DETECTION CHALLENGE
=============================

Transaction stream: 10,000 transactions/second
Fraud rate: 0.1% (1 in 1000)
Latency budget: 100ms
False positive tolerance: <1% (can't annoy customers)

                    ┌─────────────────────────────────┐
                    │       Fraud Rate: 0.1%          │
                    │                                 │
                    │  ████████████████████████████   │  99.9% legitimate
                    │  ▒                              │   0.1% fraud
                    │                                 │
                    └─────────────────────────────────┘

The needle in the haystack problem!
```

---

## Phase 1: Rule-Based MVP

```
RULE-BASED FRAUD DETECTION
==========================

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   Transaction ──►  Rule Engine  ──►  Decision                           │
│                         │                                               │
│                         ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                                                                 │   │
│   │   Rule 1: Amount > $10,000                    → Review          │   │
│   │   Rule 2: Country ≠ home country              → +50 risk points │   │
│   │   Rule 3: New device                          → +30 risk points │   │
│   │   Rule 4: Time = 2-5 AM                       → +20 risk points │   │
│   │   Rule 5: Velocity > 5 tx/hour                → +40 risk points │   │
│   │   Rule 6: Merchant category = high risk       → +25 risk points │   │
│   │   ...                                                           │   │
│   │   Rule N: ...                                                   │   │
│   │                                                                 │   │
│   │   Total risk > 100 → BLOCK                                      │   │
│   │   Total risk > 50  → REVIEW                                     │   │
│   │   Total risk < 50  → APPROVE                                    │   │
│   │                                                                 │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

Problems:
- Rules are static (fraudsters adapt)
- Rules interact in unexpected ways
- Can't capture complex patterns
- Too many false positives
```

---

## Phase 2: ML-Enhanced Detection

```
ML FRAUD DETECTION ARCHITECTURE
===============================

                    Transaction Event
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                              FEATURE COMPUTATION (< 10ms)                            │
│                                                                                      │
│   ┌────────────────────────┐   ┌────────────────────────┐   ┌────────────────────┐   │
│   │   Transaction Features │   │   User Features        │   │   Context Features │   │
│   │                        │   │   (from Feature Store) │   │                    │   │
│   │   - amount             │   │   - account_age        │   │   - time_of_day    │   │
│   │   - merchant_category  │   │   - avg_transaction    │   │   - day_of_week    │   │
│   │   - is_card_present    │   │   - tx_count_30d       │   │   - is_holiday     │   │
│   │   - entry_mode         │   │   - fraud_history      │   │   - device_fingerp │   │
│   │                        │   │   - typical_location   │   │                    │   │
│   └────────────────────────┘   └────────────────────────┘   └────────────────────┘   │
│              │                          │                          │                │
│              └──────────────────────────┴──────────────────────────┘                │
│                                         │                                           │
│                                         ▼                                           │
│                              ┌─────────────────────┐                                │
│                              │  Feature Vector     │                                │
│                              │  (150 features)     │                                │
│                              └─────────────────────┘                                │
│                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                              MODEL INFERENCE (< 5ms)                                 │
│                                                                                      │
│   ┌─────────────────────────────────────────────────────────────────────────────┐    │
│   │                         ENSEMBLE MODEL                                      │    │
│   │                                                                             │    │
│   │   ┌───────────────┐    ┌───────────────┐    ┌───────────────┐               │    │
│   │   │   XGBoost     │    │  Neural Net   │    │   Isolation   │               │    │
│   │   │   (tabular)   │    │  (embeddings) │    │   Forest      │               │    │
│   │   │               │    │               │    │   (anomaly)   │               │    │
│   │   │   Score: 0.7  │    │   Score: 0.6  │    │   Score: 0.8  │               │    │
│   │   └───────────────┘    └───────────────┘    └───────────────┘               │    │
│   │          │                    │                    │                        │    │
│   │          └────────────────────┴────────────────────┘                        │    │
│   │                              │                                              │    │
│   │                              ▼                                              │    │
│   │                    ┌─────────────────┐                                      │    │
│   │                    │  Meta-Learner   │                                      │    │
│   │                    │  (stacking)     │                                      │    │
│   │                    │                 │                                      │    │
│   │                    │  Final: 0.72    │                                      │    │
│   │                    └─────────────────┘                                      │    │
│   │                                                                             │    │
│   └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                              DECISION LAYER (< 1ms)                                  │
│                                                                                      │
│   ┌─────────────────────────────────────────────────────────────────────────────┐    │
│   │                                                                             │    │
│   │   Score < 0.3    →    APPROVE                                               │    │
│   │   Score 0.3-0.7  →    STEP-UP (2FA, call customer)                          │    │
│   │   Score > 0.7    →    DECLINE                                               │    │
│   │                                                                             │    │
│   │   + Business rules override:                                                │    │
│   │     - VIP customers: higher threshold                                       │    │
│   │     - Known fraud patterns: auto-decline                                    │    │
│   │     - First transaction: step-up always                                     │    │
│   │                                                                             │    │
│   └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────┘


WHY ENSEMBLE?

XGBoost:        Great at tabular patterns, fast inference
Neural Net:     Learns embeddings for categorical variables (merchant, device)
Isolation Forest: Catches novel/unseen fraud patterns (anomaly detection)

Combined: Better than any single model
```

---

## Phase 3: Real-Time Feature Engineering

```
FEATURE STORE ARCHITECTURE
==========================

The hardest part of real-time fraud detection: computing features in <10ms

                    ┌───────────────────────────────────────────────────────────┐
                    │                   FEATURE STORE                           │
                    │                   (Feast / Tecton)                        │
                    │                                                           │
                    │   ┌─────────────────────────────────────────────────┐     │
                    │   │              ONLINE STORE (Redis)               │     │
                    │   │                                                 │     │
                    │   │   user_123: {                                   │     │
                    │   │     tx_count_1h: 3,                             │     │
                    │   │     tx_count_24h: 12,                           │     │
                    │   │     avg_amount_7d: 156.50,                      │     │
                    │   │     last_location: "NYC",                       │     │
                    │   │     device_count_30d: 2,                        │     │
                    │   │     ...                                         │     │
                    │   │   }                                             │     │
                    │   │                                                 │     │
                    │   │   Latency: < 1ms per lookup                     │     │
                    │   └─────────────────────────────────────────────────┘     │
                    │                         ▲                                 │
                    │                         │                                 │
                    │   ┌─────────────────────┴─────────────────────────┐       │
                    │   │           FEATURE COMPUTATION                 │       │
                    │   │                                               │       │
                    │   │   Batch (hourly):     Streaming (real-time):  │       │
                    │   │   - avg_amount_7d     - tx_count_1h           │       │
                    │   │   - merchant_freq     - velocity              │       │
                    │   │   - typical_hours     - session_features      │       │
                    │   │                                               │       │
                    │   │   Spark jobs          Flink / Kafka Streams   │       │
                    │   └───────────────────────────────────────────────┘       │
                    │                         ▲                                 │
                    │                         │                                 │
                    │   ┌─────────────────────┴─────────────────────────┐       │
                    │   │              OFFLINE STORE (S3/BigQuery)      │       │
                    │   │                                               │       │
                    │   │   Historical feature values for training      │       │
                    │   │   Point-in-time correct features              │       │
                    │   │                                               │       │
                    │   └───────────────────────────────────────────────┘       │
                    │                                                           │
                    └───────────────────────────────────────────────────────────┘


STREAMING FEATURE COMPUTATION
=============================

Transaction Stream (Kafka)
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Flink Streaming Job                                │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                                                                     │   │
│   │   // Sliding window aggregations                                    │   │
│   │   SELECT                                                            │   │
│   │     user_id,                                                        │   │
│   │     COUNT(*) OVER (PARTITION BY user_id                             │   │
│   │                     ORDER BY event_time                             │   │
│   │                     RANGE BETWEEN INTERVAL '1' HOUR PRECEDING       │   │
│   │                     AND CURRENT ROW) as tx_count_1h,                │   │
│   │     SUM(amount) OVER (...) as amount_sum_1h,                        │   │
│   │     COUNT(DISTINCT merchant_id) OVER (...) as unique_merchants_1h   │   │
│   │   FROM transactions                                                 │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   Output: Updated features written to Redis in real-time                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Full Production Architecture

```
COMPLETE FRAUD DETECTION SYSTEM
===============================

┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                                 │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                                         │
│   │ Mobile App  │    │   Website   │    │   Partner   │                                         │
│   │             │    │             │    │    API      │                                         │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                                         │
│          │                  │                  │                                                │
│          └──────────────────┴──────────────────┘                                                │
│                             │                                                                   │
│                             ▼                                                                   │
│   ┌─────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                              API GATEWAY                                                │   │
│   │                              (Kong / AWS API Gateway)                                   │   │
│   │                              Rate limiting, authentication                              │   │
│   └─────────────────────────────────────────────────────────────────────────────────────────┘   │
│                             │                                                                   │
│                             ▼                                                                   │
│   ┌─────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                           TRANSACTION SERVICE                                           │   │
│   │                                                                                         │   │
│   │   1. Receive transaction                                                                │   │
│   │   2. Enrich with context                                                                │   │
│   │   3. Call Fraud Service (async)                                                         │   │
│   │   4. Wait for response (timeout: 100ms)                                                 │   │
│   │   5. If timeout: approve + flag for async review                                        │   │
│   │                                                                                         │   │
│   └────────────────────────────────────────────┬────────────────────────────────────────────┘   │
│                                                │                                                │
│                    ┌───────────────────────────┼───────────────────────────┐                    │
│                    │                           │                           │                    │
│                    ▼                           ▼                           ▼                    │
│   ┌─────────────────────────┐    ┌─────────────────────────┐    ┌─────────────────────────┐     │
│   │     FRAUD SERVICE       │    │     FEATURE STORE       │    │     RULES ENGINE        │     │
│   │                         │    │                         │    │                         │     │
│   │   Kubernetes pods       │    │   Redis Cluster         │    │   Business rules        │     │
│   │   Auto-scaling          │    │   User features         │    │   Blacklists            │     │
│   │   Multiple replicas     │    │   Merchant features     │    │   Whitelists            │     │
│   │                         │    │   Device features       │    │   Override logic        │     │
│   │   XGBoost + NN          │    │                         │    │                         │     │
│   │   Ensemble inference    │    │   < 1ms latency         │    │                         │     │
│   │   < 5ms inference       │    │                         │    │                         │     │
│   └─────────────────────────┘    └─────────────────────────┘    └─────────────────────────┘     │
│                    │                           │                           │                    │
│                    └───────────────────────────┴───────────────────────────┘                    │
│                                                │                                                │
│                                                ▼                                                │
│   ┌─────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                           DECISION + RESPONSE                                           │   │
│   │                                                                                         │   │
│   │   Combine:  ML Score + Rules + Business Logic                                           │   │
│   │   Response: APPROVE / DECLINE / STEP_UP                                                 │   │
│   │   Latency:  < 50ms p99                                                                  │   │
│   └─────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                │                                                │
│                    ┌───────────────────────────┴───────────────────────────┐                    │
│                    │                                                       │                    │
│                    ▼                                                       ▼                    │
│   ┌─────────────────────────────────┐                     ┌─────────────────────────────────┐   │
│   │        EVENT STREAM             │                     │        FEEDBACK LOOP            │   │
│   │        (Kafka)                  │                     │                                 │   │
│   │                                 │                     │   Transaction outcome:          │   │
│   │   - All decisions logged        │                     │   - Customer disputed           │   │
│   │   - Latency metrics             │                     │   - Chargeback received         │   │
│   │   - Feature values              │                     │   - False positive reported     │   │
│   │                                 │                     │                                 │   │
│   └─────────────────────────────────┘                     └─────────────────────────────────┘   │
│                    │                                                       │                    │
│                    ▼                                                       ▼                    │
│   ┌─────────────────────────────────┐                     ┌─────────────────────────────────┐   │
│   │        MONITORING               │                     │        TRAINING PIPELINE        │   │
│   │                                 │                     │                                 │   │
│   │   - Score distribution          │                     │   - Daily retraining            │   │
│   │   - Latency percentiles         │                     │   - Label delay: 30-90 days     │   │
│   │   - Fraud rate by segment       │                     │   - Challenger models           │   │
│   │   - Model drift detection       │                     │   - A/B testing                 │   │
│   │                                 │                     │                                 │   │
│   └─────────────────────────────────┘                     └─────────────────────────────────┘   │
│                                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘


LATENCY BREAKDOWN
=================

Component                  p50         p99
─────────                  ───         ───
Network (API Gateway)      2ms         10ms
Feature lookup (Redis)     1ms         3ms
Model inference           3ms         8ms
Rules engine              1ms         2ms
Decision logic            1ms         2ms
Network (response)        2ms         10ms
─────────────────────────────────────────
TOTAL                     10ms        35ms   ✓ Under 100ms budget
```

---

# Summary: Patterns Across All Applications

```
COMMON PRODUCTION PATTERNS
==========================

1. START SIMPLE
───────────────
MVP: Monolith + pretrained model + simple storage
     Get feedback before investing in infrastructure

2. SEPARATE CONCERNS
────────────────────
Phase 2: Split into services
- Data ingestion
- Feature computation
- Model serving
- Business logic
- Monitoring

3. ADD FEEDBACK LOOPS
─────────────────────
Phase 3: Human-in-the-loop, active learning
- Corrections improve model
- Monitoring catches drift
- A/B testing validates changes

4. SCALE HORIZONTALLY
─────────────────────
- Stateless inference services
- Replicated feature stores
- Regional deployments

5. AUTOMATE EVERYTHING
──────────────────────
- CI/CD for models
- Automated retraining
- Automated monitoring alerts


THE EVOLUTION TIMELINE
======================

Week 1-2:     MVP (prove concept)
              │
              ▼
Month 1-2:    Production v1 (handle real traffic)
              │
              ▼
Month 3-6:    Operationalize (monitoring, CI/CD)
              │
              ▼
Month 6-12:   Optimize (latency, cost, accuracy)
              │
              ▼
Year 2+:      Scale & innovate (new features, regions)


THE BUILD vs BUY DECISION
=========================

                    Build (OSS)                  Buy (Cloud)
                    ───────────                  ───────────
Control             High                         Low
Initial cost        High (eng time)              Low (API)
Ongoing cost        Medium (ops)                 High (per-request)
Customization       Unlimited                    Limited
Time to MVP         Slow                         Fast
Time to scale       Medium                       Fast


RECOMMENDED APPROACH:

1. Start with cloud (fast MVP)
2. Measure actual costs and limitations
3. Build custom for:
   - Highest volume components
   - Unique requirements
   - Competitive differentiation
4. Keep cloud for:
   - Commodity services (OCR, speech)
   - Low volume features
   - Rapidly changing requirements
```

---

*This document shows how production ML systems actually work. The key lesson:
the ML model is just one component. Data pipelines, feature engineering,
serving infrastructure, monitoring, and feedback loops make up the majority
of a real system.*
