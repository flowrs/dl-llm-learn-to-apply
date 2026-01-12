# CLAUDE.md - Workspace Guide

## What This Workspace Is

This is a comprehensive deep learning course repository based on **CSE 493G1 / CS231n**,
enhanced with practical documentation connecting theory to production practice.

## Workspace Evolution

### Phase 1: Original Course Materials

The workspace began as a standard deep learning course with:

```
Original Structure:
├── lectures/           # 19 lectures covering foundations → frontiers
├── assignments/        # 3 programming assignments (k-NN → Transformers)
├── recitations/        # Hands-on Python/PyTorch sessions
└── readings/           # Research paper references
```

**Topics covered:**
- Weeks 1-2: Image classification, linear classifiers, neural networks
- Weeks 3-4: CNNs, training techniques, optimization
- Weeks 5-6: Self-supervised learning, RNNs, LSTMs
- Week 7: Attention mechanisms, Transformers, ViT
- Weeks 8-10: Detection, segmentation, generative models, RL
- Week 11: Large Language Models (tokenization, pre-training, fine-tuning, RLHF, prompting, RAG)

### Phase 2: Unified Conceptual Documentation

Added `DEEP_LEARNING_UNIFIED_INTUITION.md` to address the fragmentation problem:

```
Problem: Students learn topics in isolation without seeing the big picture

Solution: A single document that:
├── Shows evolution from simple → complex (k-NN → Transformers)
├── Connects every concept to what comes before/after
├── Uses ASCII diagrams to visualize architectures and data flow
├── Explains the "why" behind each technique
└── Provides a mental model for the entire field
```

**Key sections:**
- The Grand Picture (learning = optimization)
- Evolution from k-NN → Linear → Neural → CNN → RNN → Attention → Transformer
- Backpropagation and gradient flow patterns
- Optimization algorithms and their motivations
- Architecture deep-dives with ASCII diagrams
- Generative models (VAE, GAN, Diffusion)
- The unified view showing how everything connects

### Phase 3: Pitfalls and Debugging Guide

Added `DEEP_LEARNING_PITFALLS_GUIDE.md` to capture practitioner wisdom:

```
Problem: Students make the same mistakes repeatedly

Solution: A survival guide covering:
├── Conceptual pitfalls (wrong mental models)
├── Data pitfalls (leakage, imbalance, augmentation)
├── Training pitfalls (learning rate, numerical stability)
├── Architecture pitfalls (overcomplication, wrong receptive field)
├── Evaluation pitfalls (wrong metrics, test contamination)
├── Production pitfalls (train/inference mismatch, drift)
└── Debugging decision trees and symptom → cause tables
```

**Includes:**
- Concrete code examples showing wrong vs right approaches
- Visual explanations of failure modes
- The debugging decision tree
- Practitioner's checklist (before/during/after training)

### Phase 4: Production Systems Guide

Added `DEEP_LEARNING_IN_PRODUCTION.md` to bridge theory and practice:

```
Problem: Course teaches models but not how real systems work

Solution: Four complete production case studies:
├── E-commerce visual search (MVP → multi-region scale)
├── Manufacturing defect detection (edge → federated learning)
├── Document processing (rules → LayoutLMv3 → human-in-loop)
└── Fraud detection (rules → real-time ML → streaming features)
```

**Each application shows:**
- Phase 1 (MVP): Minimal viable architecture, code structure
- Phase 2 (Production v1): Scaling, separation of concerns
- Phase 3 (Mature): Full architecture with all components
- Data flows with latency breakdowns
- OSS vs Cloud decision matrices
- Ongoing operations workflows

### Phase 5: Organizational Guide

Added `BUILDING_ML_ORGANIZATIONS.md` to cover the human side:

```
Problem: Technical skills alone don't build successful companies

Solution: Four company stories showing organizational evolution:
├── Who to hire at each stage (MVP → Scale)
├── Roles beyond software engineer (domain experts, operations, field teams)
├── Team structures (Platform + Pods, Research/Production split, HITL ops)
├── How roles map to system components from the production doc
└── Weekly operating rhythms and on-call structures
```

**Key insights:**
- Domain expertise often matters more than ML expertise early on
- Operations teams are first-class citizens (annotation, field technicians)
- The Research ↔ Production split happens around 15 ML people
- Regulated industries need Model Risk and Compliance roles
- Only ~30% of a mature ML company is "engineering"

### Phase 6: Narrative Learning

Added `STYLEMATCH_THE_FULL_STORY.md` to make concepts tangible:

```
Problem: Abstract concepts don't stick without human context

Solution: A 1,200+ line narrative following StyleMatch from founding to scale:
├── Characters with real educational backgrounds (Stanford, Berkeley, UW, MIT PhDs)
├── Technical concepts embedded in story scenes and dialogues
├── Professors and PhD advisors as mentors shaping careers
├── The journey from classroom learning to production breakthroughs
└── How different backgrounds (systems, research, business) complement each other
```

**Technical concepts woven into the story:**
- Priya's CS231n homework → vectorized distance computation
- Professor Kowalski's lecture → backpropagation as message passing
- The color dominance bug → CNN feature hierarchies
- Alex's question → receptive fields and attention
- Jordan's optimization → GPU utilization, quantization, TensorRT
- Chen's migration → HNSW approximate nearest neighbor
- Dr. Adesanya's work → cross-attention for compatibility
- The data flywheel → continuous learning from user clicks

### Phase 7: Boundaries and Limitations

Added `LIMITS_OF_DEEP_LEARNING.md` to set realistic expectations:

```
Problem: Students need to know when NOT to use deep learning

Solution: Comprehensive guide to DL limitations:
├── Fundamental limitations (causation, OOD generalization, explainability)
├── Structural business challenges (strategy, relationships, ethics)
├── Practical constraints (cost-benefit, organizational readiness)
└── Decision framework for when to use DL vs alternatives
```

**Key categories covered:**
- When DL fundamentally can't solve the problem (causation, small data)
- When DL can solve it but shouldn't (regulatory, ethical, cost)
- When DL is overkill (simple rules would work)
- Decision tree for evaluating DL applicability

### Phase 8: Bridge to Modern AI

Added `DEEP_LEARNING_TO_LLM_APPLICATIONS.md` to connect course to LLMs:

```
Problem: Course teaches DL but industry is moving to LLM applications

Solution: Bridge document showing:
├── How course concepts apply to understanding LLMs
├── The LLM application stack (prompting, RAG, fine-tuning, agents)
├── Complete example: building a customer support bot
└── When to use traditional ML/DL vs LLMs
```

**Key sections:**
- Transformer architecture from course → LLM architecture
- New skills needed: prompt engineering, RAG, evaluation
- Production patterns for LLM applications
- Hybrid architectures combining DL and LLMs

### Phase 9: Hybrid Production Systems

Added `DL_AND_LLM_IN_PRODUCTION.md` to show real-world integration:

```
Problem: Businesses need both DL and LLMs, but how do they work together?

Solution: Four complete case studies of hybrid systems:
├── ShopSmart (E-commerce): DL for visual search, LLM for query understanding
├── MedAssist (Healthcare): DL for imaging, LLM for clinical explanations
├── WealthGuard (Finance): DL for risk models, LLM for advisory
└── MediaFlow (Content): DL for video processing, LLM for moderation
```

**Key patterns covered:**
- DL for perception, LLM for reasoning
- LLM for understanding, DL for execution
- DL for scale, LLM for edge cases
- Cost optimization through intelligent routing
- Confidence-based escalation

### Phase 10: The LLM Era Story

Added `STYLEMATCH_PART_2_THE_LLM_ERA.md` to continue the narrative:

```
Problem: How do existing AI companies adapt to the LLM revolution?

Solution: StyleMatch story continues (2026-2028):
├── Strategy sessions deciding where to apply LLMs
├── First LLM feature launches (ContentAI, SmartSearch)
├── Shopping Assistant with alignment challenges
├── Two flywheels: visual + language intelligence
└── Lessons on hybrid DL+LLM architectures
```

**Technical concepts woven into the story:**
- Query classification for routing (DL vs LLM)
- Hybrid search architecture (LLM understanding → DL retrieval)
- Graceful degradation when APIs fail
- The alignment problem in AI assistants
- Fine-tuning vs API tradeoffs
- Cross-flywheel data synergies

### Phase 11: LLM Week and Interactive Visualizations

Added `Week_11_Large_Language_Models.md` to provide comprehensive LLM coverage:

```
Problem: LLMs are the frontier but course materials end at Week 10

Solution: Complete LLM guide covering the full stack:
├── Tokenization (BPE, vocabulary, token quirks)
├── Pre-training (next token prediction, data, scaling)
├── Scaling laws (Chinchilla, emergent capabilities)
├── Fine-tuning (full, LoRA, QLoRA)
├── RLHF (reward models, PPO, DPO)
├── Prompting (zero/few-shot, chain-of-thought, self-consistency)
├── In-context learning
├── Inference optimization (KV cache, quantization, batching)
└── Building LLM applications (RAG, agents, tool use)
```

Added `visualizations/` folder with interactive ASCII diagram programs:

```
Problem: Static diagrams in markdown don't show step-by-step processes

Solution: Python console programs for each week:
├── week_01_02_visualizer.py   # Neurons, forward pass, gradient descent
├── week_03_04_visualizer.py   # Convolutions, pooling, CNN architectures
├── week_05_06_visualizer.py   # SSL, RNNs, LSTM, vanishing gradients
├── week_07_visualizer.py      # Attention, transformers, KV cache
├── week_08_10_visualizer.py   # VAE, GAN, diffusion, ethics
├── week_11_visualizer.py      # Tokenization, pre-training, RLHF, RAG
└── run_visualizations.py      # Main launcher for all weeks
```

**Features of visualizers:**
- Step-by-step animated ASCII diagrams
- Interactive menus for exploring concepts
- Conceptual explanations alongside visualizations
- No dependencies (pure Python)

## Current Workspace Structure

```
4931g/
├── CLAUDE.md                              # This file - workspace guide
│
├── DEEP_LEARNING_UNIFIED_INTUITION.md     # Conceptual connections (theory)
├── DEEP_LEARNING_PITFALLS_GUIDE.md        # Common mistakes (practice)
├── DEEP_LEARNING_IN_PRODUCTION.md         # Real systems (industry)
├── BUILDING_ML_ORGANIZATIONS.md           # Teams and hiring (people)
├── STYLEMATCH_THE_FULL_STORY.md           # Narrative case study (story)
├── LIMITS_OF_DEEP_LEARNING.md             # What DL can't do (boundaries)
├── DEEP_LEARNING_TO_LLM_APPLICATIONS.md   # Course to LLMs (frontier)
├── DL_AND_LLM_IN_PRODUCTION.md            # Hybrid DL+LLM systems (integration)
├── STYLEMATCH_PART_2_THE_LLM_ERA.md       # StyleMatch LLM story (narrative cont.)
│
├── Week_01_02_Foundations.md              # Neural network basics
├── Week_03_04_CNNs_Training.md            # CNNs and training techniques
├── Week_05_06_SSL_RNNs.md                 # Self-supervised learning, RNNs
├── Week_07_Attention_Transformers.md      # Attention and Transformers
├── Week_08_10_Advanced_Topics.md          # Advanced topics
├── Week_11_Large_Language_Models.md       # LLM comprehensive guide (NEW)
│
├── visualizations/                        # Interactive ASCII visualizers (NEW)
│   ├── run_visualizations.py              # Main launcher
│   ├── week_01_02_visualizer.py           # Foundations visualizer
│   ├── week_03_04_visualizer.py           # CNNs visualizer
│   ├── week_05_06_visualizer.py           # SSL/RNNs visualizer
│   ├── week_07_visualizer.py              # Attention visualizer
│   ├── week_08_10_visualizer.py           # Advanced topics visualizer
│   └── week_11_visualizer.py              # LLM visualizer
│
├── lectures/                              # Original course lectures
├── assignments/                           # Programming assignments
├── recitations/                           # Hands-on sessions
└── [other course materials]
```

## How the Documents Connect

```
                         LEARNING PATH
                         ═════════════

    ┌───────────────────────────────────────────────────────────┐
    │                Course Lectures & Assignments              │
    │                (foundations, theory, code)                │
    └─────────────────────────┬─────────────────────────────────┘
                              │
                              ▼
    ┌───────────────────────────────────────────────────────────┐
    │            DEEP_LEARNING_UNIFIED_INTUITION.md             │
    │                                                           │
    │   "How does it all fit together?"                         │
    │   - Concept evolution and connections                     │
    │   - ASCII architecture diagrams                           │
    │   - Mathematical intuitions                               │
    └─────────────────────────┬─────────────────────────────────┘
                              │
                              ▼
    ┌───────────────────────────────────────────────────────────┐
    │             DEEP_LEARNING_PITFALLS_GUIDE.md               │
    │                                                           │
    │   "What goes wrong and how to fix it?"                    │
    │   - Common mistakes with solutions                        │
    │   - Debugging strategies                                  │
    │   - Practitioner checklists                               │
    └─────────────────────────┬─────────────────────────────────┘
                              │
                              ▼
    ┌───────────────────────────────────────────────────────────┐
    │             DEEP_LEARNING_IN_PRODUCTION.md                │
    │                                                           │
    │   "How do real systems work?"                             │
    │   - MVP → Scale evolution                                 │
    │   - Data pipelines and MLOps                              │
    │   - Build vs Buy decisions                                │
    └─────────────────────────┬─────────────────────────────────┘
                              │
                              ▼
    ┌───────────────────────────────────────────────────────────┐
    │             BUILDING_ML_ORGANIZATIONS.md                  │
    │                                                           │
    │   "Who builds these systems?"                             │
    │   - Roles beyond software engineer                        │
    │   - Hiring sequences and team structures                  │
    │   - How people map to system components                   │
    └─────────────────────────┬─────────────────────────────────┘
                              │
                              ▼
    ┌───────────────────────────────────────────────────────────┐
    │              STYLEMATCH_THE_FULL_STORY.md                 │
    │                                                           │
    │   "What does this look like in practice?"                 │
    │   - Education → Production journey                        │
    │   - Technical concepts in human context                   │
    │   - How different backgrounds complement each other       │
    └─────────────────────────┬─────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
    ┌─────────────────────────┐     ┌─────────────────────────┐
    │  LIMITS_OF_DEEP_        │     │  DL_TO_LLM_             │
    │  LEARNING.md            │     │  APPLICATIONS.md        │
    │                         │     │                         │
    │  "When NOT to use DL?"  │     │  "What about LLMs?"     │
    │  - Fundamental limits   │     │  - Course → LLM bridge  │
    │  - Business constraints │     │  - Prompting, RAG, etc. │
    └─────────────────────────┘     └────────────┬────────────┘
                                                 │
                                                 ▼
                              ┌───────────────────────────────────┐
                              │     DL_AND_LLM_IN_PRODUCTION.md   │
                              │                                   │
                              │   "How do DL and LLM work together│
                              │    in real production systems?"   │
                              │   - 4 hybrid case studies         │
                              │   - Integration patterns          │
                              └────────────────┬──────────────────┘
                                               │
                                               ▼
                              ┌───────────────────────────────────┐
                              │   STYLEMATCH_PART_2_LLM_ERA.md    │
                              │                                   │
                              │   "How does StyleMatch adapt      │
                              │    to the LLM revolution?"        │
                              │   - Strategy and implementation   │
                              │   - Alignment challenges          │
                              └───────────────────────────────────┘


    ALTERNATIVE READING PATHS:
    ══════════════════════════

    For technical depth (DL focus):
    Course → UNIFIED_INTUITION → PITFALLS → IN_PRODUCTION

    For technical depth (DL + LLM):
    Course → UNIFIED_INTUITION → DL_TO_LLM → DL_AND_LLM_IN_PRODUCTION

    For career guidance:
    IN_PRODUCTION → BUILDING_ORGANIZATIONS → STYLEMATCH_STORY → PART_2

    For narrative learners:
    STYLEMATCH_STORY → STYLEMATCH_PART_2 → reference technical docs as needed

    For understanding boundaries:
    UNIFIED_INTUITION → LIMITS_OF_DEEP_LEARNING → DL_AND_LLM_IN_PRODUCTION

    For LLM application builders:
    DL_TO_LLM_APPLICATIONS → DL_AND_LLM_IN_PRODUCTION → STYLEMATCH_PART_2

    For founders/leaders:
    BUILDING_ORGANIZATIONS → IN_PRODUCTION → PITFALLS
```

## Usage Recommendations

### For Students New to Deep Learning
1. Start with course lectures (theory foundation)
2. Run `python visualizations/run_visualizations.py` to see concepts animated
3. Read `UNIFIED_INTUITION.md` alongside lectures (see connections)
4. Reference `PITFALLS_GUIDE.md` when doing assignments (avoid mistakes)
5. Read `IN_PRODUCTION.md` after course completion (see real applications)
6. Read `Week_11_Large_Language_Models.md` for comprehensive LLM coverage
7. Read `STYLEMATCH_STORY.md` to see how education connects to careers

### For Practitioners
1. Skim `UNIFIED_INTUITION.md` to refresh concepts
2. Use `PITFALLS_GUIDE.md` as debugging reference
3. Study `IN_PRODUCTION.md` for architecture patterns
4. Reference `BUILDING_ORGANIZATIONS.md` when growing teams

### For Building New Projects
1. Check `IN_PRODUCTION.md` for similar application architecture
2. Use `PITFALLS_GUIDE.md` checklist before/during/after training
3. Reference `UNIFIED_INTUITION.md` for architecture decisions
4. Use `BUILDING_ORGANIZATIONS.md` to plan hiring sequence

### For Founders and Leaders
1. Start with `BUILDING_ORGANIZATIONS.md` for team structure
2. Study `IN_PRODUCTION.md` for technical architecture decisions
3. Read `STYLEMATCH_STORY.md` for how technical and business decisions interplay
4. Reference `PITFALLS_GUIDE.md` to understand what your team faces

### For Career Planning
1. Read `STYLEMATCH_STORY.md` to see different career paths
2. Study `BUILDING_ORGANIZATIONS.md` for role definitions
3. Use course materials to build foundational skills
4. Reference `IN_PRODUCTION.md` to understand industry expectations

## Key Design Decisions

### ASCII Diagrams Over Images
- Work in any text editor or terminal
- Version control friendly (diff-able)
- Copy-paste into documentation
- Force clarity through simplicity

### Concrete Code Examples
- Python/PyTorch throughout
- Show wrong AND right approaches
- Production-ready patterns
- Copy-paste usable

### Progressive Complexity
- Each document builds on previous
- MVP → Production evolution pattern
- Simple → Complex in every section

### Narrative Learning
- Technical concepts embedded in human stories
- Characters with real educational backgrounds
- Dialogue-driven explanations of complex ideas
- Shows how different skill sets complement each other

### Multi-Perspective Coverage
- Theory (UNIFIED_INTUITION) → how concepts work
- Practice (PITFALLS) → what goes wrong
- Systems (IN_PRODUCTION) → how to build DL at scale
- People (ORGANIZATIONS) → who builds these systems
- Story (STYLEMATCH) → what the DL journey looks like
- Boundaries (LIMITS) → when NOT to use deep learning
- Bridge (DL_TO_LLM) → connecting course to modern LLMs
- Integration (DL_AND_LLM) → how DL and LLM work together in production
- Continuation (STYLEMATCH_PART_2) → adapting to the LLM era
- LLM Deep Dive (WEEK_11) → complete LLM stack from tokenization to RAG
- Interactive (VISUALIZATIONS) → animated ASCII diagrams for all concepts

## Future Extensions

Potential additions to this workspace:

```
Possible future documents:
├── INTERVIEW_PREP.md           # Common interview questions + answers
├── PAPER_READING_GUIDE.md      # How to read ML papers effectively
├── PROJECT_IDEAS.md            # Graduated project suggestions
├── MATH_FOUNDATIONS.md         # Linear algebra, calculus, probability review
└── FRAMEWORKS_COMPARISON.md    # PyTorch vs TensorFlow vs JAX
```

Potential story expansions:
```
├── VISIONGUARD_STORY.md        # Manufacturing AI narrative (edge, federated)
├── DOCUFLOW_STORY.md           # Document AI narrative (human-in-loop)
├── SHIELDPAY_STORY.md          # Fraud detection narrative (regulated industry)
└── PROFESSOR_PERSPECTIVES.md   # Stories from the advisor/teaching side
```

Potential enhancements:
- Interactive Jupyter notebooks for each concept
- Video walkthroughs of ASCII diagrams
- Industry guest lecture notes
- Competition (Kaggle) strategy guide

## Notes for Claude

When working in this workspace:

1. **Maintain ASCII diagram style** - All architecture diagrams should use ASCII art
   for consistency and accessibility

2. **Follow the evolution pattern** - When adding content, show progression from
   simple → complex, MVP → production

3. **Include concrete examples** - Abstract concepts should have code or specific
   examples alongside

4. **Cross-reference documents** - New content should link to existing materials
   where relevant

5. **Practitioner focus** - Balance theory with practical application; always
   answer "how would I use this?"

6. **Narrative integration** - When creating stories, embed technical concepts in
   dialogue and scenes; characters should have realistic educational backgrounds

7. **Organizational awareness** - Technical content should acknowledge the human
   side: who builds this, who operates it, what skills are needed

8. **Multi-level explanation** - Cover concepts at multiple levels: intuitive
   (for beginners), technical (for practitioners), and systemic (for architects)

## Document Relationships

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DOCUMENT MAP                                     │
│                                                                             │
│   UNIFIED_INTUITION ◄──── theoretical foundation ────► PITFALLS             │
│          │                                                  │               │
│          │ concepts                              debugging  │               │
│          ▼                                                  ▼               │
│   IN_PRODUCTION ◄─────────── architecture ──────────► ORGANIZATIONS         │
│          │                                                  │               │
│          │ systems                                  people  │               │
│          └────────────────────┬─────────────────────────────┘               │
│                               │                                             │
│                               ▼                                             │
│                      STYLEMATCH_STORY                                       │
│                      (integrates DL era)                                    │
│                               │                                             │
│          ┌────────────────────┼────────────────────┐                        │
│          │                    │                    │                        │
│          ▼                    │                    ▼                        │
│   LIMITS_OF_DL                │           DL_TO_LLM_APPS                    │
│   (boundaries)                │           (bridge to LLMs)                  │
│                               │                    │                        │
│                               │                    ▼                        │
│                               │          DL_AND_LLM_IN_PRODUCTION           │
│                               │          (hybrid systems)                   │
│                               │                    │                        │
│                               └────────────────────┘                        │
│                                         │                                   │
│                                         ▼                                   │
│                              STYLEMATCH_PART_2                              │
│                              (integrates LLM era)                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

*Last updated: January 2026 (Phase 11)*
*Workspace purpose: Comprehensive deep learning education from theory to production to people*

## Changelog

- **Phase 1**: Original course materials (lectures, assignments, recitations)
- **Phase 2**: Added DEEP_LEARNING_UNIFIED_INTUITION.md (concept connections)
- **Phase 3**: Added DEEP_LEARNING_PITFALLS_GUIDE.md (practitioner wisdom)
- **Phase 4**: Added DEEP_LEARNING_IN_PRODUCTION.md (4 production case studies)
- **Phase 5**: Added BUILDING_ML_ORGANIZATIONS.md (teams, hiring, roles)
- **Phase 6**: Added STYLEMATCH_THE_FULL_STORY.md (narrative case study with educational backgrounds)
- **Phase 7**: Added LIMITS_OF_DEEP_LEARNING.md (what DL can't do and why)
- **Phase 8**: Added DEEP_LEARNING_TO_LLM_APPLICATIONS.md (bridging course to modern LLM development)
- **Phase 9**: Added DL_AND_LLM_IN_PRODUCTION.md (4 case studies of hybrid DL+LLM production systems)
- **Phase 10**: Added STYLEMATCH_PART_2_THE_LLM_ERA.md (StyleMatch story continues into LLM integration)
- **Phase 11**: Added Week_11_Large_Language_Models.md (comprehensive LLM guide) and visualizations/ folder (interactive ASCII visualizers for all weeks)
