# Building ML Organizations: The People Behind the Systems

## A CTO's Guide to Hiring, Structuring, and Winning

This document tells the story of building four ML companies from founding to scale.
Each story follows the architecture evolution from `DEEP_LEARNING_IN_PRODUCTION.md`,
showing who builds each component, how teams grow, and what organizational challenges emerge.

---

# Story 1: StyleMatch — Visual Search for E-Commerce

## Chapter 1: The Founding Team (MVP Phase)

It's January. Three co-founders sit in a cramped WeWork, staring at a whiteboard.

```
THE FOUNDING TEAM
=================

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   SARAH (CEO/Product)        MARCUS (CTO)         PRIYA (ML Lead)   │
│   ─────────────────          ──────────           ──────────────    │
│   Ex-product manager         Full-stack           PhD dropout       │
│   at major retailer          engineer             (computer vision) │
│                                                                     │
│   "I know what               "I can build        "I can make       │
│   customers want"            anything fast"       models work"      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Week 1-4: Building the MVP**

Marcus builds the monolith Flask API. He's coding 14-hour days, handling everything
from the AWS account setup to the nginx configuration. The architecture is simple:

```
WHO BUILDS WHAT (MVP)
=====================

Marcus (CTO/Full-Stack):
├── Flask API server
├── Image upload handling
├── Basic frontend
├── DevOps (single EC2 instance)
└── Database schema

Priya (ML Lead):
├── ResNet feature extraction
├── Embedding generation script
├── Similarity search (NumPy)
└── Model evaluation metrics

Sarah (CEO/Product):
├── Customer interviews
├── Demo presentations
├── Pilot partnership negotiations
└── Defining success metrics
```

Priya spends nights running embeddings on her gaming PC. "We don't need cloud GPUs yet,"
she says. "50,000 products takes 6 hours. I just run it overnight."

**The First Hire: A Generalist**

By month 2, Marcus is drowning. They hire **Alex**, a backend engineer who can
"figure things out." Alex's job description is simple: "Help Marcus not die."

```
TEAM: MONTH 2
=============

Sarah (CEO) ─────────────────────────────────────────────
                    │
        ┌───────────┴───────────┐
        │                       │
    Marcus (CTO)            Priya (ML)
        │
      Alex (Backend)

Total: 4 people
Burn rate: $80K/month
Runway: 14 months
```

Alex takes over the API endpoints while Marcus focuses on infrastructure. Priya
is still the only one who understands the ML code. This is fine for now.

---

## Chapter 2: First Production System (Month 3-6)

They land a pilot with a mid-size fashion retailer. 100,000 products. Real traffic.
The monolith won't scale.

**The Crisis**

Week 1 of the pilot: the site crashes during a flash sale. Marcus gets the 3 AM page.
The single EC2 instance ran out of memory loading embeddings for 100K products.

"We need to split this up," Marcus tells the team. "And I need help."

**The Scaling Hires**

```
NEW ROLES NEEDED
================

Problem                          Role Needed
───────                          ───────────
Can't run ML on production      ML Engineer (not researcher!)
server without crashes          Someone who knows TensorRT,
                                model optimization, serving

Database is a mess,             Data Engineer
embeddings stored as files      Build proper pipelines,
                                vector database

Site keeps crashing,            DevOps/SRE
no monitoring                   Kubernetes, monitoring,
                                on-call rotations

Don't know if users             Product Analyst
are actually finding what       Metrics, A/B testing,
they want                       user behavior analysis
```

They hire:
- **Jordan** (ML Engineer): Came from a self-driving car company. Knows how to make
  models fast. Immediately starts converting Priya's PyTorch models to TensorRT.
- **Chen** (Data Engineer): Ex-Spotify. Sets up Airflow, builds the embedding pipeline,
  migrates to Milvus for vector search.
- **Raj** (DevOps): Kubernetes expert. Gets them on EKS within a month.

```
WHO OWNS WHAT (Production V1)
=============================

Referring to the architecture:

┌─────────────────────────────────────────────────────────────────┐
│                     KUBERNETES CLUSTER                          │
│                                                                 │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │  API Pods ◄──── Alex & Marcus                            │  │
│   │  (FastAPI)      Handle request/response logic            │  │
│   └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│   ┌──────────────────────────▼───────────────────────────────┐  │
│   │  Model Serving ◄──── Jordan                              │  │
│   │  (TorchServe)        Optimized inference, GPU mgmt       │  │
│   └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│   ┌──────────────────────────▼───────────────────────────────┐  │
│   │  Vector DB ◄──── Chen                                    │  │
│   │  (Milvus)        Embedding pipeline, index management    │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │  Infrastructure ◄──── Raj                                │  │
│   │  (K8s, monitoring)    On-call, scaling, reliability      │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Meanwhile:
- Priya: Experiments with new models, improves accuracy
- Sarah: Manages retailer relationship, defines roadmap
- New: Lisa (Product Analyst) - Defines metrics, runs A/B tests
```

**The Organizational Tension**

A conflict emerges. Priya develops a new model that's 5% more accurate. She wants
to ship it. Jordan pushes back: "It's 3x slower. We can't serve it at this latency."

Marcus has to mediate. They establish a rule: **models need both accuracy AND
latency approval before shipping.**

```
THE MODEL SHIPPING PROCESS (born from conflict)
================================================

Priya trains     Jordan optimizes     Both sign off     Raj deploys
new model   ──►  for production  ──►  on metrics   ──►  to staging
    │                  │                   │                │
    ▼                  ▼                   ▼                ▼
Accuracy           Latency p99         Accuracy ✓        Canary
metrics            < 50ms              Latency ✓         rollout
                                       Cost ✓
```

This is the birth of their **MLOps process**.

---

## Chapter 3: Scaling the Organization (Month 6-18)

StyleMatch raises a Series A. They have 5 retail customers and need to scale.

**The New Challenges**

```
SCALING PROBLEMS
================

Technical:
- Each retailer wants customization
- Need personalization (not just visual similarity)
- Customers in Europe = GDPR compliance
- 10x traffic growth expected

Organizational:
- Priya is bottleneck for all ML decisions
- No documentation, everything in people's heads
- On-call is burning out Raj (he's alone)
- Product requests pile up faster than eng can build
```

**The Hiring Wave**

Sarah and Marcus plan a hiring spree. But what roles?

```
THE HIRING MATRIX
=================

                        IMMEDIATE NEED    HIRE WHEN?
                        ──────────────    ──────────
ML Research Scientist   Medium            When Priya overwhelmed
ML Engineer             HIGH              Now (Jordan needs help)
Backend Engineer        HIGH              Now (Alex overwhelmed)
Data Engineer           HIGH              Now (Chen needs help)
DevOps/SRE              HIGH              Now (Raj burning out)
Frontend Engineer       Medium            After backend stable
Product Manager         HIGH              Now (Sarah can't do both)
Data Analyst            Medium            When PM joins
Designer                Low               Outsource for now
Security Engineer       Medium            Before enterprise deals
```

They hire 8 people in 3 months:
- 2 ML Engineers (under Jordan)
- 2 Backend Engineers (under Alex, who becomes a team lead)
- 1 Data Engineer (under Chen)
- 1 SRE (under Raj)
- 1 Product Manager (**Nina**)
- 1 ML Research Scientist (**Dr. Okonkwo**, PhD in recommendation systems)

**The New Org Structure**

```
MONTH 12 ORGANIZATION
=====================

                            Sarah (CEO)
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
        Marcus (CTO)        Nina (Product)     [Sales - hiring]
            │                   │
    ┌───────┼───────┬───────────┤
    │       │       │           │
 Priya    Jordan   Alex       Chen      Raj
 (ML       (ML      (Backend   (Data     (Platform/
 Research) Eng)     Eng)       Eng)      SRE)
    │       │        │          │          │
Dr. O    2 MLEs   2 BEs      1 DE       1 SRE


WHAT EACH TEAM OWNS:

ML Research (Priya + Dr. Okonkwo):
├── New model architectures
├── Accuracy improvements
├── Research → prototype handoff
└── Academic partnerships

ML Engineering (Jordan + 2):
├── Model optimization (TensorRT, quantization)
├── Model serving infrastructure
├── Feature store implementation
├── A/B testing infrastructure for models

Backend Engineering (Alex + 2):
├── API development
├── Integration with retailers
├── Caching layer (Redis)
├── Business logic

Data Engineering (Chen + 1):
├── Embedding pipelines (Airflow)
├── Vector database management
├── Data quality monitoring
├── ETL for training data

Platform/SRE (Raj + 1):
├── Kubernetes management
├── Monitoring & alerting
├── On-call rotation (finally!)
├── Cost optimization
├── Security & compliance
```

**The Specialization of Priya**

A key moment: Priya stops writing production code.

Before, she did everything: research, training, deployment, debugging production issues.
Now, with Jordan's team handling production ML, she focuses purely on research.

"I miss being in the code," she admits. "But Dr. Okonkwo and I can now explore ideas
that would never survive the pressure of production deadlines."

This is the **research/production split** that mature ML orgs need.

```
THE ML TEAM SPLIT
=================

BEFORE (Priya does everything):

    Idea ──► Research ──► Train ──► Optimize ──► Deploy ──► Monitor
                            │
                         Priya
                     (bottleneck!)


AFTER (specialized roles):

    Idea ──► Research ──► Train ──► Optimize ──► Deploy ──► Monitor
                │           │           │           │          │
             Priya      Priya/      Jordan's    Jordan's    Raj +
             Dr. O      Jordan      team        team      Jordan

    Research hands off trained models to ML Eng
    ML Eng handles everything production
```

---

## Chapter 4: The Mature Organization (Year 2+)

StyleMatch now serves 50 retailers across 3 continents. The company has 85 employees.

**The Full Organization**

```
YEAR 2 ORGANIZATION CHART
=========================

                                    Board of Directors
                                           │
                                      Sarah (CEO)
                                           │
                ┌──────────────┬───────────┼───────────┬──────────────┐
                │              │           │           │              │
           Marcus (CTO)    Nina (CPO)   CFO       VP Sales      VP People
                │              │                      │
    ┌───────────┴──────────┐   │              ┌──────┴──────┐
    │                      │   │              │             │
VP Engineering        VP ML/AI │          Sales Team   Customer
    │                     │    │          (8 people)    Success
    │                     │    │                        (5 people)
    │                     │    │
┌───┴────┬────────┐   ┌──┴────┴────┐
│        │        │   │           │
Platform Backend  tic ML        ML        Product  Design  Analytics
(Raj)    (Alex)  tic  Research  Platform  (3 PMs)  (2)     (3)
 │        │   │      (Priya)   (Jordan)
 │        │   │        │          │
4 SREs  8 BEs  │     4 Researchers  6 MLEs
              │                │
           3 FEs          3 Data Eng


SPECIALIZED ROLES THAT EMERGED:

ML Research Scientist (4):
- PhD-level researchers
- Publish papers, attend conferences
- 6-month research horizons
- Priya now manages, doesn't code daily

ML Engineer (6):
- Production model optimization
- Feature engineering at scale
- Model monitoring and retraining
- Jordan promoted to VP, manages team

MLOps Engineer (2, under Jordan):
- CI/CD for models
- Experiment tracking (MLflow)
- Model registry management
- Training infrastructure

Data Engineer (3):
- Real-time feature pipelines
- Data quality and lineage
- Vector database scaling

Analytics Engineer (2):
- Business metrics pipelines
- A/B test analysis
- Executive dashboards

Product Manager (3):
- Nina hired 2 PMs under her
- One owns "Search," one owns "Recommendations"
- Third owns "Retailer Platform"

Product Analyst (3):
- Deep-dive analysis
- Feature impact measurement
- Customer behavior modeling

Solutions Engineer (3, under Sales):
- Technical sales support
- Customer integrations
- Custom requirements gathering
```

**The Weekly Rhythm**

```
THE OPERATING CADENCE
=====================

MONDAY:
├── 9 AM: Leadership sync (Sarah + directs)
├── 10 AM: ML Research review (Priya's team)
└── 2 PM: Sprint planning (each team)

TUESDAY:
├── 10 AM: Platform review (incidents, metrics)
└── 2 PM: Product-Engineering sync

WEDNESDAY:
├── 10 AM: Model review (accuracy, latency, cost)
├── 2 PM: Customer success sync
└── 4 PM: All-hands (bi-weekly)

THURSDAY:
├── 10 AM: Data quality review
└── 2 PM: Security review (monthly)

FRIDAY:
├── 10 AM: Demo day (show what shipped)
├── 2 PM: Retrospectives
└── 4 PM: Research reading group (optional)


WHO ATTENDS MODEL REVIEW (WEDNESDAY 10 AM):

A critical meeting where all ML decisions are made:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Attendees:                                                    │
│   - Jordan (ML Platform VP) - chairs                            │
│   - Priya (ML Research VP) - presents research results          │
│   - Raj (Platform VP) - infrastructure constraints              │
│   - Product PM for affected product                             │
│   - On-call MLE from last week                                  │
│                                                                 │
│   Decisions made:                                               │
│   - Which models to promote to production                       │
│   - Rollout strategy (% traffic)                                │
│   - Kill decisions for underperforming models                   │
│   - Resource allocation for experiments                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

# Story 2: VisionGuard — Manufacturing Defect Detection

## Chapter 1: The Factory Floor Origin

Different company, different challenges. VisionGuard starts inside a manufacturing
company—it's an internal project that spins out.

```
THE ORIGIN STORY
================

ACME Manufacturing has a quality problem. 2% defect rate, costing $5M/year.

The plant manager, **Tom**, convinces the CTO to try "that AI thing."

They hire:
- **Dr. Wei** (Consultant): Computer vision expert, part-time
- **Jake** (Controls Engineer): Knows PLCs, cameras, factory floor

This isn't a startup. It's a skunkworks project inside a traditional manufacturer.
```

**The Unusual Team**

Manufacturing AI requires a different skill mix than web companies:

```
MANUFACTURING ML TEAM (MVP)
===========================

Dr. Wei (CV Consultant, 2 days/week):
├── Model architecture selection
├── Training pipeline setup
├── Accuracy benchmarking
└── Research guidance

Jake (Controls Engineer):
├── Camera selection and mounting
├── Lighting setup (critical!)
├── PLC integration
├── Trigger timing
└── Edge device selection

Tom (Plant Manager, sponsor):
├── Business case
├── Production line access
├── Operator buy-in
├── Defect sample collection
└── Ground truth labeling coordination

Maria (Quality Inspector, part-time):
├── Defect taxonomy definition
├── Labeling training data
├── Validating model decisions
└── Edge case identification

Notice: No traditional "software engineer" yet!
The first hires are domain experts, not coders.
```

**The Labeling Challenge**

Here's something web companies don't face: getting training data requires
*stopping the production line*.

```
THE DATA COLLECTION PROBLEM
===========================

Web company:                    Manufacturing:

Users generate data             Must manufacture defects
constantly                      intentionally, or wait
       │                               │
       ▼                               ▼
Millions of images              Maybe 500 defect images
per day                         after 3 months
       │                               │
       ▼                               ▼
Labels from user                Labels from quality
clicks (implicit)               inspectors (expensive)


SOLUTION: Active involvement of quality team

Maria (Quality Inspector) becomes critical:
- She defines what counts as a defect
- She reviews model predictions
- She catches edge cases the model misses
- Her expertise IS the training data

Maria is not "just labeling" — she's encoding decades
of quality expertise into the model.
```

---

## Chapter 2: From Project to Product

The pilot works. 0.5% defect rate, down from 2%. $3.5M annual savings.

Tom pitches the board: "We should sell this to other manufacturers."

VisionGuard spins out as a separate company.

**The New Hires**

```
SPINNING OUT: NEW ROLES NEEDED
==============================

Jake becomes CTO (he knows the domain)

Immediate hires:

1. ML Engineer (Edge Specialist) - "Yuki"
   - TensorRT, Jetson, model optimization
   - Unlike web MLEs, must understand hardware
   - Previous experience: autonomous vehicles

2. Embedded Systems Engineer - "Pavel"
   - Camera SDKs, frame grabbers
   - Real-time Linux, latency optimization
   - Previous experience: industrial automation

3. Solutions Engineer - "Ahmed"
   - Customer factory assessments
   - Lighting and camera recommendations
   - Installation and calibration
   - Part technical, part sales

4. Data Annotation Lead - "Lisa"
   - Manages labeling workforce
   - Quality control on labels
   - Works with customer quality teams
   - Often former quality inspector herself


NOTABLY ABSENT (for now):
- Backend engineers (no web app yet)
- Data scientists (domain expertise > statistics)
- Product managers (Jake is playing that role)
```

**The Organizational Insight**

Manufacturing AI companies are *hardware-software hybrids*. The org structure
reflects this:

```
VISIONGUARD ORG (YEAR 1)
========================

                        Jake (CEO/CTO)
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
      ML Team            Field Team          Operations
          │                   │                   │
    ┌─────┴─────┐       ┌─────┴─────┐            │
    │           │       │           │            │
  Yuki       Dr. Wei  Ahmed      Pavel         Lisa
  (MLE)    (Research) (Solutions) (Embedded)  (Annotation)
             part-time


WHAT MAKES THIS DIFFERENT:

1. "Field Team" exists
   - These people GO TO FACTORIES
   - Install cameras, calibrate systems
   - Train operators on the system
   - Handle on-site troubleshooting

2. Solutions Engineer is hybrid role
   - Technical enough to assess factory conditions
   - Sales-y enough to scope projects
   - Travels 60% of the time

3. Annotation is in-house
   - Manufacturing defects need domain expertise
   - Can't crowdsource to Mechanical Turk
   - Lisa hires former quality inspectors
```

---

## Chapter 3: Scaling to Multiple Factories

Year 2: VisionGuard has 15 factory deployments across 8 customers.

**The Complexity Explosion**

```
THE MULTI-CUSTOMER CHALLENGE
============================

Each customer has:
- Different products (cars vs electronics vs food)
- Different defect types
- Different camera setups
- Different lighting conditions
- Different integration requirements (PLCs, MES systems)

One model doesn't fit all.

SOLUTION: Platform + customization team

┌─────────────────────────────────────────────────────────────────┐
│                      PLATFORM TEAM                              │
│                                                                 │
│   Builds reusable components:                                   │
│   - Base model architecture                                     │
│   - Transfer learning pipeline                                  │
│   - Edge deployment framework                                   │
│   - Monitoring and alerting                                     │
│   - Annotation tooling                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CUSTOMER SUCCESS PODS                        │
│                                                                 │
│   Pod 1 (Automotive):      Pod 2 (Electronics):                 │
│   - 1 Solutions Eng        - 1 Solutions Eng                    │
│   - 1 MLE (fine-tuning)    - 1 MLE (fine-tuning)                │
│   - 1 Field Tech           - 1 Field Tech                       │
│   - Shared: Annotation     - Shared: Annotation                 │
│                                                                 │
│   Deep expertise in        Deep expertise in                    │
│   automotive defects       PCB defects                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**The Pod Structure**

```
CUSTOMER SUCCESS POD (detailed)
===============================

┌─────────────────────────────────────────────────────────────────┐
│                     AUTOMOTIVE POD                              │
│                     (3 customers, 6 factories)                  │
│                                                                 │
│   Pod Lead: Ahmed (Solutions Engineer)                          │
│   ├── Owns customer relationship                                │
│   ├── Scopes new deployment projects                            │
│   ├── Coordinates across functions                              │
│   │                                                             │
│   MLE: Kenji                                                    │
│   ├── Fine-tunes base model for each customer                   │
│   ├── Analyzes model failures                                   │
│   ├── Implements customer-specific features                     │
│   │                                                             │
│   Field Technician: Rosa                                        │
│   ├── On-site installation                                      │
│   ├── Camera calibration                                        │
│   ├── Operator training                                         │
│   ├── First-line troubleshooting                                │
│   │                                                             │
│   Shared: Annotation team (central)                             │
│   ├── Pod submits labeling requests                             │
│   ├── Central team manages workforce                            │
│   └── Quality reviewed by pod MLE                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

WHY PODS WORK FOR MANUFACTURING:

1. Domain expertise accumulates
   - Kenji becomes THE expert on automotive paint defects
   - Rosa knows every camera angle for car body inspection

2. Customer relationships are sticky
   - Ahmed knows the plant managers by name
   - Trust is built over years, not months

3. Knowledge doesn't need to transfer
   - Automotive pod doesn't need to know about PCBs
   - Reduces cognitive load
```

---

## Chapter 4: The Mature Manufacturing AI Company

Year 4: VisionGuard has 200 employees, 80 factory deployments, $50M ARR.

```
VISIONGUARD ORG (MATURE)
========================

                                 Board
                                   │
                              Jake (CEO)
                                   │
        ┌──────────┬───────────────┼───────────────┬──────────────┐
        │          │               │               │              │
     CTO        VP Product      VP Sales      VP Customer      CFO
   (new hire)                                  Success
        │          │               │               │
   ┌────┴────┐     │               │         ┌─────┴─────┐
   │         │     │               │         │           │
Platform  Research  Product       Sales    CS Pods    Training
   │         │      (3 PMs)       (10)     (5 pods)   Academy
   │         │                               │
┌──┴──┐   ┌──┴──┐                     ┌──────┴──────┐
│     │   │     │                     │             │
Edge  ML  CV    Applied           Pod Lead    Field Ops
Infra  Ops  Research  ML              │         (15 techs)
(8)   (4)  (6)       (8)              │
                                 ┌────┴────┐
                                 │    │    │
                               Mfg  Auto  Food
                               (12) (10)  (8)


ROLES UNIQUE TO MANUFACTURING AI:

Field Operations Manager:
├── Manages 15 field technicians
├── Scheduling and logistics
├── Equipment inventory (cameras, edge devices)
├── Installation standards and procedures
└── Safety training and compliance

Training Academy Lead:
├── Trains customer operators
├── Creates training materials
├── Certification programs
├── "Train the trainer" programs

Applied ML Engineer (different from Platform MLE):
├── Lives in customer success pods
├── Fine-tuning, not research
├── Customer-specific model development
├── Rapid iteration, not state-of-art

Integration Engineer:
├── Connects to customer MES/ERP systems
├── PLC programming
├── Data export and reporting
└── Enterprise IT coordination
```

**The On-Call Structure**

Manufacturing AI has different on-call needs than web apps:

```
ON-CALL: MANUFACTURING vs WEB
=============================

Web company on-call:           Manufacturing on-call:
- Mostly software issues       - Hardware + software issues
- Remote debugging             - May need physical presence
- 5-minute response time       - Factory runs 24/7, but humans
- Rollback is easy               are there only during shifts
                               - "Rollback" might mean reverting
                                 to manual inspection

VISIONGUARD ON-CALL STRUCTURE:

Level 1: Platform SRE (remote)
├── Monitors all deployments
├── Handles software issues
├── Escalates hardware issues

Level 2: Pod Field Technician (regional)
├── On-call for their factories
├── Can drive to site within 2 hours
├── Handles camera, lighting, edge device issues

Level 3: Pod MLE (remote)
├── Model accuracy issues
├── Retraining decisions
├── Novel defect patterns

Level 4: Platform ML (rarely needed)
├── Fundamental model issues
├── Cross-customer patterns
```

---

# Story 3: DocuFlow — Invoice Processing

## Chapter 1: The Accountant's Nightmare

DocuFlow starts differently: it's founded by an accountant.

**Maya** ran AP (accounts payable) at a healthcare company. Her team processed
8,000 invoices per month. Manually.

"I spent $400K/year on a team doing data entry," she says. "There has to be a better way."

She recruits **Dmitri**, a friend's son who "knows computers."

```
THE FOUNDING DYNAMIC
====================

Maya (CEO):                      Dmitri (CTO):
- Domain expert                  - Technical generalist
- Knows invoice processing       - Taught himself Python
  inside and out                 - Never worked with ML
- Has customer relationships     - Eager to learn
- Can't code at all

This is a DOMAIN-LED founding, not a TECH-LED founding.

The advantage: Maya knows exactly what to build.
The risk: Can Dmitri actually build it?
```

**The Template MVP**

Dmitri builds the rule-based system:

```
MVP ARCHITECTURE (see IN_PRODUCTION.md)
=======================================

Dmitri builds everything:
├── PDF to text (pdfplumber)
├── Template regex rules
├── Simple web interface
├── Database (PostgreSQL)
└── Deployment (Heroku)

Maya:
├── Defines the templates
├── Tests on real invoices
├── Recruits pilot customer (her old employer!)
├── Defines accuracy requirements

Key insight: The MVP has NO ML.
It's pure templates and regex.
This is intentional.
```

Maya tests every invoice personally. "This one failed because the vendor
changed their format." She updates the template.

**The First Technical Hire**

After 6 months, Dmitri is maintaining 50 vendor templates. It's unsustainable.

They hire **Aisha**, an ML engineer who actually knows NLP.

```
AISHA'S ASSESSMENT
==================

"You don't need 50 templates. You need a model that generalizes."

She proposes:
1. LayoutLMv3 for entity extraction
2. Keep templates as fallback for edge cases
3. Human-in-the-loop for low-confidence predictions

Maya's reaction: "Will it be as accurate as my templates?"

Aisha: "Eventually, yes. And it will handle vendors
       you've never seen before."

This is the TEMPLATE → ML transition.
```

---

## Chapter 2: The Human-in-the-Loop System

Aisha builds the ML system, but Maya insists on human review.

"In accounting, accuracy isn't 95%. It's 100%. We need humans checking everything
until the AI proves itself."

```
THE HITL ARCHITECTURE (from IN_PRODUCTION.md)
=============================================

WHO OPERATES EACH COMPONENT:

┌─────────────────────────────────────────────────────────────────────────┐
│                         PROCESSING LAYER                                │
│                                                                         │
│   Document Workers ◄──── Dmitri (maintains infrastructure)              │
│        │                                                                │
│        ├── OCR ◄──── Cloud service (Google Vision)                      │
│        ├── Classification ◄──── Aisha (model development)               │
│        ├── Extraction ◄──── Aisha (LayoutLMv3 fine-tuned)               │
│        └── Validation ◄──── Maya (defines business rules)               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
    High Confidence            Low Confidence
    (auto-process)             (human review)
            │                       │
            │                       ▼
            │              ┌─────────────────┐
            │              │  Review Queue   │
            │              │                 │
            │              │  Operated by:   │
            │              │  DOCUMENT       │
            │              │  PROCESSORS     │
            │              │  (new role!)    │
            │              └─────────────────┘
            │                       │
            ▼                       ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                      Corrections feed back to training              │
    │                      ◄──── Aisha uses for model improvement         │
    └─────────────────────────────────────────────────────────────────────┘


THE DOCUMENT PROCESSOR ROLE
===========================

This is a NEW role that doesn't exist in most tech companies.

Document Processor:
├── Reviews low-confidence extractions
├── Corrects model mistakes
├── Flags new document types
├── Maintains extraction quality
└── NOT a software role — often former data entry specialists

Why this role matters:
1. Their corrections become training data
2. They catch patterns the model misses
3. They're the quality guarantee
4. They understand accounting context

Maya hires 3 document processors initially.
They're her former colleagues from AP.
```

**The Annotation Team Structure**

```
DOCUFLOW: THE HUMAN LAYER
=========================

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Document Processing Team (reports to Operations, not Eng)    │
│                                                                 │
│   Lead: Sandra (former AP supervisor)                           │
│   ├── Document Processor: Kim                                   │
│   ├── Document Processor: Luis                                  │
│   ├── Document Processor: Priti                                 │
│   │                                                             │
│   Responsibilities:                                             │
│   - Review queue management                                     │
│   - Quality targets (99.5% accuracy)                            │
│   - SLA management (4-hour turnaround)                          │
│   - Training new processors                                     │
│   - Escalation to Aisha for model issues                        │
│                                                                 │
│   Key metrics:                                                  │
│   - Documents processed per day                                 │
│   - Accuracy rate                                               │
│   - Auto-processing rate (goal: increase over time)             │
│   - Time per document (decreasing = good)                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

The relationship between Aisha and Sandra is critical:

Sandra's team finds problems → Reports to Aisha → Aisha improves model
                                                        │
                                                        ▼
                              Fewer problems → Sandra's team more efficient

This is the FEEDBACK LOOP that makes the system improve.
```

---

## Chapter 3: Scaling the Platform

Year 2: DocuFlow has 50 customers, processing 500K invoices/month.

**The Org Evolution**

```
DOCUFLOW ORG (YEAR 2)
=====================

                            Maya (CEO)
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
        Dmitri (CTO)       VP Operations         VP Sales
            │                   │                  (new)
    ┌───────┴───────┐     ┌─────┴─────┐
    │               │     │           │
Engineering       ML    Document    Customer
    │               │    Processing  Onboarding
    │               │        │            │
┌───┴───┐      ┌───┴───┐    │       ┌────┴────┐
│       │      │       │    │       │         │
Backend  DevOps Aisha   Data    Sandra   Onboarding
(3)      (2)    (Lead)  Eng     (Lead)   Specialists
                  │      │        │       (3)
               3 MLEs    2     12 Doc
                               Processors


NEW ROLES THAT EMERGED:

Customer Onboarding Specialist:
├── Helps new customers configure
├── Collects sample invoices
├── Sets up vendor templates (still needed for edge cases!)
├── Trains customer AP teams
└── Bridges Sales and Operations

Data Engineer (document-focused):
├── OCR pipeline optimization
├── Document storage and retrieval
├── Training data management
├── Quality metrics pipelines

Operations Manager (Sandra promoted):
├── Manages document processing team
├── Workforce planning
├── Quality assurance
├── Process optimization
└── Hires for their team independently
```

**The Quality Metrics Dashboard**

```
WHAT OPERATIONS MONITORS
========================

Sandra's daily dashboard:

┌─────────────────────────────────────────────────────────────────┐
│                     DOCUFLOW OPERATIONS                         │
│                     Date: 2024-01-15                            │
│                                                                 │
│   Today's Volume:        2,847 invoices                         │
│   Auto-Processed:        2,562 (90.0%)  ↑ from 85% last month   │
│   Human Reviewed:          285 (10.0%)                          │
│                                                                 │
│   Average Processing Time:                                      │
│   ├── Auto: 12 seconds                                          │
│   └── Human: 4.2 minutes                                        │
│                                                                 │
│   Accuracy (sampled):    99.7%                                  │
│                                                                 │
│   Queue Health:                                                 │
│   ├── Current queue:     47 documents                           │
│   ├── Oldest item:       23 minutes                             │
│   └── SLA breaches:      0                                      │
│                                                                 │
│   Top Failure Reasons:                                          │
│   1. New vendor format (34%)                                    │
│   2. Poor scan quality (28%)                                    │
│   3. Handwritten notes (21%)                                    │
│   4. Multi-page tables (17%)                                    │
│                                                                 │
│   [Flag for Aisha: New vendor "GlobalTech" - 15 failures]       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Sandra and Aisha meet daily to review failure patterns.
This meeting is where model improvement happens.
```

---

# Story 4: ShieldPay — Fraud Detection

## Chapter 1: The Bank's Problem

ShieldPay is different again: it's a B2B infrastructure company,
not a consumer product.

Banks need fraud detection but don't want to build it themselves.
ShieldPay sells "Fraud Detection as a Service."

```
THE FOUNDING TEAM
=================

Marcus (CEO):
- Former bank CTO
- Deep relationships with banks
- Understands regulatory requirements
- Technical enough to hire well

Dr. Fatima (Chief Scientist):
- PhD in adversarial ML
- Former research scientist at payment processor
- Published papers on fraud detection
- Understands the fraud arms race

Ravi (CTO):
- Ex-Stripe infrastructure
- Built real-time systems at scale
- Knows payments inside and out

This is a DEEP DOMAIN team. All three have 10+ years
in payments/fraud. This is not an accident.
```

**The Regulatory Reality**

Fraud detection in finance is different from other ML:

```
WHAT MAKES FRAUD DETECTION SPECIAL
==================================

1. REGULATORY REQUIREMENTS

   The model isn't just code. It's auditable.

   Compliance Officer: Reviews model for:
   ├── Disparate impact (is it biased by protected class?)
   ├── Explainability (can we explain why we declined?)
   ├── Documentation (is every decision logged?)
   └── Change management (who approved this model?)

2. ADVERSARIAL ENVIRONMENT

   Fraudsters actively try to defeat your model.
   This requires:
   ├── Security Engineer: Protects model from extraction
   ├── Fraud Analyst: Studies attack patterns
   └── Continuous monitoring for model degradation

3. EXTREME LATENCY REQUIREMENTS

   100ms budget. Every millisecond matters.
   Infrastructure is the product.

4. FALSE POSITIVE SENSITIVITY

   Block a legitimate customer = they leave the bank.
   This is different from spam (who cares if you miss spam?)
```

---

## Chapter 2: The Real-Time Architecture Team

The architecture from `IN_PRODUCTION.md` requires specialized roles:

```
WHO BUILDS THE FRAUD SYSTEM
===========================

Referring to the architecture:

┌─────────────────────────────────────────────────────────────────────────────┐
│                              FEATURE COMPUTATION (< 10ms)                   │
│                                                                             │
│   Transaction Features    User Features         Context Features            │
│         │                      │                      │                     │
│         └──────────────────────┴──────────────────────┘                     │
│                              │                                              │
│                       FEATURE STORE                                         │
│                              │                                              │
│                    ◄──── Built by: Feature Platform Team                    │
│                              │                                              │
│   Team Lead: Yuki (Staff Engineer)                                          │
│   ├── Streaming Engineer: builds Flink jobs                                 │
│   ├── Data Engineer: manages offline/online sync                            │
│   ├── Backend Engineer: Redis cluster, latency optimization                 │
│   └── SRE: feature store reliability                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MODEL INFERENCE (< 5ms)                        │
│                                                                             │
│   ┌───────────────┐    ┌───────────────┐    ┌───────────────┐               │
│   │   XGBoost     │    │  Neural Net   │    │   Isolation   │               │
│   │               │    │               │    │   Forest      │               │
│   └───────────────┘    └───────────────┘    └───────────────┘               │
│           │                   │                    │                        │
│           └───────────────────┴────────────────────┘                        │
│                              │                                              │
│                    ◄──── Built by: ML Engineering Team                      │
│                              │                                              │
│   Team Lead: Jorge (ML Engineering Manager)                                 │
│   ├── MLE: XGBoost optimization and training                                │
│   ├── MLE: Neural network training and serving                              │
│   ├── MLE: Anomaly detection models                                         │
│   └── MLOps Engineer: Model deployment, A/B testing                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DECISION LAYER (< 1ms)                         │
│                                                                             │
│   ML Score + Rules + Business Logic                                         │
│                              │                                              │
│                    ◄──── Built by: Rules Engine Team                        │
│                              │                                              │
│   Team Lead: Lisa (Senior Backend)                                          │
│   ├── Backend Engineer: Rules execution engine                              │
│   ├── Backend Engineer: Customer configuration                              │
│   └── Product Manager: Works with banks on rule logic                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**The Fraud Analyst: A Unique Role**

```
THE FRAUD ANALYST
=================

This role doesn't exist at most ML companies.

Fraud Analyst: Samira
├── Background: Former bank fraud investigator
├── NOT a software engineer
├── NOT a data scientist
│
├── What she does:
│   ├── Reviews fraud patterns manually
│   ├── Identifies new attack vectors
│   ├── Proposes new features for models
│   ├── Works with banks on their specific fraud patterns
│   ├── Creates rules for novel fraud types
│   └── Writes fraud intelligence reports
│
├── Who she works with:
│   ├── ML team: "Here's a new pattern, can you detect it?"
│   ├── Rules team: "Add this rule for velocity checking"
│   ├── Customers: "Here's what we're seeing in your data"
│   └── Compliance: "This is why the model flagged this"
│
└── Why she's critical:
    Models learn from history. Samira anticipates the future.
    Fraudsters change tactics. Samira catches new patterns
    before they're in the training data.


FRAUD TEAM STRUCTURE:

             Dr. Fatima (Chief Scientist)
                        │
        ┌───────────────┼───────────────┐
        │               │               │
   ML Research      Fraud Intel      Model Risk
        │               │               │
    3 Research       Samira          2 Analysts
    Scientists       + 2 Jr Analysts  (compliance)
        │               │               │
    New models      Pattern       Bias testing,
    Better           detection,    explainability,
    architectures    intelligence  audit prep
```

---

## Chapter 3: The 24/7 Operation

Fraud detection runs 24/7. So does the team.

```
THE FOLLOW-THE-SUN MODEL
========================

ShieldPay operates globally. Fraud never sleeps.

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   US (San Francisco)      Europe (London)       Asia (Singapore)            │
│   8 AM - 6 PM PST         8 AM - 6 PM GMT       8 AM - 6 PM SGT            │
│                                                                             │
│   Primary team:           Primary team:         Primary team:               │
│   - Engineering           - SRE                 - SRE                       │
│   - ML Development        - Fraud Analysts      - Fraud Analysts            │
│   - Product               - Support             - Support                   │
│                                                                             │
│   During their hours:     During their hours:   During their hours:         │
│   - Feature development   - Monitoring          - Monitoring                │
│   - Model training        - Incident response   - Incident response         │
│   - Customer calls (US)   - Customer calls (EU) - Customer calls (APAC)     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

INCIDENT RESPONSE:

Severity 1 (fraud attack in progress):
├── Singapore SRE detects anomaly at 3 AM PST
├── Pages Singapore fraud analyst
├── Analyst confirms attack pattern
├── Deploys emergency rule (pre-approved)
├── Logs incident for US team review in morning

Severity 2 (model degradation):
├── Automated alert fires
├── On-call SRE investigates
├── If model issue: pages ML on-call
├── If infrastructure: handles directly

The key: Empower regional teams to act without waiting for US.
```

**The Model Risk Team**

Financial services require Model Risk Management (MRM):

```
MODEL RISK MANAGEMENT
=====================

Required by regulators (OCC, Fed) for any model used in
credit decisions or fraud detection.

Model Risk Manager: Dr. Chen
├── Background: PhD Statistics, former bank MRM
├── Reports to: Chief Scientist (independence from engineering)
│
├── What he does:
│   ├── Reviews all models before production
│   ├── Validates model assumptions
│   ├── Tests for disparate impact (bias)
│   ├── Documents model limitations
│   ├── Maintains model inventory
│   └── Prepares for regulatory exams
│
├── Model Approval Process:
│   │
│   │   ML team develops model
│   │           │
│   │           ▼
│   │   Model review request
│   │           │
│   │           ▼
│   │   Dr. Chen reviews:
│   │   - Development documentation
│   │   - Validation testing
│   │   - Bias analysis
│   │   - Ongoing monitoring plan
│   │           │
│   │           ▼
│   │   ┌───────┴───────┐
│   │   │               │
│   │ Approved       Rejected
│   │   │            (with findings)
│   │   ▼               │
│   │ Production        │
│   │                   │
│   │   ◄───────────────┘
│   │   Fix issues and resubmit
│
└── This process adds 2-4 weeks to any model change.
    This is the cost of operating in regulated industries.
```

---

## Chapter 4: The Mature Fraud Organization

Year 5: ShieldPay processes $200B in transactions annually for 50 banks.

```
SHIELDPAY ORG (MATURE)
======================

                                    Board
                                      │
                                Marcus (CEO)
                                      │
        ┌───────────┬─────────────────┼─────────────────┬───────────┐
        │           │                 │                 │           │
      Ravi        Fatima           VP Sales         VP Customer    CFO
     (CTO)     (Chief Scientist)                    Success
        │           │                 │                 │
   ┌────┴────┐  ┌───┴───┐            │           ┌─────┴─────┐
   │         │  │       │            │           │           │
Platform  Product ML    Fraud      Sales      Solutions   Support
   │         │  │       Intel       (12)      Engineering  (15)
   │         │  │       │                        (8)
   │         │  │       │
┌──┴──┐   (3 PMs)  │   ┌──┴──┐
│     │        │   │   │     │
Infra  ML      │   │ Fraud  Model
(12)  Platform │   │ Analysts Risk
      (8)      │   │  (6)    (3)
               │   │
           ┌───┴───┴───┐
           │           │
        ML Eng     ML Research
         (12)        (6)


TOTAL HEADCOUNT: ~150

Engineering + ML: 45 (30%)
Fraud Intelligence + Risk: 9 (6%)
Product: 5 (3%)
Sales + Solutions: 20 (13%)
Customer Success + Support: 20 (13%)
Operations + Finance + Legal + HR: 51 (34%)

Note: Only 30% are "engineering"!
This is normal for a mature B2B company.


ROLES SPECIFIC TO FRAUD/FINTECH:

Compliance Officer:
├── Ensures regulatory compliance
├── Manages bank audits
├── Oversees MRM function
├── Regulatory relationship management

Bank Integration Engineer (in Solutions):
├── Understands bank core systems
├── Implements data feeds
├── Manages bank-specific configurations
├── Often former bank technologist

Fraud Intelligence Analyst (in Fraud Intel):
├── Investigates fraud rings
├── Collaborates with law enforcement
├── Shares intelligence across banks (anonymized)
├── Industry conference participation

Model Validator (in Model Risk):
├── Independent model testing
├── Statistical validation
├── Bias testing
├── Challenge function to ML team
```

---

# Summary: The Roles That Make ML Systems Work

```
ROLES BEYOND SOFTWARE ENGINEER
==============================

TECHNICAL ROLES:

ML Research Scientist
├── Develops new model architectures
├── Publishes papers, attends conferences
├── PhD typically required
├── Long-term research horizon

ML Engineer
├── Productionizes models
├── Optimization (latency, memory)
├── Feature engineering
├── Model serving infrastructure

MLOps Engineer
├── CI/CD for ML
├── Experiment tracking
├── Model registry
├── Training infrastructure

Data Engineer
├── Data pipelines
├── Feature stores
├── Data quality
├── ETL and ELT

Analytics Engineer
├── Business metrics
├── A/B test analysis
├── Dashboards
├── Data modeling

Platform/Infrastructure Engineer
├── Kubernetes, cloud
├── Databases, caching
├── Networking, security
├── Developer experience

Embedded/Edge Engineer (manufacturing)
├── Edge devices
├── Real-time systems
├── Hardware integration
├── Firmware


DOMAIN ROLES:

Fraud Analyst (finance)
├── Pattern investigation
├── Rule creation
├── Intelligence sharing
├── Attack anticipation

Quality Inspector / Annotator Lead (manufacturing)
├── Defect taxonomy
├── Labeling quality
├── Domain expertise encoding
├── Edge case identification

Document Processor (document AI)
├── Human review
├── Correction feedback
├── Quality assurance
├── Training data generation

Solutions Engineer (all B2B)
├── Technical sales
├── Customer integration
├── Requirements gathering
├── Proof of concept delivery


OPERATIONAL ROLES:

Operations Manager
├── Team management
├── Quality metrics
├── Process optimization
├── Workforce planning

Field Technician (manufacturing)
├── On-site installation
├── Hardware troubleshooting
├── Operator training
├── First-line support

Customer Onboarding Specialist
├── New customer setup
├── Configuration
├── Training
├── Early success


GOVERNANCE ROLES:

Model Risk Manager (finance)
├── Model validation
├── Bias testing
├── Regulatory compliance
├── Audit preparation

Compliance Officer
├── Regulatory requirements
├── Policy development
├── Audit management
├── Risk assessment


PRODUCT ROLES:

Product Manager
├── Roadmap
├── Prioritization
├── Customer voice
├── Success metrics

Product Analyst
├── Deep-dive analysis
├── Feature impact
├── User behavior
├── Experimentation


THE RATIO VARIES BY STAGE:

Early stage (10 people):
├── 70% engineers
├── 20% domain experts
├── 10% business

Growth stage (50 people):
├── 50% engineers
├── 20% operations
├── 15% domain experts
├── 15% business

Mature (200 people):
├── 30% engineers
├── 25% operations
├── 15% domain experts
├── 30% business/support
```

---

## The Hiring Sequence

```
WHEN TO HIRE EACH ROLE
======================

PHASE 1: MVP (1-5 people)
├── Founders wear all hats
├── First hire: generalist engineer
├── Domain expert (even if part-time)
└── No specialists yet

PHASE 2: First Production (5-15 people)
├── ML Engineer (model optimization)
├── Data Engineer (pipelines)
├── DevOps/SRE (reliability)
├── First Product Manager
└── First Sales hire

PHASE 3: Scaling (15-50 people)
├── Specialized MLEs (research vs production)
├── Operations team (if human-in-loop)
├── Solutions Engineering (if B2B)
├── Analytics/Data Science
├── More PMs
└── Customer Success

PHASE 4: Mature (50-200 people)
├── Managers for each function
├── Compliance/Risk (if regulated)
├── Training/Enablement
├── Full executive team
├── International teams
└── Specialized support roles


THE HIRING MISTAKE TO AVOID:

Early stage: Hiring specialists too early
             (You don't need an "MLOps Engineer" at 10 people)

Growth stage: Not hiring operations/domain experts
              (Engineers can't do everything)

Mature stage: Not developing management capability
              (ICs don't automatically become managers)
```

---

## The Organizational Patterns

```
PATTERN 1: THE PLATFORM + POD MODEL
====================================

Works for: Multi-customer B2B with customization needs
Examples: VisionGuard (manufacturing), ShieldPay (fraud)

     ┌─────────────────────────────────────┐
     │           PLATFORM TEAM             │
     │   (shared infrastructure)           │
     └───────────────────┬─────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    ┌─────────┐    ┌─────────┐    ┌─────────┐
    │ Pod A   │    │ Pod B   │    │ Pod C   │
    │ (Auto)  │    │ (Elec)  │    │ (Food)  │
    └─────────┘    └─────────┘    └─────────┘

Benefits:
- Domain expertise concentrates
- Customer relationships deepen
- Platform evolves from pod needs


PATTERN 2: THE RESEARCH + PRODUCTION SPLIT
==========================================

Works for: Companies where model innovation is competitive advantage
Examples: StyleMatch (visual search)

     ┌─────────────────────────────────────┐
     │         ML RESEARCH                 │
     │   (long-term, exploratory)          │
     └───────────────────┬─────────────────┘
                         │ Model handoff
                         ▼
     ┌─────────────────────────────────────┐
     │         ML ENGINEERING              │
     │   (production, reliability)         │
     └─────────────────────────────────────┘

Benefits:
- Researchers free to explore
- Production quality maintained
- Clear ownership boundaries


PATTERN 3: THE HUMAN-IN-THE-LOOP OPERATION
==========================================

Works for: High-accuracy requirements with human review
Examples: DocuFlow (documents)

     ┌─────────────────────────────────────┐
     │         ENGINEERING                 │
     │   (automation)                      │
     └───────────────────┬─────────────────┘
                         │
                         ▼
     ┌─────────────────────────────────────┐
     │         OPERATIONS                  │
     │   (human review)                    │
     └───────────────────┬─────────────────┘
                         │ Corrections
                         ▼
     ┌─────────────────────────────────────┐
     │         ML IMPROVEMENT              │
     │   (learn from humans)               │
     └─────────────────────────────────────┘

Benefits:
- Quality guaranteed by humans
- Models continuously improve
- Clear escalation path
```

---

## Final Thoughts: The CTO's Checklist

```
BUILDING AN ML ORGANIZATION
===========================

□ START WITH DOMAIN EXPERTISE
  - Hire people who understand the problem
  - Domain experts > ML experts initially
  - The best model can't fix wrong problem framing

□ GENERALISTS FIRST, SPECIALISTS LATER
  - Early hires wear multiple hats
  - Specialize as team grows
  - Don't hire MLOps engineer #1 at 10 people

□ OPERATIONS IS NOT AN AFTERTHOUGHT
  - Human-in-the-loop needs humans
  - Annotation quality determines model quality
  - Field teams are first-class citizens

□ SEPARATE RESEARCH FROM PRODUCTION
  - At ~15 ML people, split the functions
  - Researchers need freedom to explore
  - Production needs reliability focus

□ HIRE FOR THE DOMAIN
  - Fraud: hire former fraud investigators
  - Manufacturing: hire quality inspectors
  - Documents: hire AP clerks
  - They know what you're trying to automate

□ INVEST IN GOVERNANCE EARLY (IF REGULATED)
  - Model risk management isn't optional
  - Compliance is a feature, not a tax
  - Better early than retrofitting

□ BUILD THE FEEDBACK LOOP
  - Human corrections → training data
  - Customer escalations → model improvements
  - On-call insights → reliability improvements

□ REMEMBER: ML IS 10% OF THE SYSTEM
  - Data pipelines, serving, monitoring matter more
  - Hire accordingly
  - Don't overweight ML researchers
```

---

*This document tells the story of building ML organizations. The lesson:
successful ML companies are not just collections of ML engineers.
They're cross-functional teams where domain experts, operations staff,
and engineers work together to build systems that actually work in
the real world.*
