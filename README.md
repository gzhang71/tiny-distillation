# tiny-distillation

A compact, modular reference pipeline for distilling a teacher model into a
small PyTorch student. It supports hard labels, calibrated soft labels,
chain-of-thought (CoT) traces, combined training, and exact speculative
sampling.

## Workflow architecture

```text
                         TRAINING-TIME PIPELINE

 TrainingExample
       |
       v
 Teacher.generate()  <---------------- API model / local model / rule teacher
       |
       v
 generate_reasoning()                one or more candidates per example
       |
       v
 TeacherPrediction[]
       |
       v
 CompositeScorer                     correctness + reasoning + confidence
       |
       v
 best candidate / threshold filter
       |
       v
 LabelCalibrator                     fit temperature + quality weights
       |
       v
 CalibratedLabel[]
       |
       +--------------+---------------+----------------+
       |              |               |                |
       v              v               v                v
   hard CE         soft KL        CoT token CE      combined loss
       |              |               |                |
       +--------------+---------------+----------------+
                              |
                              v
                    StudentTrainer / TinyStudent
                              |
                    +---------+---------+
                    |                   |
                    v                   v
              classification      reasoning decoder
                    |
                    v
        accuracy / NLL / Brier / ECE / reasoning metrics

                         INFERENCE-TIME PATH

 TinyStudent (draft) ---- proposed token block ----+
                                                  v
 Target model ---------------------------- batched verification
                                                  |
                                                  v
                              accepted tokens or residual resample
```

`PipelineArtifacts` is the boundary between data preparation and student
training. It retains the raw teacher predictions, selected scores, calibrated
labels, and fitted temperature so experiments can reuse identical teacher data.
Speculative decoding is independent of training and can wrap any draft and
target models implementing the autoregressive interface.

## Directory overview

```text
tiny-distillation/
├── .gitignore
├── pyproject.toml
├── README.md
├── tiny_distillation/
│   ├── core/
│   │   ├── types.py                  shared pipeline records
│   │   └── math_utils.py             softmax, argmax, and clamping
│   ├── teachers/
│   │   └── teacher.py                protocol, adapter, demo teacher
│   ├── preparation/
│   │   ├── generate_reasoning.py     multi-candidate trace generation
│   │   ├── score.py                  quality scoring and filtering
│   │   ├── socre.py                  original-spelling compatibility import
│   │   └── calibrated_labels.py      temperature and target calibration
│   ├── training/
│   │   └── student_training.py       student model and four objectives
│   ├── inference/
│   │   └── speculative_decoding.py   draft/verify sampling
│   ├── evaluation/
│   │   └── metrics.py                task, calibration, and CoT metrics
│   ├── pipeline.py                   stage orchestration
│   ├── demo.py                       offline comparison experiment
│   └── __main__.py                   command-line entry point
└── tests/
    ├── test_pipeline.py
    └── test_speculative_decoding.py
```

Each subpackage exports its public types through its own `__init__.py`. The
top-level package re-exports the main API, so application code can continue to
use `from tiny_distillation import ...`. Internal modules also use absolute
package imports, for example `from tiny_distillation.core import TrainingExample`.

## Run the experiment

The bundled arithmetic teacher keeps the demo deterministic and offline:

```bash
.venv/bin/python -m pip install -e .
.venv/bin/python -m tiny_distillation --mode all --epochs 12
```

Run one strategy at a time with `--mode hard`, `soft`, `cot`, or `combined`.
The JSON result compares loss, accuracy, calibration, reasoning token F1, and
speculative-decoding acceptance statistics.

The flat package can run directly from the repository root. Using this
repository's virtual environment:

```bash
.venv/bin/python -m tiny_distillation --mode all --epochs 12
.venv/bin/python -m unittest discover -s tests -v
```

## Use with a real teacher

Adapt an API or local model with `CallableTeacher`. The callback returns a
`TeacherPrediction` containing:

- final answer text;
- a reasoning trace;
- logits over the student's label space;
- the teacher confidence.

```python
from tiny_distillation import (
    CallableTeacher,
    DistillationPipeline,
    TeacherPrediction,
)

def call_teacher(example, include_reasoning, candidate_index):
    response = your_model.generate(example.prompt)
    return TeacherPrediction(
        example_id=example.id,
        prompt=example.prompt,
        answer=response.answer,
        reasoning=response.reasoning if include_reasoning else "",
        logits=tuple(response.label_logits),
        confidence=response.confidence,
        candidate_index=candidate_index,
    )

pipeline = DistillationPipeline(CallableTeacher(call_teacher))
artifacts = pipeline.prepare(training_examples)
```

`PipelineArtifacts.labels` carries all target forms at once, so experiments use
the same teacher data and differ only by `TrainerConfig.mode`.

CoT traces can contain sensitive data or teacher mistakes. Score and sanitize
them before training in production; the scorer's optional `reward_fn` is the
extension point for a task-specific verifier or safety filter.
