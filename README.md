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
 LabelCalibrator                     calibrate + build + filter + weight
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
              StudentTrainer / TinyStudent / TransformerStudent
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
│   │   ├── base.py                   abstract Teacher and shared projection
│   │   ├── callable.py               application-owned callback adapter
│   │   ├── openai_teacher.py         ChatGPT / Responses API
│   │   ├── anthropic_teacher.py      Claude / Messages API
│   │   ├── deepseek_teacher.py       DeepSeek / Chat Completions API
│   │   ├── huggingface_teacher.py    Llama, T5, and Qwen3.5
│   │   └── rule_based.py             deterministic arithmetic teacher
│   ├── generate_reasoning/
│   │   ├── base.py                   abstract strategy contract
│   │   ├── strategies.py             built-in reasoning behaviors
│   │   └── generator.py              candidate orchestration
│   ├── score/
│   │   ├── base.py                   abstract scoring contract
│   │   ├── scorer.py                 weighted composite scorer
│   │   └── strategies.py             specialized scoring variants
│   ├── calibrated_labels/
│   │   ├── base.py                   four abstract strategy contracts
│   │   ├── calibration.py            identity and temperature calibration
│   │   ├── label_builders.py         teacher and ground-truth targets
│   │   ├── filtering.py              acceptance and uncertainty filters
│   │   ├── weighting.py              score and uncertainty weights
│   │   └── calibrator.py             config and strategy orchestration
│   ├── student_training/
│   │   ├── models.py                 abstract, GRU, and Transformer students
│   │   ├── losses.py                 modular distillation objectives
│   │   ├── schedules.py              static and curriculum loss weights
│   │   └── student_training.py       trainer, data, and experiment state
│   ├── inference/
│   │   └── speculative_decoding.py   draft/verify sampling
│   ├── evaluation/
│   │   ├── base.py                   metric context and abstract contract
│   │   ├── classification.py         task, confidence, and calibration
│   │   ├── reasoning.py              rationale comparison metrics
│   │   └── metrics.py                orchestration and evaluation report
│   ├── pipeline.py                   stage orchestration
│   ├── demo.py                       offline comparison experiment
│   └── __main__.py                   command-line entry point
└── tests/
    ├── test_calibrated_labels.py
    ├── test_evaluation_metrics.py
    ├── test_pipeline.py
    ├── test_reasoning_strategies.py
    ├── test_scoring_strategies.py
    ├── test_speculative_decoding.py
    ├── test_student_training.py
    └── test_teachers.py
```

Each subpackage exports its public types through its own `__init__.py`. The
top-level package re-exports the main API, so application code can continue to
use `from tiny_distillation import ...`. Internal modules also use absolute
package imports, for example `from tiny_distillation.core import TrainingExample`.

## Reasoning strategies

`ReasoningGenerationConfig.strategy` controls how the teacher creates its
supervision trace:

| Strategy | Behavior |
| --- | --- |
| `direct` | Final answer without a rationale |
| `rationale` | Concise, self-contained justification |
| `step_by_step` | Numbered and verifiable solution steps |
| `answer_then_rationale` | Commit to the answer before explaining |
| `critique_revision` | Draft once, then critique and revise in a second call |
| `self_consistency` | Produce independently prompted candidates |

```python
from tiny_distillation import (
    DistillationPipeline,
    ReasoningGenerationConfig,
)

pipeline = DistillationPipeline(
    teacher,
    generation_config=ReasoningGenerationConfig(
        strategy="self_consistency",
        candidates_per_example=5,
        custom_instruction="Keep the rationale under four sentences.",
        deduplicate_candidates=True,
    ),
)
```

`include_reasoning` normally follows the strategy: it is disabled for `direct`
and enabled for the others. Setting it explicitly overrides that default.
Candidate ranking and acceptance still happen in the `score` stage.

Custom behavior inherits `ReasoningStrategy` and implements
`build_instruction`:

```python
from tiny_distillation import ReasoningStrategy

class EvidenceFirstStrategy(ReasoningStrategy):
    name = "evidence_first"

    def build_instruction(self, example, candidate_index):
        return "List the decisive evidence before giving the final answer."

config = ReasoningGenerationConfig(strategy=EvidenceFirstStrategy())
```

## Scoring strategies

All scorers inherit `ScoringStrategy` and produce the same
`ScoredPrediction` contract:

| Scorer | Total-score source | Useful for |
| --- | --- | --- |
| `CompositeScorer` | Answer, reasoning, and confidence weights | General default |
| `ExactAnswerScorer` | Reference-answer agreement | Tasks with trusted labels |
| `ConfidenceScorer` | Teacher confidence | Confidence ablations |
| `ReasoningQualityScorer` | Rationale quality heuristics | CoT filtering |
| `RewardScorer` | External verifier callback | Code tests, judges, reward models |
| `ConsensusScorer` | Base quality plus candidate agreement | Self-consistency |

Consensus scoring is most useful with multiple independently generated
candidates:

```python
from tiny_distillation import (
    CompositeScorer,
    ConsensusScorer,
    ConsensusScoringConfig,
)

pipeline = DistillationPipeline(
    teacher,
    generation_config=ReasoningGenerationConfig(
        strategy="self_consistency",
        candidates_per_example=5,
    ),
    scorer=ConsensusScorer(
        base_scorer=CompositeScorer(),
        config=ConsensusScoringConfig(
            consensus_weight=0.4,
            acceptance_threshold=0.65,
        ),
    ),
)
```

For task-specific verification, make the callback return a normalized score
from zero to one:

```python
from tiny_distillation import RewardScorer

scorer = RewardScorer(
    lambda prediction, example: run_task_verifier(prediction.answer),
    acceptance_threshold=0.8,
)
```

`best_per_example` is implemented by the shared base class, so every strategy
works with `DistillationPipeline` without special orchestration.

## Calibrated labels

The calibrated-label stage has four independently replaceable steps:

```text
teacher logits -> probability calibration -> quality filter
               -> hard/soft target builder -> sample weighting
               -> CalibratedLabel
```

Each step shares an abstract contract:

| Step | Base class | Built-in strategies |
| --- | --- | --- |
| Probability calibration | `CalibrationStrategy` | `TemperatureCalibration`, `IdentityCalibration` |
| Target construction | `LabelBuilder` | `TeacherLabelBuilder`, `GroundTruthBlendLabelBuilder` |
| Quality filtering | `LabelFilter` | `AcceptedLabelFilter`, `QualityLabelFilter`, `CompositeLabelFilter` |
| Sample weighting | `WeightingStrategy` | Score, confidence, inverse entropy, or top-two margin |

`CalibrationConfig` selects the built-ins without requiring custom classes:

```python
from tiny_distillation import CalibrationConfig, LabelCalibrator

calibrator = LabelCalibrator(
    CalibrationConfig(
        calibration_method="temperature",
        label_building_method="ground_truth_blend",
        ground_truth_weight=0.25,
        label_smoothing=0.05,
        top_k=4,
        minimum_confidence=0.65,
        maximum_entropy=0.8,
        minimum_margin=0.1,
        weighting_method="entropy",
        minimum_weight=0.1,
    )
)
```

Temperature fitting uses labeled examples to minimize score-weighted negative
log-likelihood. It leaves teacher ranking unchanged but adjusts how sharp the
soft targets are. `identity` skips fitting and applies standard softmax.

The teacher target builder uses the calibrated argmax for hard CE and the full
distribution for soft KL. It can retain only the top-k classes and apply label
smoothing. Ground-truth blending keeps a trusted dataset label as the hard
target and mixes its one-hot distribution into the teacher soft target.

Quality filters can reject labels based on the scorer's acceptance decision,
maximum calibrated probability, normalized entropy, and the probability margin
between the top two classes. Retained labels can then be weighted by source
score, confidence, inverse entropy, or margin. Strategy names are recorded in
each `CalibratedLabel.metadata` mapping for experiment provenance.

Application-specific behavior can be injected directly:

```python
calibrator = LabelCalibrator(
    calibration_strategy=my_calibration,
    label_builder=my_label_builder,
    label_filter=my_filter,
    weighting_strategy=my_weighting,
)
```

## Student training

`StudentTrainer` accepts any `StudentModel` implementation. The package
includes the compact GRU-based `TinyStudent` and a `TransformerStudent` prompt
encoder with the same answer-conditioned rationale decoder.

Each objective inherits `DistillationLoss`:

| Mode | Strategy | Objective |
| --- | --- | --- |
| `hard` | `HardCrossEntropyLoss` | Cross-entropy against the hard target |
| `soft` | `SoftKLDistillationLoss` | KL divergence against calibrated teacher probabilities |
| `cot` | `CoTTokenCrossEntropyLoss` | Masked rationale-token cross-entropy |
| `combined` | `CombinedDistillationLoss` | Configurable weighted sum of all objectives |

Soft-label temperature is applied to both distributions. Stored teacher
probabilities are converted with `softmax(log(probabilities) / temperature)`,
while student logits are divided by the same temperature. This keeps KL
temperature experiments internally consistent even after label calibration.

```python
from tiny_distillation import (
    SamplingMethod,
    StudentTrainer,
    TrainerConfig,
    TransformerStudent,
    Vocabulary,
)

vocabulary = Vocabulary.from_labels(artifacts.labels)
student = TransformerStudent(
    len(vocabulary),
    num_labels=number_of_labels,
    embedding_dim=128,
    hidden_dim=128,
    num_heads=4,
    pad_id=vocabulary.pad_id,
)
trainer = StudentTrainer(
    student,
    vocabulary,
    TrainerConfig(
        mode="combined",
        epochs=20,
        validation_fraction=0.15,
        early_stopping_patience=3,
        warmup_epochs=2,
        weight_decay=0.01,
        gradient_accumulation_steps=4,
        mixed_precision=True,
        sampling_method=SamplingMethod.CLASS_BALANCED,
        cot_warmup_epochs=5,
        teacher_forcing_ratio=1.0,
        final_teacher_forcing_ratio=0.5,
        max_reasoning_tokens=128,
        checkpoint_path="artifacts/student.pt",
        experiment_log_path="artifacts/experiment.json",
    ),
)
history = trainer.train(artifacts.labels)
```

Training supports:

- deterministic train/validation splitting, best-state restoration, and early
  stopping;
- AdamW weight decay, learning-rate warmup plus cosine decay, gradient
  accumulation, clipping, and CUDA mixed precision;
- shuffled, quality-weighted, or class-balanced sampling;
- linear CoT-loss warmup and scheduled teacher forcing;
- answer-conditioned reasoning, rationale truncation, and explicit token masks;
- checkpoint and JSON experiment records containing configuration, seed,
  per-loss curves, learning rates, label-calibration metadata, and evaluation
  metrics.

`DistillationPipeline.evaluate` automatically records its metrics in the
trainer's configured checkpoint and experiment log. For evaluation performed
outside the pipeline, call `trainer.record_evaluation(metrics)`.

Custom loss combinations can be passed through `loss_strategy`, and custom
curricula can inherit `LossWeightScheduler`:

```python
trainer = StudentTrainer(
    student,
    vocabulary,
    config,
    loss_strategy=my_loss,
    loss_weight_scheduler=my_schedule,
)
```

Mixed precision activates only when the configured device is CUDA. It remains
disabled on CPU even when `mixed_precision=True`.

## Evaluation metrics

Every metric inherits `EvaluationMetric` and consumes a shared
`EvaluationContext`.

| Category | Metrics |
| --- | --- |
| Classification | Accuracy, top-k accuracy, macro precision, recall, and F1 |
| Probabilistic | Negative log-likelihood and Brier score |
| Calibration | Expected and maximum calibration error |
| Uncertainty | Mean confidence and predictive entropy |
| Reasoning | Exact match and token precision, recall, and F1 |

`evaluate_classification` computes all applicable built-ins. Reasoning metrics
are included when generated and reference traces are supplied:

```python
report = evaluate_classification(
    logits,
    targets,
    top_k=3,
    num_bins=15,
    generated_reasoning=student_reasoning,
    reference_reasoning=teacher_reasoning,
)

print(report.macro_f1)
print(report.maximum_calibration_error)
print(report.reasoning_token_recall)
```

Custom metrics implement one scalar computation and can be added without
changing `EvaluationReport`:

```python
from tiny_distillation import EvaluationMetric

class AbstentionRateMetric(EvaluationMetric):
    name = "abstention_rate"

    def compute(self, context):
        return float(context.confidence.lt(0.6).float().mean())

report = evaluate_classification(
    logits,
    targets,
    additional_metrics=[AbstentionRateMetric()],
)
print(report["abstention_rate"])
```

Every computed value is also available through `report.metric_values`.

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

Every implementation inherits the abstract `Teacher` class. The base class
builds the structured prompt, parses the provider response, validates the
answer, and converts it to a `TeacherPrediction`. Subclasses only implement the
backend request.

Install only the provider integrations needed by the experiment:

```bash
# ChatGPT, Claude, and DeepSeek
.venv/bin/python -m pip install -e '.[api]'

# Llama, T5, and Qwen3.5 through Transformers
.venv/bin/python -m pip install -e '.[hf]'

# Every example teacher
.venv/bin/python -m pip install -e '.[teachers]'
```

Hosted adapters use `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and
`DEEPSEEK_API_KEY`, respectively. Clients can also be constructed by the
application and passed through the `client` argument.

```python
from tiny_distillation import (
    ChatGPTTeacher,
    ClaudeTeacher,
    DeepSeekTeacher,
    DistillationPipeline,
    LlamaTeacher,
    Qwen35Teacher,
    T5Teacher,
)

labels = ["A", "B", "C", "D"]

teachers = {
    "chatgpt": ChatGPTTeacher(labels),
    "claude": ClaudeTeacher(labels),
    "deepseek": DeepSeekTeacher(labels),
    "llama": LlamaTeacher(labels),
    "t5": T5Teacher(labels),
    "qwen3.5": Qwen35Teacher(labels),
}

pipeline = DistillationPipeline(teachers["chatgpt"])
artifacts = pipeline.prepare(training_examples)
```

The defaults are examples, not fixed requirements. Override `model`, inject a
configured SDK client, or supply already-loaded Transformers objects:

```python
teacher = Qwen35Teacher(
    labels,
    model="Qwen/Qwen3.5-2B",
    model_kwargs={"device_map": "auto", "dtype": "auto"},
    generation_kwargs={"do_sample": True, "temperature": 0.8},
)
```

Hosted chat APIs generally return generated text rather than logits over an
application-defined label space. By default, `Teacher` requires the generated
answer to exactly match a configured label and turns the model's confidence
into a categorical distribution. For free-form answers, provide a
`label_projector` that returns one logit per label.

The original `CallableTeacher` remains available when the application already
produces a complete `TeacherPrediction` containing:

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

The adapters follow the official [OpenAI Responses API](https://developers.openai.com/api/docs/guides/text),
[Anthropic Python SDK](https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/python),
[DeepSeek API](https://api-docs.deepseek.com/), and
[Transformers generation](https://huggingface.co/docs/transformers/main_classes/text_generation)
interfaces. Local chat models use their tokenizer's chat template; Qwen3.5 uses
the official Transformers-compatible checkpoint.

`PipelineArtifacts.labels` carries all target forms at once, so experiments use
the same teacher data and differ only by `TrainerConfig.mode`.

CoT traces can contain sensitive data or teacher mistakes. Score and sanitize
them before training in production; the scorer's optional `reward_fn` is the
extension point for a task-specific verifier or safety filter.
