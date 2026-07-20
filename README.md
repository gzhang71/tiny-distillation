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
│   │   └── calibrator.py             temperature and target calibration
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
    ├── test_reasoning_strategies.py
    ├── test_scoring_strategies.py
    ├── test_speculative_decoding.py
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
