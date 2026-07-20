"""Configurable orchestration for student-model distillation."""

from __future__ import annotations

import copy
import json
import math
import random
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import (
    DataLoader,
    Dataset,
    WeightedRandomSampler,
)

from tiny_distillation.core.types import CalibratedLabel, TrainingHistory
from tiny_distillation.student_training.losses import (
    CoTTokenCrossEntropyLoss,
    CombinedDistillationLoss,
    DistillationLoss,
    DistillationLossOutput,
    HardCrossEntropyLoss,
    LossWeights,
    SoftKLDistillationLoss,
)
from tiny_distillation.student_training.models import StudentModel
from tiny_distillation.student_training.schedules import (
    LinearCoTCurriculum,
    LossWeightScheduler,
    StaticLossWeightScheduler,
)


class DistillationMode(str, Enum):
    HARD = "hard"
    SOFT = "soft"
    COT = "cot"
    COMBINED = "combined"


class SamplingMethod(str, Enum):
    SHUFFLE = "shuffle"
    QUALITY = "quality"
    CLASS_BALANCED = "class_balanced"


class Vocabulary:
    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    UNK = "<unk>"
    _TOKEN_PATTERN = re.compile(r"\d+|[A-Za-z]+|[^\w\s]")

    def __init__(self, tokens: Iterable[str]) -> None:
        ordered = [self.PAD, self.BOS, self.EOS, self.UNK]
        ordered.extend(sorted(set(tokens) - set(ordered)))
        self.token_to_id = {token: index for index, token in enumerate(ordered)}
        self.id_to_token = tuple(ordered)

    @classmethod
    def build(cls, texts: Iterable[str]) -> "Vocabulary":
        return cls(token for text in texts for token in cls.tokenize(text))

    @classmethod
    def from_labels(cls, labels: Iterable[CalibratedLabel]) -> "Vocabulary":
        items = list(labels)
        return cls.build(
            text
            for item in items
            for text in (item.prompt, item.reasoning, item.answer)
        )

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        return cls._TOKEN_PATTERN.findall(text.lower())

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.PAD]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.BOS]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.EOS]

    def __len__(self) -> int:
        return len(self.id_to_token)

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        ids = [
            self.token_to_id.get(token, self.token_to_id[self.UNK])
            for token in self.tokenize(text)
        ]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int], *, skip_special: bool = True) -> str:
        specials = {self.PAD, self.BOS, self.EOS, self.UNK}
        tokens = [
            self.id_to_token[index]
            for index in ids
            if not skip_special or self.id_to_token[index] not in specials
        ]
        return " ".join(tokens)


class _DistillationDataset(Dataset[CalibratedLabel]):
    def __init__(self, labels: Sequence[CalibratedLabel]) -> None:
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> CalibratedLabel:
        return self.labels[index]


@dataclass(frozen=True)
class TrainerConfig:
    mode: DistillationMode | str = DistillationMode.COMBINED
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 3e-3
    weight_decay: float = 1e-2
    warmup_epochs: int = 0
    soft_temperature: float = 1.0
    hard_weight: float = 1.0
    soft_weight: float = 1.0
    cot_weight: float = 0.5
    cot_warmup_epochs: int = 0
    gradient_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = False
    validation_fraction: float = 0.0
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    restore_best_model: bool = True
    sampling_method: SamplingMethod | str = SamplingMethod.SHUFFLE
    teacher_forcing_ratio: float = 1.0
    final_teacher_forcing_ratio: float = 1.0
    max_reasoning_tokens: int = 64
    checkpoint_path: str | None = None
    experiment_log_path: str | None = None
    seed: int = 7
    device: str = "cpu"

    def __post_init__(self) -> None:
        _enum_value(DistillationMode, self.mode, "mode")
        _enum_value(SamplingMethod, self.sampling_method, "sampling_method")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.soft_temperature <= 0:
            raise ValueError("learning_rate and soft_temperature must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.warmup_epochs < 0 or self.cot_warmup_epochs < 0:
            raise ValueError("warmup epochs must be non-negative")
        if any(
            weight < 0
            for weight in (self.hard_weight, self.soft_weight, self.cot_weight)
        ):
            raise ValueError("loss weights must be non-negative")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        if self.early_stopping_patience is not None:
            if self.early_stopping_patience < 1:
                raise ValueError("early_stopping_patience must be positive")
            if self.validation_fraction == 0:
                raise ValueError(
                    "early_stopping_patience requires validation_fraction > 0"
                )
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be non-negative")
        for name, value in (
            ("teacher_forcing_ratio", self.teacher_forcing_ratio),
            ("final_teacher_forcing_ratio", self.final_teacher_forcing_ratio),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.max_reasoning_tokens < 1:
            raise ValueError("max_reasoning_tokens must be positive")


class StudentTrainer:
    def __init__(
        self,
        model: StudentModel,
        vocabulary: Vocabulary,
        config: TrainerConfig | None = None,
        *,
        loss_strategy: DistillationLoss | None = None,
        loss_weight_scheduler: LossWeightScheduler | None = None,
    ) -> None:
        self.model = model
        self.vocabulary = vocabulary
        self.config = config or TrainerConfig()
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        self.loss_strategy = loss_strategy or self._default_loss_strategy()
        self.loss_weight_scheduler = (
            loss_weight_scheduler or self._default_loss_weight_scheduler()
        )
        self._last_history: TrainingHistory | None = None
        self._last_labels: tuple[CalibratedLabel, ...] = ()
        self._last_evaluation: dict[str, float] = {}

    def train(self, labels: Sequence[CalibratedLabel]) -> TrainingHistory:
        if not labels:
            raise ValueError("cannot train without calibrated labels")
        self._seed_everything(self.config.seed)
        train_labels, validation_labels = self._split_labels(labels)
        train_loader = self._build_loader(train_labels, training=True)
        validation_loader = (
            self._build_loader(validation_labels, training=False)
            if validation_labels
            else None
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        optimizer_steps_per_epoch = math.ceil(
            len(train_loader) / self.config.gradient_accumulation_steps
        )
        learning_rate_scheduler = self._build_learning_rate_scheduler(
            optimizer,
            optimizer_steps_per_epoch,
        )
        amp_enabled = self.config.mixed_precision and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        totals: list[float] = []
        hard_history: list[float] = []
        soft_history: list[float] = []
        cot_history: list[float] = []
        validation_history: list[float] = []
        learning_rates: list[float] = []
        teacher_forcing_ratios: list[float] = []
        best_loss = math.inf
        best_epoch: int | None = None
        best_state: dict[str, Tensor] | None = None
        epochs_without_improvement = 0
        stopped_early = False

        base_weights = LossWeights(
            hard=self.config.hard_weight,
            soft=self.config.soft_weight,
            cot=self.config.cot_weight,
        )
        for epoch in range(self.config.epochs):
            loss_weights = self.loss_weight_scheduler.weights(
                epoch,
                self.config.epochs,
                base_weights,
            )
            teacher_forcing_ratio = self._teacher_forcing_ratio(epoch)
            train_losses = self._run_epoch(
                train_loader,
                training=True,
                optimizer=optimizer,
                learning_rate_scheduler=learning_rate_scheduler,
                scaler=scaler,
                loss_weights=loss_weights,
                teacher_forcing_ratio=teacher_forcing_ratio,
                amp_enabled=amp_enabled,
            )
            totals.append(train_losses[0])
            hard_history.append(train_losses[1])
            soft_history.append(train_losses[2])
            cot_history.append(train_losses[3])
            teacher_forcing_ratios.append(teacher_forcing_ratio)
            learning_rates.append(float(optimizer.param_groups[0]["lr"]))

            monitored_loss = train_losses[0]
            if validation_loader is not None:
                validation_losses = self._run_epoch(
                    validation_loader,
                    training=False,
                    loss_weights=loss_weights,
                    teacher_forcing_ratio=1.0,
                    amp_enabled=amp_enabled,
                )
                monitored_loss = validation_losses[0]
                validation_history.append(monitored_loss)

            if monitored_loss < best_loss - self.config.early_stopping_min_delta:
                best_loss = monitored_loss
                best_epoch = epoch + 1
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            patience = self.config.early_stopping_patience
            if (
                validation_loader is not None
                and patience is not None
                and epochs_without_improvement >= patience
            ):
                stopped_early = True
                break

        if self.config.restore_best_model and best_state is not None:
            self.model.load_state_dict(best_state)

        history = TrainingHistory(
            losses=tuple(totals),
            hard_losses=tuple(hard_history),
            soft_losses=tuple(soft_history),
            cot_losses=tuple(cot_history),
            validation_losses=tuple(validation_history),
            learning_rates=tuple(learning_rates),
            teacher_forcing_ratios=tuple(teacher_forcing_ratios),
            best_epoch=best_epoch,
            stopped_early=stopped_early,
        )
        self._last_history = history
        self._last_labels = tuple(labels)
        if self.config.checkpoint_path:
            self.save_checkpoint(self.config.checkpoint_path)
        self._write_experiment_log()
        return history

    def _run_epoch(
        self,
        loader: DataLoader[dict[str, Tensor]],
        *,
        training: bool,
        loss_weights: LossWeights,
        teacher_forcing_ratio: float,
        amp_enabled: bool,
        optimizer: torch.optim.Optimizer | None = None,
        learning_rate_scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
        scaler: torch.amp.GradScaler | None = None,
    ) -> tuple[float, float, float, float]:
        if training and (optimizer is None or scaler is None):
            raise ValueError("training epochs require optimizer and scaler")
        self.model.train(training)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        totals = [0.0, 0.0, 0.0, 0.0]

        for batch_index, batch in enumerate(loader):
            with torch.set_grad_enabled(training):
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=amp_enabled,
                ):
                    output = self._losses(
                        batch,
                        loss_weights,
                        teacher_forcing_ratio,
                    )
            if training:
                assert optimizer is not None
                assert scaler is not None
                scaled_loss = (
                    output.total / self.config.gradient_accumulation_steps
                )
                scaler.scale(scaled_loss).backward()
                should_step = (
                    (batch_index + 1) % self.config.gradient_accumulation_steps == 0
                    or batch_index + 1 == len(loader)
                )
                if should_step:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if learning_rate_scheduler is not None:
                        learning_rate_scheduler.step()
            for index, loss in enumerate(
                (output.total, output.hard, output.soft, output.cot)
            ):
                totals[index] += float(loss.detach())
        return (
            totals[0] / len(loader),
            totals[1] / len(loader),
            totals[2] / len(loader),
            totals[3] / len(loader),
        )

    def _losses(
        self,
        batch: dict[str, Tensor],
        loss_weights: LossWeights,
        teacher_forcing_ratio: float,
    ) -> DistillationLossOutput:
        class_logits, reasoning_logits = self.model(
            batch["prompt_ids"],
            batch["prompt_lengths"],
            batch["reasoning_input_ids"],
            batch["hard_labels"],
            teacher_forcing_ratio,
        )
        if reasoning_logits is None:
            raise RuntimeError("reasoning logits are required for distillation loss")
        return self.loss_strategy.compute(
            class_logits,
            reasoning_logits,
            batch,
            soft_temperature=self.config.soft_temperature,
            weights=loss_weights,
        )

    @torch.no_grad()
    def predict(self, prompts: Sequence[str]) -> tuple[Tensor, list[str]]:
        if not prompts:
            return torch.empty((0, self.model.num_labels)), []
        prompt_ids, prompt_lengths = self._encode_prompts(prompts)
        self.model.eval()
        class_logits, _ = self.model(prompt_ids, prompt_lengths)
        reasoning_ids = self.model.generate_reasoning(
            prompt_ids,
            prompt_lengths,
            bos_id=self.vocabulary.bos_id,
            eos_id=self.vocabulary.eos_id,
            max_new_tokens=self.config.max_reasoning_tokens,
        )
        reasoning = [self.vocabulary.decode(row.tolist()) for row in reasoning_ids]
        return class_logits.cpu(), reasoning

    def record_evaluation(self, metrics: Mapping[str, float]) -> None:
        """Attach evaluation metrics to future checkpoints and experiment logs."""

        self._last_evaluation = {
            str(name): float(value) for name, value in metrics.items()
        }
        if self.config.checkpoint_path and self._last_history is not None:
            self.save_checkpoint(self.config.checkpoint_path)
        self._write_experiment_log()

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        history: TrainingHistory | None = None,
        labels: Sequence[CalibratedLabel] | None = None,
        evaluation: Mapping[str, float] | None = None,
    ) -> None:
        """Persist model state and reproducibility metadata."""

        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        selected_history = history or self._last_history
        selected_labels = tuple(labels) if labels is not None else self._last_labels
        selected_evaluation = (
            dict(evaluation) if evaluation is not None else self._last_evaluation
        )
        torch.save(
            {
                "format_version": 1,
                "model_class": type(self.model).__name__,
                "model_state_dict": self.model.state_dict(),
                "trainer_config": _json_safe(asdict(self.config)),
                "loss_strategy": self.loss_strategy.name,
                "loss_weight_scheduler": type(
                    self.loss_weight_scheduler
                ).__name__,
                "vocabulary": self.vocabulary.id_to_token,
                "training_history": (
                    _json_safe(asdict(selected_history))
                    if selected_history is not None
                    else None
                ),
                "label_metadata": self._label_metadata(selected_labels),
                "evaluation": _json_safe(selected_evaluation),
            },
            checkpoint_path,
        )

    def load_model_state(self, path: str | Path) -> None:
        checkpoint = torch.load(
            Path(path),
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def _split_labels(
        self,
        labels: Sequence[CalibratedLabel],
    ) -> tuple[list[CalibratedLabel], list[CalibratedLabel]]:
        items = list(labels)
        if self.config.validation_fraction == 0 or len(items) < 2:
            return items, []
        indexes = list(range(len(items)))
        random.Random(self.config.seed).shuffle(indexes)
        validation_size = min(
            len(items) - 1,
            max(1, round(len(items) * self.config.validation_fraction)),
        )
        validation_indexes = set(indexes[:validation_size])
        return (
            [item for index, item in enumerate(items) if index not in validation_indexes],
            [item for index, item in enumerate(items) if index in validation_indexes],
        )

    def _build_loader(
        self,
        labels: Sequence[CalibratedLabel],
        *,
        training: bool,
    ) -> DataLoader[dict[str, Tensor]]:
        generator = torch.Generator().manual_seed(
            self.config.seed + (0 if training else 1)
        )
        sampler = None
        shuffle = training
        sampling_method = _enum_value(
            SamplingMethod,
            self.config.sampling_method,
            "sampling_method",
        )
        if training and sampling_method is not SamplingMethod.SHUFFLE:
            class_counts = Counter(item.hard_label for item in labels)
            sample_weights = []
            for item in labels:
                weight = max(float(item.weight), 1e-8)
                if sampling_method is SamplingMethod.CLASS_BALANCED:
                    weight /= class_counts[item.hard_label]
                sample_weights.append(weight)
            sampler = WeightedRandomSampler(
                sample_weights,
                num_samples=len(labels),
                replacement=True,
                generator=generator,
            )
            shuffle = False
        return DataLoader(
            _DistillationDataset(labels),
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            generator=generator,
            collate_fn=self._collate,
        )

    def _collate(self, labels: Sequence[CalibratedLabel]) -> dict[str, Tensor]:
        prompt_sequences = [
            self.vocabulary.encode(item.prompt, add_eos=True) for item in labels
        ]
        reasoning_sequences = [
            [
                self.vocabulary.bos_id,
                *self.vocabulary.encode(item.reasoning)[
                    : self.config.max_reasoning_tokens - 1
                ],
                self.vocabulary.eos_id,
            ]
            for item in labels
        ]
        prompt_ids = self._pad(prompt_sequences)
        reasoning_ids = self._pad(reasoning_sequences)
        reasoning_targets = reasoning_ids[:, 1:]
        return {
            "prompt_ids": prompt_ids.to(self.device),
            "prompt_lengths": torch.tensor(
                [len(sequence) for sequence in prompt_sequences],
                device=self.device,
            ),
            "hard_labels": torch.tensor(
                [item.hard_label for item in labels],
                dtype=torch.long,
                device=self.device,
            ),
            "soft_labels": torch.tensor(
                [item.soft_labels for item in labels],
                dtype=torch.float32,
                device=self.device,
            ),
            "weights": torch.tensor(
                [item.weight for item in labels],
                dtype=torch.float32,
                device=self.device,
            ),
            "reasoning_input_ids": reasoning_ids[:, :-1].to(self.device),
            "reasoning_target_ids": reasoning_targets.to(self.device),
            "reasoning_token_mask": reasoning_targets.ne(
                self.vocabulary.pad_id
            ).to(self.device),
        }

    def _encode_prompts(self, prompts: Sequence[str]) -> tuple[Tensor, Tensor]:
        sequences = [
            self.vocabulary.encode(prompt, add_eos=True) for prompt in prompts
        ]
        return (
            self._pad(sequences).to(self.device),
            torch.tensor(
                [len(sequence) for sequence in sequences],
                device=self.device,
            ),
        )

    def _pad(self, sequences: Sequence[Sequence[int]]) -> Tensor:
        max_length = max(len(sequence) for sequence in sequences)
        result = torch.full(
            (len(sequences), max_length),
            self.vocabulary.pad_id,
            dtype=torch.long,
        )
        for row, sequence in enumerate(sequences):
            result[row, : len(sequence)] = torch.tensor(
                sequence,
                dtype=torch.long,
            )
        return result

    def _default_loss_strategy(self) -> DistillationLoss:
        mode = _enum_value(DistillationMode, self.config.mode, "mode")
        strategies: dict[DistillationMode, type[DistillationLoss]] = {
            DistillationMode.HARD: HardCrossEntropyLoss,
            DistillationMode.SOFT: SoftKLDistillationLoss,
            DistillationMode.COT: CoTTokenCrossEntropyLoss,
            DistillationMode.COMBINED: CombinedDistillationLoss,
        }
        return strategies[mode](self.vocabulary.pad_id)

    def _default_loss_weight_scheduler(self) -> LossWeightScheduler:
        if self.config.cot_warmup_epochs > 0:
            return LinearCoTCurriculum(self.config.cot_warmup_epochs)
        return StaticLossWeightScheduler()

    def _teacher_forcing_ratio(self, epoch: int) -> float:
        if self.config.epochs == 1:
            return self.config.final_teacher_forcing_ratio
        progress = epoch / (self.config.epochs - 1)
        return (
            self.config.teacher_forcing_ratio
            + progress
            * (
                self.config.final_teacher_forcing_ratio
                - self.config.teacher_forcing_ratio
            )
        )

    def _build_learning_rate_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        steps_per_epoch: int,
    ) -> torch.optim.lr_scheduler.LambdaLR:
        total_steps = max(1, steps_per_epoch * self.config.epochs)
        warmup_steps = min(
            total_steps,
            steps_per_epoch * self.config.warmup_epochs,
        )

        def multiplier(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)

    def _write_experiment_log(self) -> None:
        if not self.config.experiment_log_path or self._last_history is None:
            return
        log_path = Path(self.config.experiment_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_class": type(self.model).__name__,
            "trainer_config": _json_safe(asdict(self.config)),
            "loss_strategy": self.loss_strategy.name,
            "loss_weight_scheduler": type(self.loss_weight_scheduler).__name__,
            "training_history": _json_safe(asdict(self._last_history)),
            "label_metadata": self._label_metadata(self._last_labels),
            "evaluation": _json_safe(self._last_evaluation),
        }
        log_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _label_metadata(
        labels: Sequence[CalibratedLabel],
    ) -> list[dict[str, Any]]:
        return [
            {
                "example_id": item.example_id,
                "weight": item.weight,
                "source_score": item.source_score,
                "metadata": _json_safe(dict(item.metadata)),
            }
            for item in labels
        ]

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _enum_value(
    enum_type: type[Enum],
    value: Enum | str,
    field_name: str,
) -> Any:
    try:
        return enum_type(value)
    except ValueError as error:
        options = ", ".join(str(item.value) for item in enum_type)
        raise ValueError(f"{field_name} must be one of: {options}") from error


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
