"""PyTorch student model and hard/soft/chain-of-thought training objectives."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from enum import Enum
from collections.abc import Iterable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

from tiny_distillation.core.types import CalibratedLabel, TrainingHistory


class DistillationMode(str, Enum):
    HARD = "hard"
    SOFT = "soft"
    COT = "cot"
    COMBINED = "combined"


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


class TinyStudent(nn.Module):
    """GRU encoder with a classifier and conditional reasoning decoder."""

    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        *,
        embedding_dim: int = 48,
        hidden_dim: int = 64,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_labels)
        self.decoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.reasoning_head = nn.Linear(hidden_dim, vocab_size)

    def encode(self, prompt_ids: Tensor, prompt_lengths: Tensor) -> Tensor:
        embedded = self.embedding(prompt_ids)
        packed = pack_padded_sequence(
            embedded,
            prompt_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.encoder(packed)
        return hidden

    def forward(
        self,
        prompt_ids: Tensor,
        prompt_lengths: Tensor,
        reasoning_input_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        hidden = self.encode(prompt_ids, prompt_lengths)
        class_logits = self.classifier(hidden[-1])
        if reasoning_input_ids is None:
            return class_logits, None
        decoder_input = self.embedding(reasoning_input_ids)
        decoder_output, _ = self.decoder(decoder_input, hidden)
        return class_logits, self.reasoning_head(decoder_output)

    @torch.no_grad()
    def generate_reasoning(
        self,
        prompt_ids: Tensor,
        prompt_lengths: Tensor,
        *,
        bos_id: int,
        eos_id: int,
        max_new_tokens: int = 32,
    ) -> Tensor:
        hidden = self.encode(prompt_ids, prompt_lengths)
        current = torch.full(
            (prompt_ids.shape[0], 1),
            bos_id,
            dtype=torch.long,
            device=prompt_ids.device,
        )
        generated: list[Tensor] = []
        finished = torch.zeros(prompt_ids.shape[0], dtype=torch.bool, device=prompt_ids.device)
        for _ in range(max_new_tokens):
            output, hidden = self.decoder(self.embedding(current), hidden)
            next_token = self.reasoning_head(output[:, -1]).argmax(dim=-1)
            generated.append(next_token)
            finished |= next_token.eq(eos_id)
            if bool(finished.all()):
                break
            current = next_token.unsqueeze(1)
        return torch.stack(generated, dim=1)


class _DistillationDataset(Dataset[CalibratedLabel]):
    def __init__(self, labels: Sequence[CalibratedLabel]) -> None:
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> CalibratedLabel:
        return self.labels[index]


@dataclass(frozen=True)
class TrainerConfig:
    mode: DistillationMode = DistillationMode.COMBINED
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 3e-3
    soft_temperature: float = 1.0
    hard_weight: float = 1.0
    soft_weight: float = 1.0
    cot_weight: float = 0.5
    gradient_clip: float = 1.0
    seed: int = 7
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.soft_temperature <= 0:
            raise ValueError("learning_rate and soft_temperature must be positive")
        if any(
            weight < 0
            for weight in (self.hard_weight, self.soft_weight, self.cot_weight)
        ):
            raise ValueError("loss weights must be non-negative")


class StudentTrainer:
    def __init__(
        self,
        model: TinyStudent,
        vocabulary: Vocabulary,
        config: TrainerConfig | None = None,
    ) -> None:
        self.model = model
        self.vocabulary = vocabulary
        self.config = config or TrainerConfig()
        self.device = torch.device(self.config.device)
        self.model.to(self.device)

    def train(self, labels: Sequence[CalibratedLabel]) -> TrainingHistory:
        if not labels:
            raise ValueError("cannot train without calibrated labels")
        self._seed_everything(self.config.seed)
        generator = torch.Generator().manual_seed(self.config.seed)
        loader = DataLoader(
            _DistillationDataset(labels),
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=self._collate,
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )
        totals: list[float] = []
        hard_history: list[float] = []
        soft_history: list[float] = []
        cot_history: list[float] = []
        for _ in range(self.config.epochs):
            epoch_losses = [0.0, 0.0, 0.0, 0.0]
            batches = 0
            self.model.train()
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                losses = self._losses(batch)
                losses[0].backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                optimizer.step()
                for index, loss in enumerate(losses):
                    epoch_losses[index] += float(loss.detach())
                batches += 1
            totals.append(epoch_losses[0] / batches)
            hard_history.append(epoch_losses[1] / batches)
            soft_history.append(epoch_losses[2] / batches)
            cot_history.append(epoch_losses[3] / batches)
        return TrainingHistory(
            losses=tuple(totals),
            hard_losses=tuple(hard_history),
            soft_losses=tuple(soft_history),
            cot_losses=tuple(cot_history),
        )

    @torch.no_grad()
    def predict(self, prompts: Sequence[str]) -> tuple[Tensor, list[str]]:
        if not prompts:
            return torch.empty((0, self.model.classifier.out_features)), []
        prompt_ids, prompt_lengths = self._encode_prompts(prompts)
        self.model.eval()
        class_logits, _ = self.model(prompt_ids, prompt_lengths)
        reasoning_ids = self.model.generate_reasoning(
            prompt_ids,
            prompt_lengths,
            bos_id=self.vocabulary.bos_id,
            eos_id=self.vocabulary.eos_id,
        )
        reasoning = [self.vocabulary.decode(row.tolist()) for row in reasoning_ids]
        return class_logits.cpu(), reasoning

    def _losses(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        class_logits, reasoning_logits = self.model(
            batch["prompt_ids"],
            batch["prompt_lengths"],
            batch["reasoning_input_ids"],
        )
        weights = batch["weights"]
        hard_per_item = F.cross_entropy(
            class_logits,
            batch["hard_labels"],
            reduction="none",
        )
        hard_loss = self._weighted_mean(hard_per_item, weights)

        temperature = self.config.soft_temperature
        soft_per_item = F.kl_div(
            F.log_softmax(class_logits / temperature, dim=-1),
            batch["soft_labels"],
            reduction="none",
        ).sum(dim=-1) * temperature**2
        soft_loss = self._weighted_mean(soft_per_item, weights)

        if reasoning_logits is None:
            raise RuntimeError("reasoning logits are required for CoT loss")
        token_losses = F.cross_entropy(
            reasoning_logits.transpose(1, 2),
            batch["reasoning_target_ids"],
            reduction="none",
            ignore_index=self.vocabulary.pad_id,
        )
        token_mask = batch["reasoning_target_ids"].ne(self.vocabulary.pad_id)
        cot_per_item = (token_losses * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp_min(1)
        cot_loss = self._weighted_mean(cot_per_item, weights)

        mode = self.config.mode
        if mode == DistillationMode.HARD:
            total = self.config.hard_weight * hard_loss
        elif mode == DistillationMode.SOFT:
            total = self.config.soft_weight * soft_loss
        elif mode == DistillationMode.COT:
            total = self.config.cot_weight * cot_loss
        else:
            total = (
                self.config.hard_weight * hard_loss
                + self.config.soft_weight * soft_loss
                + self.config.cot_weight * cot_loss
            )
        return total, hard_loss, soft_loss, cot_loss

    def _collate(self, labels: Sequence[CalibratedLabel]) -> dict[str, Tensor]:
        prompt_sequences = [
            self.vocabulary.encode(item.prompt, add_eos=True) for item in labels
        ]
        reasoning_sequences = [
            self.vocabulary.encode(item.reasoning, add_bos=True, add_eos=True)
            for item in labels
        ]
        prompt_ids = self._pad(prompt_sequences)
        reasoning_ids = self._pad(reasoning_sequences)
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
            "reasoning_target_ids": reasoning_ids[:, 1:].to(self.device),
        }

    def _encode_prompts(self, prompts: Sequence[str]) -> tuple[Tensor, Tensor]:
        sequences = [self.vocabulary.encode(prompt, add_eos=True) for prompt in prompts]
        return (
            self._pad(sequences).to(self.device),
            torch.tensor([len(sequence) for sequence in sequences], device=self.device),
        )

    def _pad(self, sequences: Sequence[Sequence[int]]) -> Tensor:
        max_length = max(len(sequence) for sequence in sequences)
        result = torch.full(
            (len(sequences), max_length),
            self.vocabulary.pad_id,
            dtype=torch.long,
        )
        for row, sequence in enumerate(sequences):
            result[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        return result

    @staticmethod
    def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
        return (values * weights).sum() / weights.sum().clamp_min(1e-8)

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
