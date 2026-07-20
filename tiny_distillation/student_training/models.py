"""Student-model contracts and reference architectures."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence


class StudentModel(nn.Module, ABC):
    """Model interface required by the distillation trainer."""

    num_labels: int

    @abstractmethod
    def forward(
        self,
        prompt_ids: Tensor,
        prompt_lengths: Tensor,
        reasoning_input_ids: Tensor | None = None,
        hard_labels: Tensor | None = None,
        teacher_forcing_ratio: float = 1.0,
    ) -> tuple[Tensor, Tensor | None]:
        """Return classification logits and optional reasoning-token logits."""

    @abstractmethod
    def generate_reasoning(
        self,
        prompt_ids: Tensor,
        prompt_lengths: Tensor,
        *,
        bos_id: int,
        eos_id: int,
        max_new_tokens: int = 32,
    ) -> Tensor:
        """Autoregressively generate rationale token IDs."""


class TinyStudent(StudentModel):
    """GRU encoder with answer-conditioned classification and rationale heads."""

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
        self.num_labels = num_labels
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_labels)
        self.label_embedding = nn.Embedding(num_labels, hidden_dim)
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
        hard_labels: Tensor | None = None,
        teacher_forcing_ratio: float = 1.0,
    ) -> tuple[Tensor, Tensor | None]:
        hidden = self.encode(prompt_ids, prompt_lengths)
        class_logits = self.classifier(hidden[-1])
        if reasoning_input_ids is None:
            return class_logits, None
        labels = hard_labels if hard_labels is not None else class_logits.argmax(dim=-1)
        conditioned_hidden = hidden + self.label_embedding(labels).unsqueeze(0)
        reasoning_logits = self._decode_training(
            reasoning_input_ids,
            conditioned_hidden,
            teacher_forcing_ratio,
        )
        return class_logits, reasoning_logits

    def _decode_training(
        self,
        reasoning_input_ids: Tensor,
        hidden: Tensor,
        teacher_forcing_ratio: float,
    ) -> Tensor:
        if teacher_forcing_ratio >= 1.0:
            decoder_output, _ = self.decoder(
                self.embedding(reasoning_input_ids),
                hidden,
            )
            return self.reasoning_head(decoder_output)

        current = reasoning_input_ids[:, :1]
        outputs: list[Tensor] = []
        for index in range(reasoning_input_ids.shape[1]):
            output, hidden = self.decoder(self.embedding(current), hidden)
            logits = self.reasoning_head(output[:, -1])
            outputs.append(logits)
            if index + 1 >= reasoning_input_ids.shape[1]:
                continue
            predicted = logits.argmax(dim=-1)
            teacher_mask = torch.rand(
                predicted.shape,
                device=predicted.device,
            ).lt(teacher_forcing_ratio)
            next_teacher = reasoning_input_ids[:, index + 1]
            current = torch.where(teacher_mask, next_teacher, predicted).unsqueeze(1)
        return torch.stack(outputs, dim=1)

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
        labels = self.classifier(hidden[-1]).argmax(dim=-1)
        hidden = hidden + self.label_embedding(labels).unsqueeze(0)
        current = torch.full(
            (prompt_ids.shape[0], 1),
            bos_id,
            dtype=torch.long,
            device=prompt_ids.device,
        )
        generated: list[Tensor] = []
        finished = torch.zeros(
            prompt_ids.shape[0],
            dtype=torch.bool,
            device=prompt_ids.device,
        )
        for _ in range(max_new_tokens):
            output, hidden = self.decoder(self.embedding(current), hidden)
            next_token = self.reasoning_head(output[:, -1]).argmax(dim=-1)
            generated.append(next_token)
            finished |= next_token.eq(eos_id)
            if bool(finished.all()):
                break
            current = next_token.unsqueeze(1)
        if not generated:
            return torch.empty(
                (prompt_ids.shape[0], 0),
                dtype=torch.long,
                device=prompt_ids.device,
            )
        return torch.stack(generated, dim=1)


class TransformerStudent(TinyStudent):
    """Transformer prompt encoder with the same compact rationale decoder."""

    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        *,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_dim: int = 128,
        dropout: float = 0.1,
        max_sequence_length: int = 256,
        pad_id: int = 0,
    ) -> None:
        if embedding_dim != hidden_dim:
            raise ValueError("TransformerStudent requires embedding_dim == hidden_dim")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive")
        super().__init__(
            vocab_size,
            num_labels,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            pad_id=pad_id,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.position_embedding = nn.Embedding(max_sequence_length, hidden_dim)
        self.max_sequence_length = max_sequence_length

    def encode(self, prompt_ids: Tensor, prompt_lengths: Tensor) -> Tensor:
        sequence_length = prompt_ids.shape[1]
        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"prompt length {sequence_length} exceeds "
                f"max_sequence_length={self.max_sequence_length}"
            )
        positions = torch.arange(sequence_length, device=prompt_ids.device)
        embedded = self.embedding(prompt_ids) + self.position_embedding(positions)
        padding_mask = prompt_ids.eq(self.pad_id)
        encoded = self.encoder(embedded, src_key_padding_mask=padding_mask)
        final_indexes = prompt_lengths.sub(1).clamp_min(0)
        pooled = encoded[
            torch.arange(prompt_ids.shape[0], device=prompt_ids.device),
            final_indexes,
        ]
        return pooled.unsqueeze(0)
