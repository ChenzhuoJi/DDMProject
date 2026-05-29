"""DNA sequence dataset for character-level discrete diffusion.

Vocabulary (7 tokens):
    [PAD]=0  [MASK]=1  [UNK]=2  A=3  C=4  G=5  T=6

[PAD]  — padding to align variable-length sequences in a batch
[MASK] — absorbing state used by the forward diffusion process
[UNK]  — any character not in {A, C, G, T} (e.g. N, ambiguous bases)
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple


# ── Vocabulary ────────────────────────────────────────────────────────────────

class DNAVocab:
    """Bidirectional mapping between DNA characters and integer IDs.

    Special tokens occupy the lowest IDs so that the four bases always
    live at a contiguous, predictable range [3, 6].
    """

    PAD_TOKEN  = '[PAD]'
    MASK_TOKEN = '[MASK]'
    UNK_TOKEN  = '[UNK]'
    BASE_TOKENS = ['A', 'C', 'G', 'T']

    def __init__(self):
        tokens = [self.PAD_TOKEN,
                  self.MASK_TOKEN,
                  self.UNK_TOKEN] + self.BASE_TOKENS

        self.token2id: Dict[str, int] = {tok: i for i, tok in enumerate(tokens)}
        self.id2token: Dict[int, str] = {i: tok for tok, i in self.token2id.items()}

        self.pad_id  = self.token2id[self.PAD_TOKEN]   # 0
        self.mask_id = self.token2id[self.MASK_TOKEN]  # 1
        self.unk_id  = self.token2id[self.UNK_TOKEN]   # 2

    def __len__(self) -> int:
        return len(self.token2id)

    def encode(self, sequence: str) -> List[int]:
        """Convert a DNA string to a list of integer IDs.

        Any character not in {A, C, G, T} is mapped to [UNK].
        """
        return [self.token2id.get(ch.upper(), self.unk_id)
                for ch in sequence]

    def decode(self, ids: List[int]) -> str:
        """Convert a list of integer IDs back to a DNA string.

        Special tokens ([PAD], [MASK], [UNK]) are included as-is.
        """
        return ''.join(self.id2token.get(i, self.UNK_TOKEN) for i in ids)

    def summary(self) -> str:
        rows = [f"  {tok!r:10s} -> {idx}" for tok, idx in self.token2id.items()]
        return "DNAVocab (" + str(len(self)) + " tokens):\n" + "\n".join(rows)


# ── Dataset ───────────────────────────────────────────────────────────────────

class DNADataset(Dataset):
    """Character-level DNA sequence dataset.

    Each item is a dict with:
        input_ids      — LongTensor of shape (max_seq_len,)
        attention_mask — BoolTensor of shape (max_seq_len,),
                         True for real bases, False for [PAD]

    Sequences longer than max_seq_len are truncated from the right.
    Sequences shorter than max_seq_len are right-padded with [PAD].

    Args:
        sequences:    list of raw DNA strings (uppercase or lowercase).
        vocab:        DNAVocab instance; a fresh one is created if None.
        max_seq_len:  fixed output length for every sample.
    """

    def __init__(
        self,
        sequences: List[str],
        vocab: DNAVocab | None = None,
        max_seq_len: int = 256,
    ):
        self.vocab       = vocab if vocab is not None else DNAVocab()
        self.max_seq_len = max_seq_len
        self.sequences   = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        ids = self.vocab.encode(self.sequences[idx])

        # ── truncate ──────────────────────────────────────────────
        ids = ids[: self.max_seq_len]
        real_len = len(ids)

        # ── right-pad ─────────────────────────────────────────────
        pad_len = self.max_seq_len - real_len
        ids = ids + [self.vocab.pad_id] * pad_len

        input_ids      = torch.tensor(ids, dtype=torch.long)
        attention_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        attention_mask[:real_len] = True

        return {'input_ids': input_ids, 'attention_mask': attention_mask}


# ── Collate function ──────────────────────────────────────────────────────────

def dna_collate_fn(
    batch: List[Dict[str, Tensor]]
) -> Dict[str, Tensor]:
    """Stack a list of DNADataset items into a batch.

    Output keys:
        input_ids      — LongTensor  (B, max_seq_len)
        attention_mask — BoolTensor  (B, max_seq_len)

    Padding is already applied inside __getitem__, so this function
    is a plain stack — no dynamic padding needed.
    """
    return {
        'input_ids':      torch.stack([s['input_ids']      for s in batch]),
        'attention_mask': torch.stack([s['attention_mask'] for s in batch]),
    }


def build_dna_dataloader(
    sequences: List[str],
    batch_size: int = 32,
    max_seq_len: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
    vocab: DNAVocab | None = None,
) -> Tuple[DataLoader, DNAVocab]:
    """Convenience factory: returns (DataLoader, vocab)."""
    vocab   = vocab if vocab is not None else DNAVocab()
    dataset = DNADataset(sequences, vocab=vocab, max_seq_len=max_seq_len)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dna_collate_fn,
    )
    return loader, vocab


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("DNADataset Smoke Test")
    print("=" * 60)

    # ── vocabulary table ──────────────────────────────────────────
    vocab = DNAVocab()
    print(f"\n{vocab.summary()}")

    # ── encode / decode round-trip ────────────────────────────────
    seq_rt = "ACGTN"          # N should become [UNK]
    ids_rt = vocab.encode(seq_rt)
    decoded = vocab.decode(ids_rt)
    print(f"\n[Round-trip]  '{seq_rt}'  ->  {ids_rt}  ->  '{decoded}'")

    # ── mock DNA sequences ─────────────────────────────────────────
    mock_seqs = [
        "ATCGATCG",                          # short  (8 bp)
        "GCTAGCTA" * 40,                     # long   (320 bp  > 256, will be truncated)
    ]
    MAX_LEN = 256

    print(f"\n[Input sequences]")
    for i, s in enumerate(mock_seqs):
        print(f"  seq[{i}]: len={len(s):>4}  preview='{s[:24]}...'")

    # ── dataset item inspection ────────────────────────────────────
    dataset = DNADataset(mock_seqs, vocab=vocab, max_seq_len=MAX_LEN)
    print(f"\n[Dataset] len={len(dataset)}, max_seq_len={MAX_LEN}")

    item0 = dataset[0]
    item1 = dataset[1]
    print(f"\n  item[0] input_ids shape      : {item0['input_ids'].shape}")
    print(f"  item[0] attention_mask shape : {item0['attention_mask'].shape}")
    real_tokens_0 = item0['attention_mask'].sum().item()
    pad_tokens_0  = MAX_LEN - real_tokens_0
    print(f"  item[0] real tokens={real_tokens_0},  pad tokens={pad_tokens_0}")

    real_tokens_1 = item1['attention_mask'].sum().item()
    print(f"  item[1] real tokens={real_tokens_1}  (truncated from {len(mock_seqs[1])})")

    # ── dataloader batch ──────────────────────────────────────────
    loader, _ = build_dna_dataloader(
        mock_seqs,
        batch_size=2,
        max_seq_len=MAX_LEN,
        shuffle=False,
    )
    batch = next(iter(loader))

    print(f"\n[Batch]")
    print(f"  input_ids      shape : {batch['input_ids'].shape}   dtype={batch['input_ids'].dtype}")
    print(f"  attention_mask shape : {batch['attention_mask'].shape}  dtype={batch['attention_mask'].dtype}")

    print(f"\n  input_ids[0] (first 20 tokens) : {batch['input_ids'][0, :20].tolist()}")
    print(f"  input_ids[1] (first 20 tokens) : {batch['input_ids'][1, :20].tolist()}")

    print(f"\n  attention_mask[0] (first 20)   : {batch['attention_mask'][0, :20].tolist()}")
    print(f"  attention_mask[0] tail (last 5) : {batch['attention_mask'][0, -5:].tolist()}")

    # verify trailing pads on seq[0]
    assert batch['input_ids'][0, real_tokens_0:].eq(vocab.pad_id).all(), \
        "FAIL: padding region contains non-PAD tokens"
    assert batch['input_ids'].shape == (2, MAX_LEN), \
        f"FAIL: unexpected batch shape {batch['input_ids'].shape}"
    assert batch['attention_mask'].shape == (2, MAX_LEN), \
        f"FAIL: unexpected mask shape {batch['attention_mask'].shape}"

    print(f"\n  Verified: all tail positions of seq[0] == PAD_ID ({vocab.pad_id})")
    print(f"  Verified: batch shape perfectly aligned to [2, {MAX_LEN}]")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)
