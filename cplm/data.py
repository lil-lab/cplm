"""Pre-tokenized litdata streams and per-batch document bookkeeping.

``data_path`` is a litdata ``StreamingDataset`` of packed token ids (documents
separated by EOS), either a single directory with an ``index.json`` or a parent
of several such directories (one per preprocessing node), which are combined.
Each sample is ``seq_length + 1`` tokens so inputs and labels are one shift apart.
"""

import os

import torch
from litdata import (CombinedStreamingDataset, StreamingDataLoader, StreamingDataset,
                     TokensLoader, train_test_split)

SPLITS = (0.99, 0.005, 0.005)   # train / val / test, fixed seed so resumes see the same split


def _splits(path, seq_length):
    def ds(d):
        return StreamingDataset(input_dir=d, item_loader=TokensLoader(block_size=seq_length + 1),
                                shuffle=True, drop_last=True)
    if os.path.exists(os.path.join(path, "index.json")):
        dirs = [path]
    else:
        dirs = sorted(os.path.join(path, d) for d in os.listdir(path)
                      if os.path.isdir(os.path.join(path, d)))
    parts = [train_test_split(ds(d), splits=SPLITS, seed=42) for d in dirs]

    def combine(i):
        shards = [p[i] for p in parts]
        return shards[0] if len(shards) == 1 else CombinedStreamingDataset(shards, iterate_over_all=True)
    return combine(0), combine(1)


def build_loaders(path, seq_length, batch_size, num_workers):
    train, val = _splits(path, seq_length)
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    if num_workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=4)
    return StreamingDataLoader(train, shuffle=True, **kw), StreamingDataLoader(val, **kw)


def prepare_batch(batch, eos_id):
    """(input_ids, labels, position_ids, doc_ids), each [B, T].

    An EOS is the last token of its document: positions restart at 0 right after
    it and the document id increments there.
    """
    batch = batch.long()
    input_ids, labels = batch[:, :-1], batch[:, 1:]
    T = input_ids.shape[1]
    starts = torch.zeros_like(input_ids, dtype=torch.bool)
    starts[:, 1:] = input_ids[:, :-1] == eos_id
    doc_ids = starts.long().cumsum(dim=1)
    idx = torch.arange(T, device=batch.device).expand_as(input_ids)
    last_start = torch.where(starts, idx, torch.zeros_like(idx)).cummax(dim=1).values
    return input_ids, labels, idx - last_start, doc_ids
