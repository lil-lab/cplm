"""Tokenize a Hugging Face text dataset into the litdata format cplm trains on.

    python scripts/prepare_data.py --out data/fineweb-edu-350BT \
        [--dataset HuggingFaceFW/fineweb-edu --subset sample/350BT --workers 32]

Each document becomes its token ids followed by EOS, concatenated into one stream
of uint16 tokens (the SmolLM2 vocabulary has 49,152 entries) and cut into 64MB
chunks. Training reads fixed windows of seq_length + 1 tokens out of that stream.
"""

import argparse

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from litdata import TokensLoader, optimize
from transformers import AutoTokenizer


class Tokenize:
    """Called by litdata worker processes on one parquet file; yields one tensor per document."""

    def __init__(self, dataset, tokenizer, local):
        self.dataset, self.tokenizer_name, self.local = dataset, tokenizer, local
        self.tokenizer = None

    def __call__(self, file):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        eos = self.tokenizer.eos_token_id
        path = file if self.local else hf_hub_download(self.dataset, file, repo_type="dataset")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=["text"]):
            for ids in self.tokenizer(batch.column("text").to_pylist())["input_ids"]:
                yield torch.tensor(ids + [eos], dtype=torch.uint16)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--subset", default="sample/350BT", help="parquet folder inside the dataset repo")
    p.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--files", nargs="*", help="local parquet files instead of the Hub dataset")
    args = p.parse_args()

    assert len(AutoTokenizer.from_pretrained(args.tokenizer)) <= 2 ** 16, "tokens are stored as uint16"

    files = args.files or sorted(
        f for f in list_repo_files(args.dataset, repo_type="dataset")
        if f.startswith(args.subset + "/") and f.endswith(".parquet"))

    optimize(fn=Tokenize(args.dataset, args.tokenizer, local=bool(args.files)), inputs=files,
             output_dir=args.out, chunk_bytes="64MB", item_loader=TokensLoader(),
             num_workers=args.workers)


if __name__ == "__main__":
    main()
