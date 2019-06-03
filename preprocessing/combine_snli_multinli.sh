
mkdir data/raw/snli_multinli
cat data/raw/snli_1.0/snli_1.0_train.jsonl data/raw/multinli_1.0/multinli_1.0_train.jsonl  > data/raw/snli_multinli/snli_multinli_train.jsonl
cat data/raw/snli_1.0/snli_1.0_dev.jsonl data/raw/multinli_1.0/multinli_1.0_dev_mismatched.jsonl data/raw/multinli_1.0/multinli_1.0_dev_matched.jsonl > data/raw/snli_multinli/snli_multinli_dev.jsonl
