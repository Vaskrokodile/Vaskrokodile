import argparse
import gc
import shutil

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    shutil.rmtree(args.output, ignore_errors=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.save_pretrained(args.output)
    AutoTokenizer.from_pretrained(args.base, trust_remote_code=True).save_pretrained(args.output)
    del model
    gc.collect()
    print(args.output)


if __name__ == "__main__":
    main()
