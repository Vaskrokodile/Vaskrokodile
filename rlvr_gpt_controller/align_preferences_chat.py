import argparse
import json

from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    count = 0
    with open(args.input, encoding="utf-8") as src, open(args.output, "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": record["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            ) + "</think>\n"
            record["prompt"] = rendered
            record["prompt_format"] = "chat_template_plus_forced_no_think"
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    print(json.dumps({"pairs": count, "output": args.output}))


if __name__ == "__main__":
    main()
