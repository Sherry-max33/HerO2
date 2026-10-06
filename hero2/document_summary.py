import argparse
import copy
import json
import re
import time
import tqdm
import os
from ollama_backend import LLM, SamplingParams
import torch

prompt = """Your task is to read the following document carefully and summarize it into a single, coherent paragraph. Focus on capturing the main ideas and essential details without adding new information or personal opinions.
Document:
{}
"""

def all_documents(knowledge_file, top_k=-1):
  documents, urls = [], []
  
  with open(knowledge_file, "r", encoding="utf-8") as json_file:
    for i, line in enumerate(json_file):
        if len(urls) == top_k: break
      
        data = json.loads(line)
        if len(data["url2text"]) > 0:
            documents.append(data["url2text"])
            urls.append(data['url'])
  return documents, urls

def truncate_chat_prompt(prompt: str, tokenizer, max_len: int) -> str:
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len]
        return tokenizer.decode(token_ids, add_special_tokens=False)
    else:
        return prompt

def main(args):
    gpu_counts = torch.cuda.device_count()
    print(f"Using {gpu_counts} GPU{'s' if gpu_counts > 1 else ''}")
    
    llm = LLM(
        model= args.model,
        tensor_parallel_size=gpu_counts, 
        max_model_len=32768,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        skip_special_tokens=False,
        max_tokens=4096,
    )
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    for i in tqdm.tqdm(range(args.num_files)):
        documents, urls = all_documents(f"{args.knowledge_store}/{i}.json")
        out_path = f"{args.save_path}/{i}.json"
        done = 0
        if os.path.exists(out_path):
            with open(out_path, encoding="utf-8") as existing:
                done = sum(1 for line in existing if line.strip())
        with open(out_path, "a", encoding="utf-8") as f:
            for doc_i, (pd, url) in enumerate(zip(documents, urls)):
                if doc_i < done:
                    continue
                max_len = args.max_input_token_size
                while True:
                    target_prompt = copy.deepcopy(prompt)
                    truncate_doc = truncate_chat_prompt("\n".join(pd), tokenizer, max_len=max_len)
                    target_prompt = target_prompt.format(truncate_doc)
                    target_prompt = llm.get_tokenizer().apply_chat_template([{"role": "user", "content": target_prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    try:
                        response = llm.generate([target_prompt], sampling_params, use_tqdm=False)[0]
                        break
                    except RuntimeError as exc:
                        message = str(exc)
                        match = re.search(r"Prompt tokens limit exceeded: (\d+) > (\d+)", message)
                        if match and max_len > 2000:
                            used = int(match.group(1))
                            allowed = int(match.group(2))
                            max_len = max(2000, int(max_len * allowed / used * 0.8))
                            args.max_input_token_size = max_len
                            print(f"prompt over credit limit, retrying with max_len={max_len}", flush=True)
                            continue
                        afford = re.search(r"can only afford (\d+)", message)
                        if afford and sampling_params.max_tokens > 256:
                            sampling_params.max_tokens = max(256, int(afford.group(1)) - 200)
                            print(
                                f"lowering max_tokens to {sampling_params.max_tokens}",
                                flush=True,
                            )
                            continue
                        if "402" in message or "credits" in message.lower():
                            print("OpenRouter credits too low, waiting 10 minutes", flush=True)
                            time.sleep(600)
                        raise
                json_line = {
                    "url2text": response.outputs[0].text.strip(),
                    "url": url
                }
                f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{i}] summarized {doc_i + 1}/{len(documents)}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--knowledge_store', default='knowledge_store/dev')
    parser.add_argument('--save_path', default='knowledge_store/dev_summary')
    parser.add_argument('--max_input_token_size', type=int ,default=24000)
    parser.add_argument('-m', '--model', default="Qwen/Qwen3-8B")
    parser.add_argument('--num_files', type=int, default=500)
    
    args = parser.parse_args()
    main(args)
  
  