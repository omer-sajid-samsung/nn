import time
import requests

API_URL = "http://localhost:8000/v1/completions"
MODEL = "Qwen/Qwen3-8B-AWQ"

shared_context = "The quick brown fox jumps over the lazy dog. " * 400
question = "Summarize the sentence above in 5 words."

def ask(label):
    payload = {
        "model": MODEL,
        "prompt": shared_context + "\n\n" + question,
        "max_tokens": 30,
        "temperature": 0,
    }
    start = time.perf_counter()
    resp = requests.post(API_URL, json=payload)
    elapsed = time.perf_counter() - start
    resp.raise_for_status()
    text = resp.json()["choices"][0]["text"].strip()
    print(f"[{label}] {elapsed:.2f}s -> {text[:100]!r}")
    return elapsed

if __name__ == "__main__":
    cold = ask("COLD (first request)")
    time.sleep(1)
    warm = ask("WARM (same prompt again)")
    print(f"\n{cold / warm:.2f}x faster on the warm request")
