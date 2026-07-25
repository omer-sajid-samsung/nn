"""
runner.py — Unified workload runner.

Dispatched by RunPlan.framework.  Each _run_* method launches the
appropriate framework and returns a RunResult with timing + inline metrics.
"""

from __future__ import annotations
import os, sys, json, re, time, textwrap, subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .plan import RunPlan, Framework, AiOp


@dataclass
class RunResult:
    success: bool
    duration_s: float
    tokens_generated: int  = 0
    throughput_tok_s: float= 0.0
    ttft_ms: float         = 0.0
    notes: str             = ""
    stderr: str            = ""


class WorkloadRunner:

    def __init__(self, plan: RunPlan, work_dir: Path):
        self.plan     = plan
        self.work_dir = work_dir
        work_dir.mkdir(parents=True, exist_ok=True)

    # ── Public entry point ──────────────────────────────────────────────────

    def run(self, context_len: int, batch_size: int) -> RunResult:
        p = self.plan
        dispatch = {
            Framework.FIO:        self._run_fio,
            Framework.LLAMA_CPP:  self._run_llama_cpp,
            Framework.VLLM:       self._run_vllm,
            Framework.SGLANG:     self._run_sglang,
            Framework.ACCELERATE: self._run_accelerate,
            Framework.FLEXGEN:    self._run_flexgen,
            Framework.DEEPSPEED:  self._run_deepspeed,
        }
        fn = dispatch.get(p.framework)
        if fn is None:
            return RunResult(False, 0.0, notes=f"Unknown framework: {p.framework}")
        return fn(context_len, batch_size)

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _prompt(self, ctx: int) -> str:
        return "Analyze the following: " + ("token " * max(1, ctx // 2))

    def _script(self, name: str, code: str) -> Path:
        p = self.work_dir / name
        p.write_text(textwrap.dedent(code))
        return p

    def _exec(self, cmd: list | None = None,
              script: Path | None = None,
              timeout: int = 1800) -> tuple[int, str, str]:
        if script:
            cmd = [sys.executable, str(script)]
        t0 = time.time()
        r  = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr

    def _parse_json_from(self, text: str) -> dict:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try: return json.loads(m.group())
            except: pass
        return {}

    def _drop_caches(self):
        try:
            subprocess.run(["sudo","sh","-c","echo 3 > /proc/sys/vm/drop_caches"],
                           check=False, capture_output=True)
        except Exception:
            pass

    # ── fio (synthetic baseline) ─────────────────────────────────────────────

    def _run_fio(self, ctx: int, bs: int) -> RunResult:
        p   = self.plan
        op  = p.ai_op
        # Map AI op to fio rw mode and block size
        rw_map = {
            AiOp.WEIGHT_LOAD: ("read",    "1M"),
            AiOp.PREFILL:     ("write",   "4M"),
            AiOp.DECODE:      ("randread","4k"),
            AiOp.KV_EVICT:    ("randwrite","4k"),
            AiOp.CHECKPOINT:  ("write",   "4M"),
            AiOp.MIXED:       ("randrw",  "4k"),
        }
        rw, blksz = rw_map.get(op, ("read", "1M"))
        self._drop_caches()
        out = self.work_dir / "fio_out.json"
        cmd = [
            "fio", f"--name=amoprof_{rw}",
            f"--rw={rw}", f"--bs={blksz}",
            "--iodepth=64", "--numjobs=8",
            "--size=100G", "--ioengine=io_uring",
            f"--filename={p.ssd_path}/amoprof_fio.bin",
            "--direct=1", "--runtime=60", "--time_based",
            "--output-format=json", f"--output={out}",
        ]
        t0 = time.time()
        rc, stdout, stderr = self._exec(cmd, timeout=120)
        dur = round(time.time()-t0, 2)
        notes = ""
        if out.exists():
            try:
                d   = json.loads(out.read_text())
                job = d.get("jobs",[{}])[0]
                key = "write" if "write" in rw else "read"
                bw  = job.get(key,{}).get("bw",0)/1024
                iops= job.get(key,{}).get("iops",0)
                notes = f"fio_bw={bw:.1f}MB/s iops={iops:.0f}"
            except Exception:
                pass
        return RunResult(rc==0, dur, notes=notes, stderr=stderr[:300])

    # ── llama.cpp (cold weight load / weight-load op) ────────────────────────

    def _run_llama_cpp(self, ctx: int, bs: int) -> RunResult:
        p    = self.plan
        gguf = p.gguf_path or f"{p.ssd_path}/{p.model.alias}.gguf"
        if not Path(gguf).exists():
            return RunResult(False, 0.0, notes=f"GGUF not found: {gguf}")
        self._drop_caches()
        bin_ = os.environ.get("LLAMA_CPP_BIN", "llama-cli")
        cmd  = [bin_, "-m", gguf, "--mmap", "--no-mmap-cache",
                "-c", str(ctx), "-n", "16",
                "-p", self._prompt(ctx)[:4096], "--log-disable"]
        t0 = time.time()
        rc, stdout, stderr = self._exec(cmd, timeout=1200)
        dur = round(time.time()-t0, 2)
        tok_s = 0.0
        for line in (stdout+stderr).splitlines():
            m = re.search(r"([\d.]+)\s*tok/s", line)
            if m: tok_s = float(m.group(1))
        return RunResult(rc==0, dur, throughput_tok_s=tok_s,
                         notes=f"gguf={Path(gguf).name}", stderr=stderr[-200:])

    # ── vLLM ────────────────────────────────────────────────────────────────

    def _run_vllm(self, ctx: int, bs: int) -> RunResult:
        p = self.plan
        cpu_gb = int(p.dram_cap_gb) if p.dram_cap_gb else 50
        disk_gb= int(p.nvme_cap_gb) if p.nvme_cap_gb else 0
        script = self._script("vllm_run.py", f"""
            import time, json, torch
            from vllm import LLM, SamplingParams
            llm = LLM(
                model="{p.model.hf_id}",
                dtype="{p.dtype}",
                gpu_memory_utilization={p.hbm_cap_gb / 80.0:.2f},
                tensor_parallel_size={p.tensor_parallel},
                max_model_len={ctx},
                swap_space={cpu_gb},
                enforce_eager=True,
            )
            prompt = "Analyze this: " + ("token " * {ctx//2})
            params = SamplingParams(max_tokens=64)
            prompts = [prompt] * {max(bs, 1)}
            t0 = time.time()
            out = llm.generate(prompts, params)
            dur = time.time() - t0
            toks = sum(len(o.outputs[0].token_ids) for o in out)
            print(json.dumps({{"duration_s": round(dur,3),
                               "tokens": toks,
                               "tok_s": round(toks/max(dur,0.001),2)}}))
        """)
        t0 = time.time()
        rc, stdout, stderr = self._exec(script=script, timeout=1800)
        dur = round(time.time()-t0, 2)
        d   = self._parse_json_from(stdout)
        return RunResult(rc==0, d.get("duration_s",dur),
                         tokens_generated=d.get("tokens",0),
                         throughput_tok_s=d.get("tok_s",0.0),
                         stderr=stderr[-300:])

    # ── SGLang ──────────────────────────────────────────────────────────────

    def _run_sglang(self, ctx: int, bs: int) -> RunResult:
        p = self.plan
        script = self._script("sglang_run.py", f"""
            import subprocess, sys, time, json, re
            srv = subprocess.Popen([
                sys.executable, "-m", "sglang.launch_server",
                "--model-path", "{p.model.hf_id}",
                "--dtype", "{p.dtype}",
                "--mem-fraction-static", "{min(p.hbm_cap_gb/80.0, 0.95):.2f}",
                "--tp-size", "{p.tensor_parallel}",
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(60)
            bench = subprocess.run([
                sys.executable, "-m", "sglang.bench_serving",
                "--backend","sglang","--model","{p.model.hf_id}",
                "--num-prompts","50","--input-len",str({ctx}),
                "--output-len","64","--request-rate","4",
            ], capture_output=True, text=True, timeout=600)
            srv.terminate()
            tok_s = ttft = 0.0
            for line in bench.stdout.splitlines():
                m = re.search(r"Output token throughput.*?([\d.]+)", line)
                if m: tok_s = float(m.group(1))
                m = re.search(r"Mean TTFT.*?([\d.]+)", line)
                if m: ttft = float(m.group(1))
            print(json.dumps({{"tok_s":tok_s,"ttft_ms":ttft}}))
        """)
        t0 = time.time()
        rc, stdout, stderr = self._exec(script=script, timeout=900)
        dur = round(time.time()-t0, 2)
        d   = self._parse_json_from(stdout)
        return RunResult(rc==0, dur,
                         throughput_tok_s=d.get("tok_s",0.0),
                         ttft_ms=d.get("ttft_ms",0.0),
                         stderr=stderr[-300:])

    # ── HF Accelerate (DRAM/SSD CPU offload) ────────────────────────────────

    def _run_accelerate(self, ctx: int, bs: int) -> RunResult:
        p      = self.plan
        hbm_gb = max(0, int(p.hbm_cap_gb * 0.85))
        cpu_gb = int(p.dram_cap_gb)
        disk   = f'"disk":"{p.nvme_cap_gb:.0f}GB"' if p.nvme_cap_gb else ''
        dtype_map = {"fp8":"torch.float8_e4m3fn","bf16":"torch.bfloat16",
                     "fp16":"torch.float16","int4":"torch.float16"}
        torch_dt = dtype_map.get(p.dtype, "torch.float16")
        script = self._script("accel_run.py", f"""
            import time, json, torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            mem = {{0:"{hbm_gb}GB","cpu":"{cpu_gb}GB"{(','+disk) if disk else ''}}}
            t0 = time.time()
            model = AutoModelForCausalLM.from_pretrained(
                "{p.model.hf_id}", device_map="auto",
                max_memory=mem, offload_folder="{p.offload_dir}",
                torch_dtype={torch_dt})
            tok = AutoTokenizer.from_pretrained("{p.model.hf_id}")
            if tok.pad_token is None: tok.pad_token = tok.eos_token
            load_s = round(time.time()-t0, 2)
            prompt = "Analyze this: " + ("token " * {ctx//2})
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length={ctx})
            t1 = time.time()
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=64, do_sample=False)
            gen_s = round(time.time()-t1, 2)
            n = out.shape[1] - enc["input_ids"].shape[1]
            print(json.dumps({{"load_s":load_s,"gen_s":gen_s,"tokens":int(n),
                               "tok_s":round(n/max(gen_s,0.001),2)}}))
        """)
        t0 = time.time()
        rc, stdout, stderr = self._exec(script=script, timeout=2400)
        dur = round(time.time()-t0, 2)
        d   = self._parse_json_from(stdout)
        return RunResult(rc==0, dur,
                         tokens_generated=d.get("tokens",0),
                         throughput_tok_s=d.get("tok_s",0.0),
                         notes=f"load={d.get('load_s',0):.1f}s",
                         stderr=stderr[-300:])

    # ── FlexGen ─────────────────────────────────────────────────────────────

    def _run_flexgen(self, ctx: int, bs: int) -> RunResult:
        p = self.plan
        # Derive percent breakdown from capacities
        w_gb   = p.weight_gb
        cpu_pct= min(100, int(p.dram_cap_gb / w_gb * 100)) if w_gb else 0
        dsk_pct= max(0, 100 - cpu_pct)
        script = self._script("flexgen_run.py", f"""
            import subprocess, sys, json, re, time
            result = subprocess.run([
                sys.executable, "-m", "flexgen.apps.chatbot",
                "--model","{p.model.hf_id}",
                "--percent","0","{cpu_pct}","{dsk_pct}",
                "0","60","40",
                "--offload-dir","{p.offload_dir}",
                "--overlap","--sep-layer","--pin-weight",
                "--prompt-len",str({ctx}),
                "--gen-len","64",
                "--num-prompts","5",
            ], capture_output=True, text=True, timeout=1800)
            tok_s = 0.0
            for line in result.stdout.splitlines():
                m = re.search(r"throughput.*?([\d.]+)", line, re.I)
                if m: tok_s = float(m.group(1))
            print(json.dumps({{"tok_s":tok_s}}))
        """)
        t0 = time.time()
        rc, stdout, stderr = self._exec(script=script, timeout=2400)
        dur = round(time.time()-t0, 2)
        d   = self._parse_json_from(stdout)
        return RunResult(rc==0, dur,
                         throughput_tok_s=d.get("tok_s",0.0),
                         stderr=stderr[-300:])

    # ── DeepSpeed ZeRO-Infinity (checkpoint / write endurance) ───────────────

    def _run_deepspeed(self, ctx: int, bs: int) -> RunResult:
        p = self.plan
        ds_cfg = {
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {"device": "nvme",
                                      "nvme_path": p.offload_dir},
                "offload_param":     {"device": "nvme",
                                      "nvme_path": p.offload_dir},
                "overlap_comm": True, "contiguous_gradients": True,
                "sub_group_size": 1e9, "reduce_bucket_size": 1e9,
            },
            "bf16": {"enabled": p.dtype == "bf16"},
            "fp16": {"enabled": p.dtype == "fp16"},
            "train_micro_batch_size_per_gpu": bs,
            "steps_per_print": 10,
        }
        cfg_path = self.work_dir / "ds_config.json"
        cfg_path.write_text(json.dumps(ds_cfg, indent=2))
        script = self._script("ds_train.py", f"""
            import time, json, torch, deepspeed
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from datasets import load_dataset
            model = AutoModelForCausalLM.from_pretrained("{p.model.hf_id}")
            tok   = AutoTokenizer.from_pretrained("{p.model.hf_id}")
            if tok.pad_token is None: tok.pad_token = tok.eos_token
            engine,_,_,_ = deepspeed.initialize(model=model, config="{cfg_path}")
            dataset = load_dataset("wikitext","wikitext-2-raw-v1",split="train")
            texts   = dataset["text"][:60]
            total = 0; t0 = time.time()
            for step, text in enumerate(texts):
                enc = tok(text, return_tensors="pt", truncation=True,
                          max_length={ctx}, padding="max_length")
                labels = enc["input_ids"].clone()
                loss = engine(input_ids=enc["input_ids"],
                              attention_mask=enc["attention_mask"],
                              labels=labels).loss
                engine.backward(loss); engine.step()
                total += enc["input_ids"].numel()
                if step % 20 == 0 and step > 0:
                    engine.save_checkpoint("{p.offload_dir}/ckpt")
                if step >= 40: break
            dur = time.time()-t0
            print(json.dumps({{"steps":step+1,"tokens":total,
                               "tok_s":round(total/max(dur,0.001),2),
                               "dur":round(dur,2)}}))
        """)
        t0 = time.time()
        rc, stdout, stderr = self._exec(script=script, timeout=3600)
        dur = round(time.time()-t0, 2)
        d   = self._parse_json_from(stdout)
        return RunResult(rc==0, d.get("dur",dur),
                         tokens_generated=d.get("tokens",0),
                         throughput_tok_s=d.get("tok_s",0.0),
                         stderr=stderr[-400:])
