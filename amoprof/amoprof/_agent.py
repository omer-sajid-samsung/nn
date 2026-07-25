"""
_agent.py — Real SWE-bench agent loop for AMOprof.

Implements tool-use agent execution identical to SWE-agent/Moatless:
  - Persistent Docker container per instance
  - bash / str_replace / view_file / finish tools
  - Text-fallback parser for models without structured tool calling
  - git diff patch extraction (not model text)
  - Compatible with all host-level AMOprof collectors
"""
from __future__ import annotations
import json
import logging
import re
import subprocess
import time
import urllib.request
import textwrap
from pathlib import Path

log = logging.getLogger("amoprof.agent")

# ── Tool definitions (OpenAI function-calling schema) ─────────────────────────

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Execute a bash command in the repository at /testbed. "
                "Use to read files, run tests, grep for symbols, and verify changes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string",
                                "description": "Bash command. Working directory is /testbed."}
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "str_replace",
            "description": (
                "Replace an exact string in a file with a new string. "
                "old_str must match exactly and be unique in the file. "
                "Use this for all code edits — do NOT use bash echo/sed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path":    {"type": "string", "description": "File path (relative to /testbed or absolute)."},
                    "old_str": {"type": "string", "description": "Exact string to replace (must be unique in file)."},
                    "new_str": {"type": "string", "description": "Replacement string."},
                },
                "required": ["path", "old_str", "new_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_file",
            "description": "Read a file or a range of lines from it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":       {"type": "string",  "description": "File path."},
                    "start_line": {"type": "integer", "description": "First line (1-indexed). Optional."},
                    "end_line":   {"type": "integer", "description": "Last line inclusive. Optional."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Signal that all changes are complete and the failing tests "
                "should now pass. AMOprof will extract the patch via git diff."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

AGENT_SYSTEM_PROMPT = """\
You are an expert software engineer resolving a GitHub issue.

You are working inside a Docker container at /testbed which contains the
repository checked out at the commit where the bug was introduced.

Available tools:
  bash(command)                        — run commands, read files, run tests
  str_replace(path, old_str, new_str)  — edit files precisely
  view_file(path, [start_line], [end_line]) — read file contents
  finish()                             — call when done

Strategy:
1. Run the failing tests to understand the exact error.
2. Explore the code to find the root cause.
3. Make the minimal targeted fix using str_replace.
4. Re-run the failing tests to verify.
5. Call finish() once all failing tests pass.

Rules:
- Minimal change only — do not refactor unrelated code.
- Always verify with tests before calling finish().
- If str_replace fails, use bash to inspect the file first.
"""


# ── Text-based tool call parser ───────────────────────────────────────────────
# DeepSeek-R1 distill and many open-source models were not fine-tuned for
# OpenAI structured tool calling. This parser extracts tool calls from plain
# text so the agent loop works regardless of tool-calling support.

# Matches the start of any non-bash tool invocation used as a bare call.
_TOOL_INVOKE_RE = re.compile(r"^\s*(str_replace|view_file|finish)\s*\(", re.DOTALL)


def _fence_block_to_tool(cmd: str) -> "dict | None":
    """
    If a fenced bash block contains only a single non-bash tool invocation
    (str_replace / view_file / finish), return it as the correct tool call.
    Returns None for genuine bash content so it is dispatched as bash.

    This prevents the common R1-distill mistake of wrapping tool calls inside
    ```bash ... ``` blocks, which would otherwise produce a bash(str_replace(...))
    misparse alongside the real str_replace from pattern 5.
    """
    cmd = cmd.strip()
    if not _TOOL_INVOKE_RE.match(cmd):
        return None  # genuine bash — keep as-is

    # Try to parse it as a bare function call using pattern-5 logic
    m = re.match(r"^(str_replace|view_file|finish)\s*\((.*)\)\s*$", cmd, re.DOTALL)
    if not m:
        return None
    name, raw_args = m.group(1), m.group(2).strip()

    try:
        if name == "str_replace":
            parts = re.findall(r"""(['"])(.*?)\1""", raw_args, re.DOTALL)
            if len(parts) >= 3:
                vals = [p[1] for p in parts[:3]]
                return {"function": {"name": "str_replace", "arguments": json.dumps(
                    {"path": vals[0], "old_str": vals[1], "new_str": vals[2]})},
                    "id": "fenced_sr"}
        elif name == "view_file":
            parts = re.findall(r"""(['"])(.*?)\1""", raw_args, re.DOTALL)
            nums  = [int(x) for x in re.findall(r"(?<!\w)\d+(?!\w)", raw_args)]
            if parts:
                payload: dict = {"path": parts[0][1]}
                if len(nums) >= 1: payload["start_line"] = nums[0]
                if len(nums) >= 2: payload["end_line"]   = nums[1]
                return {"function": {"name": "view_file",
                                     "arguments": json.dumps(payload)},
                        "id": "fenced_vf"}
        elif name == "finish":
            return {"function": {"name": "finish", "arguments": "{}"}, "id": "fenced_fn"}
    except Exception:
        pass
    return None


def _parse_tools_from_text(text: str) -> list[dict]:
    """
    Extract tool invocations from plain text model output.

    Patterns applied in order, with deduplication:
      1. Fenced bash blocks — content that is actually a tool call is
         promoted to the correct tool type instead of dispatched as bash.
      2. XML <bash> tags
      3. XML <str_replace><path>...</path>...</str_replace>
      4. ReAct Action / Action Input format
      5. Bare function-call syntax: bash("..."), str_replace(...), finish()
      P. Post-processing dedup — removes any bash call whose command is
         itself a tool invocation (catches multi-line fenced blocks that
         mix bash with tool calls).
    """
    calls: list[dict] = []

    # Strip DeepSeek-R1 <think>...</think> reasoning preamble
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # 1. Fenced bash blocks: ```bash / ```sh / ```shell
    #    Before appending as bash, check if the content is actually a
    #    tool invocation (str_replace/view_file/finish) and promote it.
    fence_pat = re.compile(r"```(?:bash|sh|shell)\s*\n(.*?)```", re.DOTALL)
    for m in fence_pat.finditer(clean):
        cmd = m.group(1).strip()
        if not cmd:
            continue
        promoted = _fence_block_to_tool(cmd)
        if promoted:
            calls.append(promoted)
        else:
            calls.append({
                "function": {"name": "bash", "arguments": json.dumps({"command": cmd})},
                "id": f"txt_{len(calls)}",
            })

    # 2. XML <bash> tags
    for m in re.finditer(r"<bash>(.*?)</bash>", clean, re.DOTALL):
        cmd = m.group(1).strip()
        if cmd:
            calls.append({
                "function": {"name": "bash", "arguments": json.dumps({"command": cmd})},
                "id": f"txt_{len(calls)}",
            })

    # 3. XML <str_replace> block
    sr_pat = re.compile(
        r"<str_replace>\s*<path>(.*?)</path>\s*"
        r"<old_str>(.*?)</old_str>\s*<new_str>(.*?)</new_str>\s*</str_replace>",
        re.DOTALL)
    for m in sr_pat.finditer(clean):
        calls.append({
            "function": {"name": "str_replace", "arguments": json.dumps({
                "path": m.group(1).strip(),
                "old_str": m.group(2),
                "new_str": m.group(3),
            })},
            "id": f"txt_{len(calls)}",
        })

    # 4. ReAct Action/Action Input format
    react_pat = re.compile(
        r"Action:\s*(bash|str_replace|view_file|finish)\s*\n"
        r"(?:Action Input:\s*(.*?))?(?=\nObservation:|\nAction:|\Z)",
        re.DOTALL)
    for m in react_pat.finditer(clean):
        name = m.group(1).strip()
        arg  = (m.group(2) or "").strip()
        if name == "bash":
            calls.append({
                "function": {"name": "bash", "arguments": json.dumps({"command": arg})},
                "id": f"txt_{len(calls)}",
            })
        elif name == "finish":
            calls.append({
                "function": {"name": "finish", "arguments": "{}"},
                "id": f"txt_{len(calls)}",
            })

    # 5. Bare function-call syntax often emitted by reasoning models:
    #      bash("pytest -q")
    #      str_replace('path', 'old', 'new')
    #      view_file('path', 1, 50)
    bare_fn_pat = re.compile(r"\b(bash|str_replace|view_file|finish)\s*\((.*?)\)", re.DOTALL)
    for m in bare_fn_pat.finditer(clean):
        name = m.group(1).strip()
        raw_args = (m.group(2) or "").strip()
        try:
            if name == "bash":
                mm = re.match(r"""\s*(['"])(.*)\1\s*$""", raw_args, re.DOTALL)
                if mm:
                    calls.append({
                        "function": {"name": "bash", "arguments": json.dumps({"command": mm.group(2)})},
                        "id": f"txt_{len(calls)}",
                    })
            elif name == "str_replace":
                parts = re.findall(r"""(['"])(.*?)\1""", raw_args, re.DOTALL)
                if len(parts) >= 3:
                    vals = [p[1] for p in parts[:3]]
                    calls.append({
                        "function": {"name": "str_replace", "arguments": json.dumps({
                            "path": vals[0], "old_str": vals[1], "new_str": vals[2],
                        })},
                        "id": f"txt_{len(calls)}",
                    })
            elif name == "view_file":
                parts = re.findall(r"""(['"])(.*?)\1""", raw_args, re.DOTALL)
                nums = [int(x) for x in re.findall(r"(?<![\w])\d+(?![\w])", raw_args)]
                if len(parts) >= 1:
                    payload = {"path": parts[0][1]}
                    if len(nums) >= 1:
                        payload["start_line"] = nums[0]
                    if len(nums) >= 2:
                        payload["end_line"] = nums[1]
                    calls.append({
                        "function": {"name": "view_file", "arguments": json.dumps(payload)},
                        "id": f"txt_{len(calls)}",
                    })
            elif name == "finish":
                calls.append({
                    "function": {"name": "finish", "arguments": "{}"},
                    "id": f"txt_{len(calls)}",
                })
        except Exception:
            pass

    # P. Post-processing: remove any bash call whose entire command is a tool
    #    invocation — this catches multi-line fenced blocks where the promoted
    #    path above didn't fire (e.g. block has multiple tool calls on separate
    #    lines).  We keep the first occurrence of each (tool, key-args) pair
    #    and drop exact duplicates.
    filtered: list[dict] = []
    seen_sigs: set[str] = set()
    for c in calls:
        fn = c["function"]["name"]
        try:
            args_obj = json.loads(c["function"]["arguments"])
        except Exception:
            args_obj = {}

        # Drop bash calls that are just a tool invocation
        if fn == "bash":
            cmd = args_obj.get("command", "")
            if _TOOL_INVOKE_RE.match(cmd):
                continue  # misparse from fenced block — skip

        # Dedup by (tool_name, discriminating_arg)
        if fn == "bash":
            sig = f"bash:{args_obj.get('command','')[:120]}"
        elif fn == "str_replace":
            sig = f"str_replace:{args_obj.get('path','')}:{args_obj.get('old_str','')[:60]}"
        elif fn == "view_file":
            sig = f"view_file:{args_obj.get('path','')}:{args_obj.get('start_line',0)}"
        elif fn == "finish":
            sig = "finish"
        else:
            sig = f"{fn}:{json.dumps(args_obj)[:80]}"

        if sig not in seen_sigs:
            seen_sigs.add(sig)
            filtered.append(c)

    calls = filtered

    # Finish-fallback: if nothing else was found, check for a bare finish() mention
    if not calls and re.search(r"\bfinish\(\)", clean):
        calls.append({
            "function": {"name": "finish", "arguments": "{}"},
            "id": "txt_finish",
        })

    return calls


# ── Docker container session ──────────────────────────────────────────────────

# Ordered list of candidate repo paths inside SWE-bench Docker images.
# Standard images use /testbed; some third-party (e.g. jefzda/sweap-images)
# use /repo, /workspace, or /app.
_WORKDIR_CANDIDATES = ["/testbed", "/repo", "/workspace", "/app", "/home/user/repo"]


class _DockerAgentSession:
    """
    Manages a single Docker container kept alive for the full agent loop.
    The repo is located by probing candidate paths at startup — not hardcoded
    to /testbed — so PRO split images (jefzda/sweap-images:*) work correctly.
    """

    def __init__(self, image: str, work_dir: Path, timeout: int = 120):
        self.image    = image
        self.work_dir = work_dir
        self.timeout  = timeout
        self._cid     = ""
        self._repodir = "/testbed"  # updated in _start()

    def __enter__(self):
        self._start()
        return self

    def __exit__(self, *_):
        self._stop()

    # ── Workdir detection ─────────────────────────────────────────────────

    def _detect_repodir(self) -> str:
        """
        Find the git repository root inside the running container.

        Strategy (fastest first):
          1. Ask git itself — works even if the repo is in a non-standard path.
          2. Fall back to probing _WORKDIR_CANDIDATES in order.

        Called once during _start() so the overhead (~100 ms) is paid only once
        per instance, not on every bash() call.
        """
        # Strategy 1: git rev-parse from each candidate
        probe_cmd = " || ".join(
            f"git -C {p} rev-parse --show-toplevel 2>/dev/null"
            for p in _WORKDIR_CANDIDATES
        )
        try:
            r = subprocess.run(
                ["docker", "exec", self._cid, "bash", "-c", probe_cmd],
                capture_output=True, text=True, timeout=15)
            wd = r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""
            if wd and not wd.startswith("fatal"):
                log.debug(f"Container repo root (git): {wd}")
                return wd
        except Exception:
            pass

        # Strategy 2: probe candidate dirs
        for candidate in _WORKDIR_CANDIDATES:
            try:
                r = subprocess.run(
                    ["docker", "exec", self._cid, "test", "-d", candidate],
                    capture_output=True, timeout=5)
                if r.returncode == 0:
                    log.debug(f"Container repo root (probe): {candidate}")
                    return candidate
            except Exception:
                pass

        log.warning("Could not detect repo dir in container — defaulting to /testbed")
        return "/testbed"

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def _start(self):
        try:
            r = subprocess.run(
                ["docker", "run", "-d",
                 "--network", "none",
                 "--memory", "8g",
                 "--cpus", "4",
                 self.image, "sleep", "3600"],
                capture_output=True, text=True, timeout=300)
            if r.returncode != 0:
                log.warning(f"docker run failed: {r.stderr[:200]}")
                return
            self._cid = r.stdout.strip()
            log.debug(f"Agent container started: {self._cid[:12]}")
            self._repodir = self._detect_repodir()
            log.info(f"  container repo dir: {self._repodir}")
        except Exception as e:
            log.warning(f"Container start failed: {e}")

    def _stop(self):
        if self._cid:
            subprocess.run(["docker", "rm", "-f", self._cid],
                           capture_output=True, timeout=15)
            self._cid = ""

    def alive(self) -> bool:
        return bool(self._cid)

    # ── Tool implementations ──────────────────────────────────────────────

    def bash(self, command: str) -> tuple[str, int]:
        if not self._cid:
            return "Container not running.", 1
        try:
            r = subprocess.run(
                ["docker", "exec", self._cid,
                 "bash", "-c",
                 f"cd {self._repodir} && ({command}) 2>&1 | head -300"],
                capture_output=True, text=True, timeout=self.timeout)
            return (r.stdout + r.stderr), r.returncode
        except subprocess.TimeoutExpired:
            return f"Timed out after {self.timeout}s.", 1
        except Exception as e:
            return f"bash error: {e}", 1

    def str_replace(self, path: str, old_str: str, new_str: str) -> tuple[str, int]:
        if not self._cid:
            return "Container not running.", 1
        # Resolve relative paths against the repo root
        if not path.startswith("/"):
            path = f"{self._repodir}/{path}"
        script = (
            "import sys\n"
            "p, o, n = sys.argv[1], sys.argv[2], sys.argv[3]\n"
            "t = open(p).read()\n"
            "c = t.count(o)\n"
            "if c == 0: sys.exit('ERROR: old_str not found in ' + p)\n"
            "if c > 1:  sys.exit(f'ERROR: old_str found {c}x — must be unique')\n"
            "open(p,'w').write(t.replace(o, n, 1))\n"
            "print('OK: replaced in', p)\n"
        )
        try:
            r = subprocess.run(
                ["docker", "exec", "-i", self._cid,
                 "python3", "-c", script, path, old_str, new_str],
                capture_output=True, text=True, timeout=30)
            return (r.stdout + r.stderr).strip(), r.returncode
        except Exception as e:
            return f"str_replace error: {e}", 1

    def view_file(self, path: str, start_line: int = 0, end_line: int = 0) -> tuple[str, int]:
        if not self._cid:
            return "Container not running.", 1
        if not path.startswith("/"):
            path = f"{self._repodir}/{path}"
        if start_line > 0 and end_line > 0:
            cmd = f"sed -n '{start_line},{end_line}p' {path}"
        elif start_line > 0:
            cmd = f"tail -n +{start_line} {path} | head -200"
        else:
            cmd = f"cat -n {path} | head -300"
        return self.bash(cmd)

    def git_diff(self) -> str:
        if not self._cid:
            return ""
        out, rc = self.bash("git diff HEAD")
        if rc == 0 and out.strip():
            return out.strip()
        out2, _ = self.bash("git diff")
        return out2.strip()

    def apply_test_patch(self, test_patch: str) -> bool:
        """
        Apply test_patch and commit it as a baseline commit so it does NOT
        appear in the final git diff HEAD (which starts from that commit).
        Model edits via str_replace go to the working tree and show cleanly.
        """
        if not test_patch.strip():
            return True
        try:
            r = subprocess.run(
                ["docker", "exec", "-i", self._cid,
                 "bash", "-c",
                 f"cd {self._repodir} && git apply - && "
                 "git add -A && "
                 "git commit -m 'test_patch_baseline' --no-verify -q 2>/dev/null || true"],
                input=test_patch, capture_output=True, text=True, timeout=30)
            if r.returncode != 0:
                # Fallback: plain apply without commit
                r2 = subprocess.run(
                    ["docker", "exec", "-i", self._cid,
                     "bash", "-c", f"cd {self._repodir} && git apply -"],
                    input=test_patch, capture_output=True, text=True, timeout=30)
                return r2.returncode == 0
            return True
        except Exception:
            return False


# ── Agent loop ────────────────────────────────────────────────────────────────

def run_agent_loop(
    server_mid:      str,
    port:            int,
    instance:        dict,
    max_tokens:      int,
    call_timeout:    int,
    is_reasoning:    bool,
    work_dir:        Path,
    docker_image:    str  = "",
    max_steps:       int  = 30,
    step_timeout:    int  = 120,
    hbm_snapshot_fn       = None,
    fallback_fn           = None,
) -> dict:
    """
    Real SWE-bench agent loop using tool calling with text fallback.

    The model calls bash/str_replace/view_file/finish against a live Docker
    container. If the model does not support structured tool calling (e.g.
    DeepSeek-R1 distill), tool calls are parsed from plain text output.

    Patch is extracted via git diff HEAD, not from model text output.
    """
    from . import bench_swebench as _bsw

    iid        = instance.get("instance_id", "?")
    problem    = instance.get("problem_statement", "")
    hints      = instance.get("hints_text", "")
    test_patch = instance.get("test_patch", "")
    if hints:
        problem = problem.strip() + "\n\nHints:\n" + hints.strip()

    from .bench_swebench import _extract_test_ids
    test_ids    = _extract_test_ids(instance)
    test_id_str = "\n".join(test_ids[:8]) if test_ids else "(see test_patch)"
    first_tests = " ".join(test_ids[:3]) if test_ids else "tests/"

    hbm_snap = hbm_snapshot_fn or _bsw._hbm_snapshot

    # Metric accumulators
    total_pt = total_ot = 0
    total_dur = total_ttft = 0.0
    all_itl: list[float] = []
    hbm_pf_deltas: list[float] = []
    hbm_dc_deltas: list[float] = []
    steps_taken    = 0
    finish_called  = False
    final_patch    = ""
    steps_log: list[dict] = []

    if not docker_image:
        log.warning(f"[{iid}] No docker_image — falling back to _run_turns")
        if fallback_fn:
            return fallback_fn(
                server_mid, port, instance,
                num_turns=1, max_tokens=max_tokens,
                call_timeout=call_timeout, is_reasoning=is_reasoning,
                work_dir=work_dir)
        return {}

    with _DockerAgentSession(docker_image, work_dir, timeout=step_timeout) as session:
        if not session.alive():
            log.warning(f"[{iid}] Docker container failed to start — falling back")
            if fallback_fn:
                return fallback_fn(
                    server_mid, port, instance,
                    num_turns=1, max_tokens=max_tokens,
                    call_timeout=call_timeout, is_reasoning=is_reasoning,
                    work_dir=work_dir)
            return {}

        session.apply_test_patch(test_patch)

        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"GitHub Issue:\n{problem}\n\n"
                f"Tests that must pass after your fix:\n{test_id_str}\n\n"
                f"Start by running the failing tests to see the error."
            )},
        ]

        for step in range(max_steps):
            steps_taken    = step + 1
            hbm_before, _  = hbm_snap()

            # ── Model call ────────────────────────────────────────────────
            payload = json.dumps({
                "model":       server_mid,
                "messages":    messages,
                "max_tokens":  max_tokens,
                "temperature": 0.0,
                "tools":       AGENT_TOOLS,
                "tool_choice": "auto",
            }).encode()

            ttft_step = 0.0
            tpot_step = 0.0
            t0 = time.time()
            try:
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    data=payload,
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=call_timeout) as resp:
                    data = json.loads(resp.read())
                    dur  = round(time.time() - t0, 3)
                    # Extract real latency from SGLang extended usage fields
                    usage = data.get("usage", {})
                    ttft_step = float(usage.get("time_to_first_token_ms",
                                      usage.get("prompt_ms", dur * 300)))
                    tpot_step = float(usage.get("time_per_output_token_ms",
                                      usage.get("inter_token_latency_ms", 0.0)))
                    ot_step   = usage.get("completion_tokens", 0)
                    if tpot_step == 0.0 and ot_step > 1 and dur > 0:
                        ttft_s    = ttft_step / 1000.0 if ttft_step > 0 else dur * 0.2
                        tpot_step = round(max(0.0, dur - ttft_s) / max(ot_step, 1) * 1000, 2)
                    all_itl.append(tpot_step)
            except Exception as e:
                log.error(f"  [step {steps_taken}] model call failed: {e}")
                break

            hbm_after_pf, _ = hbm_snap()
            hbm_pf_deltas.append(round(hbm_after_pf - hbm_before, 2))

            usage     = data.get("usage", {})
            pt        = usage.get("prompt_tokens", 0)
            ot        = usage.get("completion_tokens", 0)
            total_pt  += pt
            total_ot  += ot
            total_dur += dur
            total_ttft += ttft_step

            choice   = (data.get("choices") or [{}])[0]
            message  = choice.get("message", {})
            finish_r = choice.get("finish_reason", "")

            messages.append({"role": "assistant", **{
                k: v for k, v in message.items() if k != "role"}})

            # ── Tool dispatch ─────────────────────────────────────────────
            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                text = message.get("content", "") or ""
                # Try to parse tool calls from plain text (for R1-distill etc.)
                parsed = _parse_tools_from_text(text)
                if parsed:
                    log.info(f"  [step {steps_taken}] parsed {len(parsed)} tool(s) from text")
                    tool_calls = parsed
                else:
                    # Check for a unified diff patch in the text
                    if _bsw._validate_patch(text):
                        final_patch = _bsw._extract_patch(text)
                        log.info(f"  [step {steps_taken}] extracted patch from text ({len(final_patch)}B)")
                        break
                    if finish_r == "stop":
                        log.info(f"  [step {steps_taken}] model stopped — no tool call, no patch")
                        break
                    # Nudge toward tool use
                    messages.append({"role": "user", "content": (
                        "Please use a tool. Example: run the failing tests with bash:\n"
                        f"  command: python -m pytest {first_tests} -x 2>&1 | tail -30"
                    )})
                    continue

            tool_results = []
            for tc in tool_calls:
                fn    = tc.get("function", {}).get("name", "")
                tc_id = tc.get("id", f"tc_{len(tool_results)}")
                try:
                    args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                except Exception:
                    args = {}

                if fn == "bash":
                    cmd    = args.get("command", "echo empty")
                    result, rc = session.bash(cmd)
                    result = result[-3000:] if len(result) > 3000 else result
                    # Prefix with exit code — model must know whether the command
                    # succeeded so it can adapt rather than repeating failed steps.
                    result = f"[exit {rc}]\n{result}" if result.strip() else f"[exit {rc}]"
                    log.info(f"  [step {steps_taken}] bash rc={rc}  {cmd[:70]!r}")

                elif fn == "str_replace":
                    result, rc = session.str_replace(
                        args.get("path", ""),
                        args.get("old_str", ""),
                        args.get("new_str", ""))
                    result = f"[exit {rc}]\n{result}" if result.strip() else f"[exit {rc}]"
                    log.info(f"  [step {steps_taken}] str_replace {args.get('path','')!r} rc={rc}")

                elif fn == "view_file":
                    result, rc = session.view_file(
                        args.get("path", ""),
                        args.get("start_line", 0),
                        args.get("end_line", 0))
                    result = result[-3000:] if len(result) > 3000 else result
                    result = f"[exit {rc}]\n{result}" if result.strip() else f"[exit {rc}]"

                elif fn == "finish":
                    finish_called = True
                    final_patch   = session.git_diff()
                    result = f"Patch extracted ({len(final_patch)} bytes)."
                    log.info(f"  [step {steps_taken}] finish() — patch={bool(final_patch)}")

                else:
                    result = f"Unknown tool '{fn}'."
                    rc = 1

                tool_results.append({
                    "role":         "tool",
                    "tool_call_id": tc_id,
                    "content":      str(result),
                })
                steps_log.append({
                    "step": steps_taken, "tool": fn,
                    "rc":   locals().get("rc", 0),
                    "result_len": len(str(result)),
                })

            messages.extend(tool_results)
            hbm_after_dc, _ = hbm_snap()
            hbm_dc_deltas.append(round(hbm_after_dc - hbm_after_pf, 2))

            if finish_called:
                break

        if not final_patch:
            final_patch = session.git_diff()
            log.info(f"[{iid}] Final git diff: {len(final_patch)} bytes")

        patch_path = work_dir / "agent.patch"
        patch_path.write_text(final_patch or "")
        (work_dir / "agent_trace.json").write_text(json.dumps(steps_log, indent=2))

    n_itl  = max(len(all_itl), 1)
    tpot_m = round(sum(all_itl) / n_itl, 2) if all_itl else 0.0
    tpot_p = 0.0
    if all_itl:
        s = sorted(all_itl)
        tpot_p = round(s[min(int(len(s) * 0.99), len(s) - 1)], 2)

    n_pf = max(len(hbm_pf_deltas), 1)
    n_dc = max(len(hbm_dc_deltas), 1)

    return {
        "final_patch":          final_patch,
        "prompt":               messages[1].get("content", "") if len(messages) > 1 else "",
        "has_code":             True,
        "total_prompt_tokens":  total_pt,
        "total_output_tokens":  total_ot,
        "total_duration_s":     round(total_dur, 3),
        "ttft_ms":              round(total_ttft / max(steps_taken, 1), 1),
        "tpot_mean_ms":         tpot_m,
        "tpot_p99_ms":          tpot_p,
        "num_turns_completed":  steps_taken,
        "turn_timeline":        json.dumps(steps_log),
        "resolved_in_loop":     finish_called,
        "hbm_prefill_delta_gb": round(sum(hbm_pf_deltas) / n_pf, 3),
        "hbm_decode_delta_gb":  round(sum(hbm_dc_deltas)  / n_dc, 3),
    }


# ════════════════════════════════════════════════════════════════════════════════
# ── SWE-agent and mini-swe-agent backends ─────────────────────────────────────
#
# Both agents live as git submodules inside scaleapi/SWE-bench_Pro-os:
#
#   SWE-bench_Pro-os/
#     SWE-agent/          ← scaleapi/SWE-agent @ 402a7b8  (SWE-agent v1.x fork)
#     mini-swe-agent/     ← scaleapi/mini-swe-agent @ d74716a
#     helper_code/
#       gather_patches.py ← collects .pred files → patches.json
#
# Users must clone the repo with submodules:
#   git clone --recurse-submodules https://github.com/scaleapi/SWE-bench_Pro-os
# and set SWEAP_REPO_ROOT to the checkout path.
#
# ── SWE-agent output format ─────────────────────────────────────────────────
# sweagent run --output_dir <dir> writes per-instance .pred files:
#   <dir>/<run_id>/<iid>/<iid>.pred   — plain git diff text
# Step count from .traj file at same path.
#
# ── mini-swe-agent output format ────────────────────────────────────────────
# mini-extra swebench-single --output <file.traj.json> --exit-immediately
# writes a single trajectory JSON, and appends to preds.json:
#   preds.json lines: {"instance_id": "...", "model_patch": "...", ...}
# Trajectory .traj.json: {"messages": [...], "info": {...}}
# ════════════════════════════════════════════════════════════════════════════════

import os as _os

# Environment variable users set to their SWE-bench_Pro-os checkout
_SWEAP_REPO_ROOT_ENV = "SWEAP_REPO_ROOT"


def _sweap_repo_root() -> "Path | None":
    """Return Path to SWE-bench_Pro-os checkout, or None if not configured."""
    v = _os.environ.get(_SWEAP_REPO_ROOT_ENV, "").strip()
    if v:
        p = Path(v)
        if p.exists():
            return p
    return None


def _find_submodule_executable(submodule_name: str, exec_names: list[str]) -> "list[str] | None":
    """
    Try to find an executable from a submodule in SWE-bench_Pro-os.

    Search order:
      1. <SWEAP_REPO_ROOT>/<submodule_name>/  (installed as editable or has scripts)
      2. System PATH (pip-installed version)

    Returns a base command list, or None if not found.
    """
    import sys as _sys

    root = _sweap_repo_root()

    # ── Try submodule venv / installed entry points ─────────────────────────
    if root:
        sub_dir = root / submodule_name
        if sub_dir.exists():
            # Try each exec name in the submodule's own bin dirs first
            for name in exec_names:
                for candidate in [
                    sub_dir / ".venv" / "bin" / name,
                    sub_dir / "venv"  / "bin" / name,
                    sub_dir / "bin"   / name,
                ]:
                    if candidate.exists():
                        return [str(candidate)]
            # Fall back: run via python in the submodule
            for name in exec_names:
                # Convert dashes to underscores for module name
                mod = name.replace("-", "_")
                # Check if module is importable from submodule's src
                src_init = sub_dir / "src" / mod / "__init__.py"
                pkg_init = sub_dir / mod / "__init__.py"
                if src_init.exists() or pkg_init.exists():
                    return [_sys.executable, "-m", mod]

    # ── Try system PATH ──────────────────────────────────────────────────────
    for name in exec_names:
        try:
            r = subprocess.run([name, "--help"],
                               capture_output=True, timeout=5)
            return [name]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        # python -m fallback
        mod = name.replace("-", "_")
        try:
            r = subprocess.run([_sys.executable, "-m", mod, "--help"],
                               capture_output=True, timeout=5)
            return [_sys.executable, "-m", mod]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    return None




# ════════════════════════════════════════════════════════════════════════════════
# ── SWE-agent v1.1.0 backend ─────────────────────────────────────────────────
#
# Source of truth: https://swe-agent.com/latest/installation/keys/#using-local-models
#
# For local non-function-calling models the correct config is:
#
#   agent:
#     model:
#       name: openai/<model_id>      # litellm openai-compat provider prefix
#       api_base: http://...:port/v1
#       api_key: EMPTY
#       per_instance_cost_limit: 0   # must be 0 — can't track local cost
#       total_cost_limit: 0
#       per_instance_call_limit: 0   # no limit
#       max_input_tokens: 0          # disable token check
#     tools:
#       parse_function:
#         type: thought_action       # non-FC models: extract last ```block```
#     history_processors: []         # REMOVE cache_control — Claude-only, breaks others
#
# Strategy: generate ONE config YAML per run in the output directory.
# All parallel sweagent subprocesses share it read-only — no conflict.
# Pass: --config config/default.yaml --config amoprof_sweagent.yaml
# (v1.1.0 merges configs hierarchically, so amoprof overrides just model+tools)
# ════════════════════════════════════════════════════════════════════════════════

_SWEAGENT_DATASET_IDS = {
    "lite":     "princeton-nlp/SWE-bench_Lite",
    "verified": "princeton-nlp/SWE-bench_Verified",
    "full":     "princeton-nlp/SWE-bench",
    "pro":      "ScaleAI/SWE-bench_Pro",
}

# Models without native function calling — need thought_action parser
_NON_FC_PATTERNS = (
    "deepseek", "r1", "qwq", "qwen",
    "llama", "mistral", "gemma", "phi",
    "starcoder", "codestral", "yi-",
)


def _is_non_fc_model(model_id: str) -> bool:
    return any(p in model_id.lower() for p in _NON_FC_PATTERNS)


import yaml

def _sweagent_write_config(
    server_mid: str,
    port: int,
    output_dir: Path,
) -> Path:
    """
    Writes a valid SWE-agent config for local SGLang models.
    Uses yaml.dump to ensure zero syntax or indentation errors.
    """
    use_thought_action = _is_non_fc_model(server_mid)
    
    # Determine the model name prefix
    litellm_model = f"openai/{server_mid}" if not server_mid.startswith("openai/") else server_mid

    # Define the configuration as a Python dictionary
    config_dict = {
        "agent": {
            "type": {
                "default"
            },
            "model": {
                "name": litellm_model,
                "api_base": f"http://127.0.0.1:{port}/v1",
                "api_key": "EMPTY",
                "per_instance_cost_limit": 0,
                "total_cost_limit": 0,
                "per_instance_call_limit": 0,
                "max_input_tokens": 0,
            },
            "config": {
                "parse_function": "thought_action" if use_thought_action else "function_calling",
                # Hardcoding default templates to ensure the agent doesn't start with 0 steps
                "system_template": (
                    "Settling into the role of a software engineer, you will solve tasks "
                    "using a bash shell. Explore the repo, run tests, and fix the issue."
                ),
                "instance_template": (
                    "We're in {{working_dir}}. Solve this issue:\n{{problem_statement}}"
                ),
            },
            "history_processors": [],
        }
    }

    # Write the YAML safely
    #config_path = output_dir / "amoprof_sweagent.yaml"
    config_path = Path(output_dir) / "amoprof_sweagent.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        # sort_keys=False keeps it readable for humans
        yaml.dump(config_dict, f, sort_keys=False, default_flow_style=False)

    return config_path

def _sweagent_default_yaml(base_cmd: list) -> "str | None":
    """
    Find config/default.yaml inside the SWE-agent installation.
    This provides the system_template / instance_template that sweagent needs.
    Without it sweagent warns 'system_template is not set'.
    """
    root = _sweap_repo_root()
    if root:
        p = root / "SWE-agent" / "config" / "default.yaml"
        if p.exists():
            return str(p)

    # Try sweagent package directory
    try:
        import sweagent as _sa
        for candidate in [
            Path(_sa.__file__).parent.parent / "config" / "default.yaml",
            Path(_sa.__file__).parent / "config" / "default.yaml",
        ]:
            if candidate.exists():
                return str(candidate)
    except Exception:
        pass

    # Try sibling of the sweagent executable
    if base_cmd and not base_cmd[0].endswith("python3") and not base_cmd[0].endswith("python"):
        p = Path(base_cmd[0]).resolve().parent.parent / "config" / "default.yaml"
        if p.exists():
            return str(p)

    return None


def _sweagent_make_instance_jsonl(instance: dict, work_dir: Path) -> Path:
    """
    Write the per-instance JSONL for --instances.type=file.

    Per https://swe-agent.com/latest/usage/batch_mode/#loading-instances-from-a-file:
    Required: instance_id, problem_statement, image_name
    The image_name field ("jefzda/sweap-images:<tag>") drives Docker.
    """
    iid = instance.get("instance_id", "unknown")
    dockerhub_tag = instance.get("dockerhub_tag", "")
    record = {
        "instance_id":       iid,
        "problem_statement": instance.get("problem_statement", ""),
        "repo_name":         instance.get("repo", ""),
        "base_commit":       instance.get("base_commit", "HEAD"),
        "image_name":        f"jefzda/sweap-images:{dockerhub_tag}",
    }
    jsonl_path = work_dir / f"{iid}.jsonl"
    jsonl_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return jsonl_path


def _sweagent_find_traj(output_dir: Path, iid: str) -> "Path | None":
    """
    Find the trajectory file written by run-batch.
    Per docs: <output_dir>/<instance_id>.traj  (flat path)
    Also search recursively as fallback.
    """
    direct = Path(output_dir) / f"{iid}.traj"
    if direct.exists():
        return direct
    candidates = sorted(
        Path(output_dir).rglob(f"{iid}.traj"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _sweagent_find_preds(output_dir: Path, iid: str) -> str:
    """Fallback: extract model_patch from all_preds.jsonl / preds.jsonl."""
    for fname in ("all_preds.jsonl", "preds.jsonl"):
        candidates = [output_dir / fname] + list(Path(output_dir).rglob(fname))
        for p in candidates:
            if not p.exists():
                continue
            try:
                for line in p.read_text().splitlines():
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    if rec.get("instance_id") == iid:
                        return rec.get("model_patch", "")
            except Exception:
                pass
    return ""


def _parse_sweagent_traj(traj_path: Path) -> dict:
    """Parse a SWE-agent trajectory defensively across schema variants."""
    def _first_num(*vals):
        for v in vals:
            if isinstance(v, (int, float)):
                return v
            if isinstance(v, str):
                try:
                    return float(v)
                except Exception:
                    pass
        return 0
    try:
        traj = json.loads(traj_path.read_text())
        info = traj.get("info", {}) if isinstance(traj, dict) else {}
        stats = info.get("model_stats", {}) if isinstance(info, dict) else {}
        trajectory = traj.get("trajectory", []) if isinstance(traj, dict) else []
        if not isinstance(trajectory, list):
            trajectory = []
        timeline = json.dumps([
            {
                "step": i + 1,
                "thought": str((s or {}).get("thought", ""))[:80],
                "action": str((s or {}).get("action", ""))[:80],
                "observation": str((s or {}).get("observation", ""))[:80],
            }
            for i, s in enumerate(trajectory[:50])
            if isinstance(s, dict)
        ])
        tokens_sent = int(_first_num(
            stats.get("tokens_sent"),
            stats.get("prompt_tokens"),
            stats.get("input_tokens"),
            info.get("tokens_sent"),
            info.get("prompt_tokens"),
            info.get("input_tokens"),
        ) or 0)
        tokens_received = int(_first_num(
            stats.get("tokens_received"),
            stats.get("completion_tokens"),
            stats.get("output_tokens"),
            info.get("tokens_received"),
            info.get("completion_tokens"),
            info.get("output_tokens"),
        ) or 0)
        patch = ""
        for src in (info, traj):
            if isinstance(src, dict):
                patch = src.get("submission") or src.get("patch") or src.get("final_patch") or patch
        exit_status = ""
        for src in (info, traj):
            if isinstance(src, dict):
                exit_status = src.get("exit_status") or src.get("status") or exit_status
        raw_stats = json.dumps({
            "info_keys": sorted(list(info.keys())) if isinstance(info, dict) else [],
            "model_stats": stats if isinstance(stats, dict) else {},
            "traj_keys": sorted(list(traj.keys())) if isinstance(traj, dict) else [],
        }, sort_keys=True)
        return {
            "patch": patch or "",
            "exit_status": exit_status or "",
            "steps": len(trajectory),
            "tokens_sent": tokens_sent,
            "tokens_received": tokens_received,
            "resolved": (exit_status == "submitted") or bool(patch),
            "timeline": timeline,
            "raw_stats": raw_stats,
        }
    except Exception as e:
        log.debug(f"sweagent traj parse error ({traj_path}): {e}")
        return {"raw_stats": json.dumps({"parse_error": str(e)})}


def _run_sweagent(
    server_mid: str,
    port: int,
    instance: dict,
    max_tokens: int,
    call_timeout: int,
    work_dir: Path,
    docker_image: str,
    max_steps: int,
    hbm_snapshot_fn=None,
    split: str = "lite",
    sweagent_options: dict | None = None,
) -> dict:
    """Run SWE-agent on one instance using the original instances file plus a slice.

    This preserves SWE-agent's native file parsing behavior while still letting
    AMOprof drive concurrency by launching multiple sweagent subprocesses, each
    with a one-element instances slice.
    """
    from . import bench_swebench as _bsw

    iid      = instance.get("instance_id", "?")
    hbm_snap = hbm_snapshot_fn or _bsw._hbm_snapshot

    # ── Find sweagent ──────────────────────────────────────────────────────
    base_cmd = _find_submodule_executable("SWE-agent", ["sweagent"])
    if base_cmd is None:
        log.warning(f"[{iid}] sweagent not found — "
                    f"set SWEAP_REPO_ROOT or: pip install sweagent")
        return {}

    # ── Output and config paths ───────────────────────────────────────────
    output_dir = Path(work_dir) / "sweagent_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    sweagent_options = sweagent_options or {}

    model_name = sweagent_options.get("model_name") or server_mid
    api_base = sweagent_options.get("api_base") or f"http://127.0.0.1:{port}/v1"
    api_key = sweagent_options.get("api_key") or "EMPTY"
    max_input_tokens = int(sweagent_options.get("max_input_tokens") or max_tokens or 50000)
    num_workers = max(1, int(sweagent_options.get("num_workers") or 1))
    instances_type = sweagent_options.get("instances_type") or "file"
    redo_existing = bool(sweagent_options.get("redo_existing", True))
    shuffle = bool(sweagent_options.get("shuffle", False))

    cfg_override = sweagent_options.get("config_path")
    amoprof_cfg = Path(cfg_override) if cfg_override else None

    default_cfg = sweagent_options.get("default_config") or _sweagent_default_yaml(base_cmd)
    if not default_cfg:
        log.warning(
            f"[{iid}] sweagent: config/default.yaml not found. "
            f"Agent will have no task instructions. "
            f"Set SWEAP_REPO_ROOT to your SWE-bench_Pro-os checkout, "
            f"or set SWEAGENT_CONFIG env var to the path of default.yaml.")
        default_cfg = _os.environ.get("SWEAGENT_CONFIG", "")

    instances_path = (
        sweagent_options.get("instances_path")
        or instance.get("__amoprof_instances_path")
        or ""
    )
    if not instances_path:
        log.warning(f"[{iid}] sweagent: missing instances_path for sliced batch launch")
        return {}
    instance_index = int(instance.get("__amoprof_index", 0) or 0)
    instance_slice = sweagent_options.get("instances_slice") or f"{instance_index}:{instance_index + 1}"

    # ── Build command ─────────────────────────────────────────────────────
    cmd = base_cmd + ["run-batch"]
    if default_cfg:
        cmd += ["--config", str(default_cfg)]
    if amoprof_cfg:
        cmd += ["--config", str(amoprof_cfg)]
    cmd += [
        "--instances.type", str(instances_type),
        "--instances.path", str(instances_path),
        "--instances.slice", str(instance_slice),
        "--instances.shuffle", "True" if shuffle else "False",
        "--agent.model.name", str(model_name),
        "--agent.model.api_base", str(api_base),
        "--agent.model.api_key", str(api_key),
        "--agent.model.per_instance_cost_limit", "0",
        "--agent.model.max_input_tokens", str(max_input_tokens),
        "--num_workers", "1",
        "--redo_existing", "True" if redo_existing else "False",
        "--output_dir", str(output_dir),
    ]

    log.info(f"[{iid}] sweagent run-batch: {' '.join(str(x) for x in cmd[:12])} ...")
    log.debug(f"[{iid}] sweagent full cmd: {cmd}")

    # Env vars as safety net — litellm also reads these
    env = {
        **_os.environ,
        "OPENAI_API_KEY":  str(api_key),
        "OPENAI_API_BASE": str(api_base),
        "OPENAI_BASE_URL": str(api_base),
    }

    # ── Execute ───────────────────────────────────────────────────────────
    hbm_before, _ = hbm_snap()
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            env=env, timeout=call_timeout + max_steps * 60)
        dur = round(time.time() - t0, 3)
    except subprocess.TimeoutExpired:
        log.warning(f"[{iid}] sweagent timed out after "
                    f"{call_timeout + max_steps * 60}s")
        dur = round(time.time() - t0, 3)
        result = type("R", (), {"returncode": 1, "stdout": "", "stderr": "timeout"})()
    hbm_after, _ = hbm_snap()

    stdout_txt = result.stdout or ""
    stderr_txt = result.stderr or ""
    (work_dir / "sweagent_stdout.txt").write_text(stdout_txt)
    (work_dir / "sweagent_stderr.txt").write_text(stderr_txt)

    if result.returncode != 0:
        combined = stderr_txt or stdout_txt
        if dur < 30:
            log.error(
                f"[{iid}] sweagent COMMAND FAILED "
                f"(rc={result.returncode}, dur={dur:.1f}s).\n"
                f"  cmd: {' '.join(str(x) for x in cmd)}\n"
                f"  stderr: {combined[:1000]}\n"
                f"  generated config: {amoprof_cfg}\n"
                f"  Full log: {work_dir}/sweagent_stderr.txt")
        else:
            log.warning(f"[{iid}] sweagent exit rc={result.returncode}  "
                        f"tail: {stderr_txt[-300:]}")

    # ── Parse trajectory ──────────────────────────────────────────────────
    traj_path = _sweagent_find_traj(Path(output_dir), iid)
    traj_data: dict = {}
    if traj_path:
        traj_data = _parse_sweagent_traj(traj_path)
        log.info(f"[{iid}] sweagent traj: {traj_path.name}  "
                 f"exit={traj_data.get('exit_status','?')}  "
                 f"steps={traj_data.get('steps',0)}  "
                 f"patch={bool(traj_data.get('patch'))}")
    else:
        log.warning(f"[{iid}] sweagent: no .traj under {traj_path}")

    final_patch = traj_data.get("patch", "")
    if not final_patch:
        final_patch = _sweagent_find_preds(Path(output_dir), iid)
        if final_patch:
            log.info(f"[{iid}] sweagent: used preds fallback")

    log.info(f"[{iid}] sweagent done: rc={result.returncode}  "
             f"patch={bool(final_patch)}  steps={traj_data.get('steps',0)}  "
             f"{dur:.1f}s")

    return {
        "final_patch":          final_patch,
        "prompt":               instance.get("problem_statement", "")[:500],
        "has_code":             bool(final_patch),
        "total_prompt_tokens":  traj_data.get("tokens_sent",     0),
        "total_output_tokens":  traj_data.get("tokens_received", 0),
        "total_duration_s":     dur,
        "ttft_ms":              0.0,
        "tpot_mean_ms":         0.0,
        "tpot_p99_ms":          0.0,
        "num_turns_completed":  traj_data.get("steps",    0),
        "turn_timeline":        traj_data.get("timeline", "[]"),
        "resolved_in_loop":     traj_data.get("resolved", False),
        "raw_traj_stats_json":  traj_data.get("raw_stats", ""),
        "hbm_prefill_delta_gb": round(hbm_after - hbm_before, 3),
        "hbm_decode_delta_gb":  0.0,
    }


def _run_mini_sweagent(
    server_mid: str,
    port: int,
    instance: dict,
    max_tokens: int,
    call_timeout: int,
    work_dir: Path,
    docker_image: str,
    max_steps: int,
    hbm_snapshot_fn=None,
    split: str = "lite",
) -> dict:
    """
    Run mini-swe-agent from the scaleapi/SWE-bench_Pro-os submodule (or system PATH).

    Invokes mini-extra swebench-single for per-instance runs:
        OPENAI_API_BASE=http://127.0.0.1:<port>/v1 OPENAI_API_KEY=EMPTY
        mini-extra swebench-single \
          --subset <subset_name_or_hf_id> \
          --split test \
          --model openai/<server_mid> \
          -i <instance_id> \
          --output <work_dir>/traj.json \
          --exit-immediately

    Patch extracted from trajectory info.model_patch or preds.json.
    Token counts from trajectory info.model_stats.

    Set SWEAP_REPO_ROOT to your SWE-bench_Pro-os checkout to use the
    pinned submodule version.
    """
    from . import bench_swebench as _bsw

    iid      = instance.get("instance_id", "?")
    hbm_snap = hbm_snapshot_fn or _bsw._hbm_snapshot

    # ── Find mini-extra executable ────────────────────────────────────────
    base_cmd = _find_submodule_executable(
        "mini-swe-agent",
        ["mini-extra", "mini_extra", "minisweagent"])

    if base_cmd is None:
        log.warning(f"[{iid}] mini-extra not found (SWEAP_REPO_ROOT={_os.environ.get(_SWEAP_REPO_ROOT_ENV, 'not set')})")
        return {}

    # Determine the right --subset value: either a shortname mini knows, or
    # the full HuggingFace dataset ID for Pro
    subset = _MINI_SUBSET_NAMES.get(split, split)

    traj_out = Path(work_dir) / f"{iid}.traj.json"
    cmd = base_cmd + [
        "swebench-single",
        "--subset",   subset,
        "--split",    "test",
        "--model",    f"openai/{server_mid}",
        "-i",         iid,
        "--output",   str(traj_out),
        "--exit-immediately",
    ]
    # docker environment class if a specific image is requested
    if docker_image:
        cmd += ["--environment-class", "docker"]

    log.info(f"[{iid}] mini-swe-agent: {' '.join(cmd[:5])} ...")

    env = {**_os.environ,
           "OPENAI_API_KEY":  "EMPTY",
           "OPENAI_API_BASE": f"http://127.0.0.1:{port}/v1",
           # litellm also respects these
           "LLM_PROVIDER":    "openai",
           "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1"}

    hbm_before, _ = hbm_snap()
    t0 = time.time()
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True,
            env=env, timeout=call_timeout + max_steps * 60)
        dur = round(time.time() - t0, 3)
    except subprocess.TimeoutExpired:
        log.warning(f"[{iid}] mini-swe-agent timed out")
        dur = round(time.time() - t0, 3)
        r = type("R", (), {"returncode": 1, "stdout": "", "stderr": "timeout"})()
    hbm_after, _ = hbm_snap()

    (work_dir / "mini_stdout.txt").write_text(r.stdout or "")
    (work_dir / "mini_stderr.txt").write_text(r.stderr or "")

    # ── Extract patch from trajectory ─────────────────────────────────────
    traj_data: dict = {}
    if traj_out.exists():
        traj_data = _mini_parse_traj(traj_out)
    else:
        # Try searching for any .traj.json written by mini
        for p in Path(work_dir).rglob("*.traj.json"):
            traj_data = _mini_parse_traj(p)
            break

    final_patch  = traj_data.get("patch", "")
    steps_taken  = traj_data.get("steps", 0)
    tokens_prompt= traj_data.get("prompt_tokens", 0)
    tokens_out   = traj_data.get("output_tokens", 0)
    resolved     = traj_data.get("resolved", False)

    # Also check preds.json as fallback (batch mode writes here)
    if not final_patch:
        for preds_path in [work_dir / "preds.json",
                           Path.cwd() / "preds.json"]:
            if preds_path.exists():
                final_patch = _mini_parse_preds_json(preds_path, iid)
                if final_patch:
                    break

    log.info(f"[{iid}] mini-swe-agent: rc={r.returncode}  "
             f"patch={bool(final_patch)}  steps={steps_taken}  {dur:.1f}s")

    return {
        "final_patch":          final_patch,
        "prompt":               instance.get("problem_statement", "")[:500],
        "has_code":             bool(final_patch),
        "total_prompt_tokens":  tokens_prompt,
        "total_output_tokens":  tokens_out,
        "total_duration_s":     dur,
        "ttft_ms":              0.0,
        "tpot_mean_ms":         0.0,
        "tpot_p99_ms":          0.0,
        "num_turns_completed":  steps_taken,
        "turn_timeline":        traj_data.get("timeline", "[]"),
        "resolved_in_loop":     resolved,
        "hbm_prefill_delta_gb": round(hbm_after - hbm_before, 3),
        "hbm_decode_delta_gb":  0.0,
    }


AGENT_BACKENDS = ("none", "amoprof", "sweagent", "mini")

def dispatch_agent(
    backend: str,
    server_mid: str,
    port: int,
    instance: dict,
    max_tokens: int,
    call_timeout: int,
    is_reasoning: bool,
    work_dir: Path,
    docker_image: str,
    max_steps: int,
    step_timeout: int,
    hbm_snapshot_fn=None,
    fallback_fn=None,
    split: str = "lite",
    sweagent_options: dict | None = None,
) -> dict:
    """
    Unified entry point for all three agent backends.

    backend values
    --------------
    "none"      — no agent loop; caller should use _run_turns() directly
    "amoprof"    — AMOprof homegrown tool-use loop (bash/str_replace/view_file/finish)
    "sweagent"  — princeton-nlp/SWE-agent via CLI subprocess
    "mini"      — mini-swe-agent via Python API with CLI fallback

    If a backend is unavailable (not installed) it logs a warning and falls
    back to the amoprof backend, then to fallback_fn if that also fails.

    All backends return the same dict contract:
        final_patch, has_code, total_prompt_tokens, total_output_tokens,
        total_duration_s, ttft_ms, tpot_mean_ms, tpot_p99_ms,
        num_turns_completed, turn_timeline, resolved_in_loop,
        hbm_prefill_delta_gb, hbm_decode_delta_gb
    """
    if backend == "none":
        raise ValueError("dispatch_agent called with backend='none'; "
                         "caller should use _run_turns() instead")

    iid = instance.get("instance_id", "?")

    if backend == "sweagent":
        result = _run_sweagent(
            server_mid=server_mid, port=port, instance=instance,
            max_tokens=max_tokens, call_timeout=call_timeout,
            work_dir=work_dir, docker_image=docker_image,
            max_steps=max_steps, hbm_snapshot_fn=hbm_snapshot_fn,
            split=split, sweagent_options=sweagent_options)
        if result:
            return result
        log.warning(f"[{iid}] sweagent failed — falling back to amoprof agent")
        backend = "amoprof"

    if backend == "mini":
        result = _run_mini_sweagent(
            server_mid=server_mid, port=port, instance=instance,
            max_tokens=max_tokens, call_timeout=call_timeout,
            work_dir=work_dir, docker_image=docker_image,
            max_steps=max_steps, hbm_snapshot_fn=hbm_snapshot_fn,
            split=split, sweagent_options=sweagent_options)
        if result:
            return result
        log.warning(f"[{iid}] mini-swe-agent failed — falling back to amoprof agent")
        backend = "amoprof"

    if backend == "amoprof":
        result = run_agent_loop(
            server_mid=server_mid, port=port, instance=instance,
            max_tokens=max_tokens, call_timeout=call_timeout,
            is_reasoning=is_reasoning, work_dir=work_dir,
            docker_image=docker_image, max_steps=max_steps,
            step_timeout=step_timeout, hbm_snapshot_fn=hbm_snapshot_fn,
            fallback_fn=fallback_fn)
        if result:
            return result

    # All backends failed — use the plain _run_turns fallback
    log.warning(f"[{iid}] All agent backends failed — falling back to _run_turns")
    if fallback_fn:
        return fallback_fn(
            server_mid, port, instance,
            num_turns=1, max_tokens=max_tokens,
            call_timeout=call_timeout, is_reasoning=is_reasoning,
            work_dir=work_dir)
    return {}
