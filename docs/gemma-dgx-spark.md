# Running Gemma on DGX Spark with Tool Calling

This guide covers how to serve the Gemma 4 model on an NVIDIA DGX Spark and configure cobot-claw to use it with proper tool calling support.

## Overview

The cobot-claw system uses `provider: local` to point at a vLLM server running on the DGX Spark. Tool calling (function calling) works via the OpenAI-compatible `/v1/chat/completions` endpoint with `tool_choice` support — the same path used for OpenAI models.

---

## 1. Serve Gemma on DGX Spark with vLLM

First, mount the model directory into the vLLM container by exporting the Docker volume argument:

```bash
export VLLM_SPARK_EXTRA_DOCKER_ARGS="-v $HOME/ai_data:/home/dgx-spark-01/ai_data"
```

This maps `~/ai_data` on the DGX Spark host into the container at the path the model arguments reference. Must be set in the same shell session before running `launch-cluster.sh`.

Then launch the server:

```bash
./launch-cluster.sh -t vllm-node-tf5 exec vllm serve /home/dgx-spark-01/ai_data/models/gemma-4-26b-moe \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --moe-backend triton \
  --trust-remote-code \
  --max-model-len 32768 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4
```

### Key flags

| Flag | Purpose |
|---|---|
| `--tool-call-parser gemma4` | Uses the **gemma4**-specific parser in vLLM — converts the model's native tool call format into the standard OpenAI `tool_calls` array that cobot-claw (via pydantic-ai) expects |
| `--enable-auto-tool-choice` | Allows the model to decide when to call a tool (equivalent to `tool_choice: "auto"`) |
| `--distributed-executor-backend ray` | Required for multi-node or multi-process execution on the DGX Spark cluster |
| `--moe-backend triton` | Uses the Triton kernel for MoE layer dispatch — needed for correct execution of the Gemma 4 26B MoE architecture |
| `--trust-remote-code` | Required to load Gemma 4's custom model code from the local checkpoint |
| `--tensor-parallel-size 2` | Splits the model across 2 GPUs |
| `--max-model-len 32768` | Caps context length to keep VRAM usage predictable; increase if you hit context-length errors |

---

## 2. Configure cobot-claw (`cobot.gemma4.yaml`)

The repo ships with `cobot.gemma4.yaml` pre-configured for the DGX Spark setup:

```yaml
llm:
  provider: local
  base_url: http://172.31.0.94:8000/v1
  api_key: "none"          # vLLM doesn't require a key; field must be present

fleet:
  - name: ur3e
    model_type: ur3e
    ip: 192.168.1.2
    arm_mcp_url: http://172.31.0.77:8765/mcp
    gripper_mcp_url: http://172.31.0.77:8001/mcp

agents:
  orchestrator:
    model: /home/dgx-spark-01/ai_data/models/gemma-4-26b-moe
  scenario:
    model: /home/dgx-spark-01/ai_data/models/gemma-4-26b-moe
  config:
    model: /home/dgx-spark-01/ai_data/models/gemma-4-26b-moe
  safety:
    model: /home/dgx-spark-01/ai_data/models/gemma-4-26b-moe
    poll_interval_sec: 1.0
    alert_threshold: WARNING
  perception:
    enabled: false
    model: /home/dgx-spark-01/ai_data/models/gemma-4-26b-moe
```

The `model` field in each agent block is the model ID passed to vLLM — it must match the path you used when launching the server.

---

## 3. Run cobot-claw against the Gemma backend

Pass the Gemma config file with `--config`:

```bash
cobot --config cobot.gemma4.yaml chat
cobot --config cobot.gemma4.yaml teach
cobot --config cobot.gemma4.yaml run <scenario>
```

---

## 4. How tool calling flows through the stack

```
cobot-claw agent (pydantic-ai)
        │  OpenAI-compatible tool schema (JSON)
        ▼
vLLM server  (http://172.31.0.94:8000/v1)
        │  Gemma generates:  <tool_call>move_joint(...)</tool_call>
        │  vLLM pythonic parser → {"tool_calls": [{"function": {...}}]}
        ▼
pydantic-ai receives standard tool_calls array
        │
        ▼
MCP tool executed on UR3e arm / gripper
```

---

## 5. Verifying tool calling works

Run a quick sanity check directly against the vLLM endpoint:

```bash
curl http://172.31.0.94:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/home/dgx-spark-01/ai_data/models/gemma-4-26b-moe",
    "messages": [{"role": "user", "content": "Move joint 1 to 90 degrees"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "move_joint",
        "description": "Move a robot joint to a target angle in degrees",
        "parameters": {
          "type": "object",
          "properties": {
            "joint": {"type": "integer"},
            "angle_deg": {"type": "number"}
          },
          "required": ["joint", "angle_deg"]
        }
      }
    }],
    "tool_choice": "auto"
  }'
```

A successful response will contain a `tool_calls` array (not raw `<tool_call>` text), confirming the pythonic parser is active.

---

## 6. Troubleshooting

### Model returns raw `<tool_call>` text instead of parsed tool calls
- Ensure `--enable-auto-tool-choice` and `--tool-call-parser gemma4` are both set when starting vLLM.
- Restart the vLLM server and retry.

### `Connection refused` / timeout from cobot-claw
- Confirm the DGX Spark is reachable: `ping 172.31.0.94`
- Check vLLM is listening: `curl http://172.31.0.94:8000/health`
- Verify no firewall is blocking port 8000.

### CUDA OOM on DGX Spark
- Reduce `--max-model-len` (e.g. `16384`) or increase `--tensor-parallel-size` if more GPUs are available.
- Alternatively, enable quantization: add `--quantization fp8` to the vLLM launch command.

### `api_key` validation error
- The `api_key: "none"` value in `cobot.gemma4.yaml` is intentional — vLLM does not validate keys but pydantic-ai requires the field to be non-empty.
