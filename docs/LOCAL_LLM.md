# Optional local matchup analysis

OctogonML uses a local Ollama provider by default for evidence-linked matchup explanations. Standard forecast models run without it. The optional AI probability overlay is separate from the ML baseline and is **experimental, unvalidated, and limited to ±5 percentage points**. No reported winner accuracy includes this overlay.

## New machine

1. Install Ollama from its [official download page](https://ollama.com/download). Start the Ollama application/service. If you use only the CLI, `ollama serve` starts it in the foreground; do not start a second copy if the application already owns the port.
2. Download the model explicitly:

   ```bash
   ollama pull qwen3:4b
   ```

3. Configure the Python service environment and start OctogonML:

   ```bash
   export CAGEMETRICS_LLM_PROVIDER=ollama
   export CAGEMETRICS_LLM_MODEL=qwen3:4b
   export OLLAMA_URL=http://127.0.0.1:11434
   cagemetrics serve
   ```

No OpenAI/API key is needed for Ollama. A clone does not include weights or an Ollama binary. The 4B Q4_K_M model is approximately 2.5 GB; the prepared check used Apple M4 with 16 GiB RAM. Resource needs and latency depend on context size, concurrent work, and hardware. API configuration requires a loopback Ollama URL. Host Ollama is not a container's `127.0.0.1`; this setup assumes both services run on the same host.

Check `http://127.0.0.1:11434/api/version` and `/api/tags` to distinguish a reachable server from an installed model. Model output is validated separately. A timeout or failed evidence check must remain an error rather than a fabricated explanation.

## Prepared workspace runtime

This workspace also has a verified portable runtime under `work/local-llm/`, outside the application repository. Its binary and models are **not** shipped by Git. On that prepared workspace only:

```bash
./scripts/start_local_llm.sh
```

The script checks for the existing binary, selects workspace model storage, binds `127.0.0.1:11434`, and disables cloud access/history. `CAGEMETRICS_LOCAL_LLM_DIR` can point to a different already-prepared runtime directory. It does not install missing software or download weights. Stop a foreground server with Ctrl-C. Ollama uses its normal small `~/.ollama` application key/configuration directory; this is separate from bulky workspace weights.

The [official Ollama 0.40.8 release](https://github.com/ollama/ollama/releases/tag/v0.40.8) supplied `ollama-darwin.tgz`. Its SHA-256 matches the published asset digest:

```
29fce20c736f62aa8885b1e83552c5fd8dcc63edc5ffc1b47f2de591a06cde44
```

[Qwen3:4b](https://ollama.com/library/qwen3:4b) is 4.02B parameters, Q4_K_M, licensed Apache 2.0. The observed model manifest digest is:

```
359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7
```

`ollama pull` verifies blob digests. Workspace `setup.json`, `server.log`, and `smoke_test.json` retain local setup evidence; these are machine-specific files outside the repository.

## Verification status and request bounds

The basic actual `/api/chat` check used a JSON schema, `stream: false`, `think: false`, temperature zero, 4,096-token context, and bounded generation. It returned valid JSON, preserved a supplied baseline probability, and emitted no thinking content. Cold elapsed time was **10.7 seconds**, including a **5.6-second load**, with approximately **8.8 generated tokens/second**. Ollama reported all 37 layers on Apple M4 Metal and approximately 3.18 GB allocated model/context memory. This is one setup observation, not a capacity benchmark.

**Complete local analyses passed in both fighter orientations**, in **37.5 and 56.6 seconds** (approximately 38/57 seconds), separately from the basic JSON check. The checked prompt is `grounded-matchup-v3-ai5pp-catalogue-2bbad14bab04`. The LLM selects sourced advantages, an exact comparison, and an eligible conditional tactical route from a server-authored catalogue, retaining the original fact IDs. Positive observed metrics determine route eligibility; zero takedowns do not establish takedown defense. Missing support allows explicit abstention. A valid actual LLM response is required: the catalogue is not a fabricated failure fallback. Rationale and the bounded probability proposal remain LLM output.

Both checked orientations proposed zero percentage-point adjustment and preserved the complementary 78.1%/21.9% ML baseline. These are successful workflow examples, not proof of universal semantic correctness, calibrated AI probabilities, or reliability under every machine/load. Timeouts and grounding rejections remain explicit failure states. The prompt version includes a hash of the entire analysis implementation, so selection changes invalidate the cache. The service allows at most a 90-second provider timeout and bounded local concurrency; keep output/context budgets appropriate for the machine.

The app's analysis endpoint accepts saved matching prediction IDs only. It builds facts on the server, verifies source/date alignment, validates citations/identities/numbers, and stores successful evidence/output immutably. Numerical and schema checks do not prove semantic truth. The original ML forecast remains unchanged, even when experimental adjusted probabilities are displayed.

Official protocol references: [Ollama chat API](https://docs.ollama.com/api/chat) and [structured output documentation](https://docs.ollama.com/capabilities/structured-outputs).

## Optional OpenAI provider

For a later hosted setup, configure the server explicitly:

```bash
export CAGEMETRICS_LLM_PROVIDER=openai
export OPENAI_MODEL=your-enabled-model-id
export OPENAI_API_KEY=your-server-side-key
cagemetrics serve
```

Use an enabled model that supports the adapter's structured Responses API contract. The provider is optional; no OpenAI account, key, or successful hosted-provider test is implied by local setup. Keep keys in the server environment/secret manager, never browser code or Git. The adapter uses the official HTTPS API endpoint. Restart after changing provider configuration.
