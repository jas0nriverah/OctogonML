# Matchup analysis providers

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

The enriched packet passed a subsequent live `POST /api/v1/matchup-analysis` check in **61.5 seconds**, using prompt `grounded-matchup-v3-ai5pp-catalogue-daa3e583dd29`. Its [saved response](../reports/matchup-analysis-local-enriched.json) retains 63 facts, with 24 supplied to Ollama, including both frozen strength summaries and latest covered results. It proposed zero adjustment, preserved the ML baseline, and returned the same immutable record through GET and a repeated POST. Tests ran concurrently, so this observation does not provide a controlled latency comparison with the earlier checks. It used Ollama only; no OpenAI request was made.

The app's analysis endpoint accepts saved matching prediction IDs only. It builds facts on the server, verifies source/date alignment, validates citations/identities/numbers, and stores successful evidence/output immutably. Numerical and schema checks do not prove semantic truth. The original ML forecast remains unchanged, even when experimental adjusted probabilities are displayed.

Official protocol references: [Ollama chat API](https://docs.ollama.com/api/chat) and [structured output documentation](https://docs.ollama.com/capabilities/structured-outputs).

## OpenAI provider

The native service can use OpenAI instead of Ollama. Run this from the project directory:

```bash
cagemetrics setup-openai
```

The command reports the key-file path and whether a key is configured; it never prints the secret. By default it creates `.secrets/openai_api_key`, with mode `0600` for a new file, and preserves an existing file. The parent directory is private when newly created. Open the reported file in your editor and save only the key text: no quotes, variable assignment, or extra spaces. Do not paste the key into chat or browser code.

Start the server after saving the file:

```bash
cagemetrics serve --llm-provider openai
```

The command selects `openai` and defaults `OPENAI_MODEL` to [`gpt-6.1-sol`](https://developers.openai.com/api/docs/models/gpt-6.1-sol). An existing `OPENAI_MODEL` value wins. The model supports structured output through the Responses API; account access and API usage billing still apply. Ordinary `cagemetrics serve` keeps the configured provider, defaulting to Ollama when none is set. Saving a key alone never enables OpenAI requests, and the local provider never reads that credential. Restart the Python service after changing provider, model, or credential configuration.

Credential precedence is:

1. A nonempty `OPENAI_API_KEY` environment secret.
2. The file selected by `OPENAI_API_KEY_FILE`.
3. `.secrets/openai_api_key` under the project root (`CAGEMETRICS_HOME` when configured).

An explicitly selected missing or invalid file is an error rather than a silent fallback. `OPENAI_API_KEY_FILE` also selects the file created by `setup-openai`. Keep keys server-side. The project's `.secrets/` directory is excluded from Git, Docker build context, and source archives. Container deployments must supply their own server environment secret or explicitly mounted private file; the image does not contain the native key file.

### Evidence and generation bounds

The OpenAI path uses a larger server-built evidence package than the small local model. It does not accept client-written facts or browse for new information during a request.

| Provider | Evidence selection | Output budget |
|---|---|---|
| Ollama | Up to 24 facts and 6,500 characters | 640 generated tokens |
| OpenAI | Up to 80 facts and 24,000 UTF-8 bytes | 4,096 output tokens by default |

The local packet prioritizes supported tactical facts, sample sizes, frozen strength indices, and the latest covered results. Redundant URLs, hashes, and source metadata stay in the full saved audit record rather than consuming local context. Fact IDs, text, identities, and relevant dates remain intact. OpenAI receives the full bounded packet.

OpenAI instructions allow up to three sourced advantages per athlete, four uncertainty options, and six supported route choices. Missing observations permit abstention. Additional context does not relax identity, numerical, citation, or source/date validation. Successful responses retain the full evidence hash and the selected evidence IDs, plus available input/output/total token usage and cached/reasoning token counts. The OpenAI output budget includes reasoning and visible output.

Optional server environment settings:

```bash
export CAGEMETRICS_OPENAI_MAX_OUTPUT_TOKENS=4096
export CAGEMETRICS_OPENAI_REASONING_EFFORT=low
```

The output budget accepts integers from `256` through `8192`. For `gpt-6.1-sol`, reasoning defaults to `low`; supported efforts are `low`, `medium`, `high`, `xhigh`, and `max`. An explicitly selected unknown model receives no default reasoning setting unless configured. These are request controls, not evidence of improved accuracy. Restart after changing them.

**Verification:** local Ollama workflow checks above passed. A live OpenAI request for this setup remains pending. Neither provider selection nor a successful API response establishes predictive improvement for the experimental overlay.
