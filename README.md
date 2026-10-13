# llm_provider_base

BanyanOS llm_engine provider plugin base (ProviderContext, host hooks, plugin metadata).

Part of the BanyanOS LLM provider plugin architecture — see
`agent_engine/docs/LLM_PROVIDER_PLUGIN_DEVELOPMENT_PLAN.md` (D7, D8, D9).
Ported from the SilvaEngine `ai_agent_handler` package; the SilvaEngine package is
unchanged and still serves `ai_agent_core_engine`.

## Contents

- `ProviderContext` — host-injected per-call context (pooled HTTP client,
  credentials, endpoint, emit / cancel hooks).
- `LlmProviderHandler` — base class every provider plugin subclasses;
  plugin metadata (`api_protocol`, `replaces_protocol`, `capabilities`,
  `default_pool_name`) and the `configuration_schema.json` loader.

## Tests

```bash
pytest
ruff check .
```
