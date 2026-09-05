# vtdb-mcp server

A minimal [FastMCP](https://gofastmcp.com) server (streamable-http at `/mcp`,
unauthenticated `/health` next to it), modeled on
`~/exen-mcp/packages/server`. Demo tools: `ping`, `echo`, `add`,
`server_info`.

## Run locally

```bash
uv sync                      # needs UV_INDEX_BIOTURING_* from repo .envrc
uv run server                # or: uv run python -m server
# -> http://127.0.0.1:8000/mcp   /health for probes
```

Config via flags or env: `--host/--port/--mcp-path/--debug`
(`VTDB_HOST`, `VTDB_PORT`, `VTDB_MCP_PATH`, `VTDB_DEBUG`).

## Docker

Context is this directory; the private-index token is a BuildKit secret:

```bash
export UV_INDEX_BIOTURING_PASSWORD=...   # in repo .envrc
docker build \
  --secret id=bioturing_token,env=UV_INDEX_BIOTURING_PASSWORD \
  -f docker/Dockerfile -t vtdb-mcp-server:dev .
```

## Full stack (server + Grafana Alloy/Loki/Grafana)

Runs on Kubernetes — see the header comments in `../../k8s/kustomization.yaml`:

```bash
kind load docker-image vtdb-mcp-server:dev   # after the docker build above
kubectl apply -k k8s/
kubectl -n vtdb-mcp port-forward svc/grafana 3000:3000   # admin / admin
# -> dashboard “vtdb-mcp · Logs”
```
