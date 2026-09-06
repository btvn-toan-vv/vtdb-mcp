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

Config via flags or env: `--host/--port/--mcp-path/--debug/--reload`
(`VTDB_HOST`, `VTDB_PORT`, `VTDB_MCP_PATH`, `VTDB_DEBUG`, `VTDB_RELOAD`).

## Develop (dev compose stack)

`../../compose.dev.yaml` runs the server + Grafana Alloy/Loki/Grafana on
plain Docker with the source bind-mounted and `VTDB_RELOAD=1` — edit a tool,
save, uvicorn restarts the worker in ~1s (no image rebuild; rebuild only when
`uv.lock` changes):

```bash
docker compose -f compose.dev.yaml up -d --build
docker compose -f compose.dev.yaml down       # add -v to wipe dev log data
```

## Docker

## Docker

Context is this directory; the private-index token is a BuildKit secret:

```bash
export UV_INDEX_BIOTURING_PASSWORD=...   # in repo .envrc
docker build \
  --secret id=bioturing_token,env=UV_INDEX_BIOTURING_PASSWORD \
  -f docker/Dockerfile -t vtdb-mcp-server:dev .
```

## Deploy (server + Grafana Alloy/Loki/Grafana)

Deployment path is Kubernetes — see the header comments in
`../../k8s/kustomization.yaml`:

```bash
kind load docker-image vtdb-mcp-server:dev   # after the docker build above
kubectl apply -k k8s/
kubectl -n vtdb-mcp port-forward svc/grafana 3000:3000   # admin / admin
# -> dashboard “vtdb-mcp · Logs”
```
