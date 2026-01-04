# ko-audio-meeting-notes-mcp (Cloudflare)

Cloudflare Workers version of the Korean audio to meeting notes MCP server.

## Overview

This is an **instruction-only** MCP server. It does NOT execute transcription or extraction - it returns instructions/patterns that guide LLMs on how to:

1. Set up the Korean Whisper model locally
2. Transcribe audio using Python
3. Extract meeting notes using Korean regex patterns

## Tools

| Tool | Description |
|------|-------------|
| `check_whisper_model` | Setup instructions for Python, ffmpeg, HuggingFace |
| `transcribe_audio` | Python script for Korean audio transcription |
| `generate_meeting_notes` | Korean regex patterns for meeting extraction |

## Development

```bash
npm install
npm run dev    # http://127.0.0.1:8787
```

## Endpoints

- `/` - Health check (JSON info)
- `/mcp` - Standard MCP protocol
- `/sse` - Server-Sent Events

## Deployment

```bash
npx wrangler login
npm run deploy
```

## Claude Desktop Config

claude mcp add --scope user --transport sse meeting-notes https://meeting-notes.agentic-worker.store/sse


```json
{
  "mcpServers": {
    "audio-to-meeting-notes": {
      "command": "npx",
      "args": ["mcp-remote", "https://meeting-notes.agentic-worker.store/sse"]
    }
  }
}
```

## Live URL

- Health check: https://meeting-notes.agentic-worker.store/
- MCP endpoint: https://meeting-notes.agentic-worker.store/mcp
- SSE endpoint: https://meeting-notes.agentic-worker.store/sse
