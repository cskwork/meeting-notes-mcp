import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { McpAgent } from "agents/mcp";
import { z } from "zod";

// Instruction content for each tool
const CHECK_WHISPER_INSTRUCTIONS = `# Korean Whisper Model Setup Instructions

## Required Dependencies
1. Python 3.8+
2. ffmpeg (for audio processing)
3. Python packages:
   - transformers
   - torch
   - librosa
   - accelerate

## Check Commands
\`\`\`bash
# Check Python
python --version  # Windows
python3 --version # macOS/Linux

# Check ffmpeg
ffmpeg -version

# Check Python packages
pip show transformers torch librosa accelerate
\`\`\`

## Installation Commands
\`\`\`bash
# Install ffmpeg (if missing)
# Windows: winget install ffmpeg
# macOS: brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg

# Install Python packages
pip install transformers torch librosa accelerate
\`\`\`

## Model Info
- Model ID: \`seastar105/whisper-small-komixv2\`
- WER: 7.36% (Korean optimized, better than whisper-large-v3)
- Auto-downloads on first use (~500MB to ~/.cache/huggingface)
- CUDA auto-detection for GPU acceleration`;

const TRANSCRIBE_INSTRUCTIONS = `# Audio Transcription Instructions

## Supported Audio Formats
m4a, caf, aac, mp3, ogg, 3gp, amr, wav, webm, flac

## Python Transcription Script
Save as \`transcribe.py\` and run with your audio file:

\`\`\`python
import torch
from transformers import pipeline
import json
import sys

# Initialize pipeline with Korean Whisper model
pipe = pipeline(
    "automatic-speech-recognition",
    model="seastar105/whisper-small-komixv2",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Transcribe audio file (pass as argument)
audio_path = sys.argv[1] if len(sys.argv) > 1 else "audio.m4a"
result = pipe(audio_path, return_timestamps=True)

# Output JSON with segments
output = {
    "text": result["text"],
    "segments": [
        {
            "start": chunk["timestamp"][0],
            "end": chunk["timestamp"][1],
            "text": chunk["text"]
        }
        for chunk in result.get("chunks", [])
    ]
}

print(json.dumps(output, ensure_ascii=False, indent=2))
\`\`\`

## Usage
\`\`\`bash
python transcribe.py your_audio_file.m4a > transcript.json
\`\`\`

## Expected Output Format
\`\`\`json
{
  "text": "Full transcript text here...",
  "segments": [
    {"start": 0.0, "end": 2.5, "text": "First segment"},
    {"start": 2.5, "end": 5.0, "text": "Second segment"}
  ]
}
\`\`\``;

const MEETING_NOTES_INSTRUCTIONS = `# Meeting Notes Extraction Patterns (Korean)

Use these regex patterns to extract meeting information from Korean transcripts.

## Decision Patterns (Regex)
\`\`\`
/(?:결정|합의|결론)(?:했|됐|되었|하기로)/i
/(?:~로|으로)\\s*(?:정|확정|결정)(?:했|됐)/i
/(?:하기로\\s*(?:했|결정|합의))/i
/(?:진행하겠습니다|진행할게요|진행하기로)/i
/(?:채택|선택|선정)(?:했|됐|되었)/i
\`\`\`

## Action Item Patterns (Regex)
\`\`\`
/(?:제가|내가|저는)\\s+.+?(?:하겠습니다|할게요|할게|하겠어요)/i
/.+?(?:님|씨|팀장|부장|과장|대리|사원)(?:이|가)?\\s*.+?(?:해\\s*주세요|해주실|담당|맡아)/i
/(?:다음|이번)\\s*(?:주|주까지|까지).+?(?:해야|합니다|해주세요)/i
/.+?(?:확인|검토|작성|준비|정리)(?:해\\s*주세요|해야|하겠습니다)/i
/.+?(?:보내|전달|공유)(?:주세요|드리겠습니다|할게요)/i
\`\`\`

## Owner Extraction
\`\`\`
/(.{1,10}?)(?:님|씨|팀장|부장|과장|대리|사원|매니저)/  -> Extract name
/(?:제가|내가|저는)/  -> Maps to "본인" (self)
\`\`\`

## Deadline Patterns
\`\`\`
/(?:이번\\s*주|다음\\s*주|금주|차주)(?:\\s*까지)?/i
/(?:월요일|화요일|수요일|목요일|금요일|토요일|일요일)(?:\\s*까지)?/i
/(\\d{1,2})월\\s*(\\d{1,2})일(?:\\s*까지)?/i
/(?:오늘|내일|모레)(?:\\s*까지)?/i
/(?:ASAP|긴급|급하게|빨리|당장)/i
\`\`\`

## Priority Keywords
- **High**: 긴급, 급하게, 급히, 시급, 중요, 핵심, 필수, 우선, 먼저, 최우선, 블로커, 꼭, 반드시, 필히
- **Low**: 나중에, 추후, 여유있으면, 시간날때, 가능하면, 천천히, 서두르지, 참고

## Follow-up Patterns
\`\`\`
/(?:확인|체크|검토)(?:해\\s*봐야|필요|해야)/i
/(?:다음\\s*회의|다음에|나중에)\\s*(?:논의|얘기|이야기)/i
/(?:알아보고|조사하고|찾아보고)\\s*(?:공유|알려)/i
/(?:추가\\s*논의|재논의)\\s*(?:필요|예정)/i
/(?:모니터링|지켜봐야|추적)/i
\`\`\`

## Markdown Output Template
\`\`\`markdown
# 회의록

## 회의 정보
- **일시:** [DATE]
- **소요시간:** [DURATION]

## 결정 사항
- [Extract sentences matching decision patterns]

## 액션 아이템
| 작업 | 담당자 | 기한 | 우선순위 |
|------|--------|------|----------|
| [task] | [owner or TBD] | [deadline or TBD] | [High/Normal/Low] |

## 후속 조치
- [ ] [Extract sentences matching follow-up patterns]

## 전체 트랜스크립트
<details>
<summary>펼쳐서 보기</summary>

[Full transcript text]

</details>
\`\`\``;

// MCP Agent with instruction tools
export class MyMCP extends McpAgent {
  server = new McpServer({
    name: "audio-to-meeting-notes",
    version: "1.0.0",
  });

  async init() {
    // Tool 1: Check Whisper model setup
    this.server.tool(
      "check_whisper_model",
      "Returns instructions for checking and installing Korean Whisper model dependencies",
      {},
      async () => ({
        content: [{ type: "text", text: CHECK_WHISPER_INSTRUCTIONS }],
      })
    );

    // Tool 2: Transcribe audio
    this.server.tool(
      "transcribe_audio",
      "Returns instructions for transcribing Korean audio using local Whisper model",
      {
        audioPath: z.string().describe("Path to the audio file to transcribe"),
        outputDir: z.string().optional().describe("Optional output directory for transcript"),
      },
      async ({ audioPath, outputDir }) => {
        const customizedInstructions = TRANSCRIBE_INSTRUCTIONS.replace(
          'audio_path = sys.argv[1] if len(sys.argv) > 1 else "audio.m4a"',
          `audio_path = "${audioPath}"`
        );

        let response = customizedInstructions;
        if (outputDir) {
          response += `\n\n## Save Location\nSave output to: \`${outputDir}/transcript.json\``;
        }

        return {
          content: [{ type: "text", text: response }],
        };
      }
    );

    // Tool 3: Generate meeting notes
    this.server.tool(
      "generate_meeting_notes",
      "Returns Korean regex patterns and instructions for extracting meeting notes from transcript",
      {
        transcriptPath: z.string().optional().describe("Path to transcript JSON file"),
        transcriptText: z.string().optional().describe("Direct transcript text to process"),
      },
      async ({ transcriptPath, transcriptText }) => {
        let response = MEETING_NOTES_INSTRUCTIONS;

        if (transcriptPath) {
          response += `\n\n## Input Source\nRead transcript from: \`${transcriptPath}\``;
        }

        if (transcriptText) {
          response += `\n\n## Provided Transcript\n\`\`\`\n${transcriptText.substring(0, 500)}${transcriptText.length > 500 ? '...' : ''}\n\`\`\``;
        }

        return {
          content: [{ type: "text", text: response }],
        };
      }
    );
  }
}

// Cloudflare Worker fetch handler
export default {
  fetch(request: Request, env: Env, ctx: ExecutionContext) {
    const url = new URL(request.url);

    // SSE endpoint for streaming
    if (url.pathname === "/sse" || url.pathname === "/sse/message") {
      return MyMCP.serveSSE("/sse").fetch(request, env, ctx);
    }

    // Standard MCP endpoint
    if (url.pathname === "/mcp") {
      return MyMCP.serve("/mcp").fetch(request, env, ctx);
    }

    // Health check / info
    if (url.pathname === "/") {
      return new Response(
        JSON.stringify({
          name: "audio-to-meeting-notes",
          version: "1.0.0",
          description: "Audio to meeting notes MCP server (instruction-only, Korean optimized)",
          endpoints: {
            mcp: "/mcp",
            sse: "/sse",
          },
          tools: [
            "check_whisper_model",
            "transcribe_audio",
            "generate_meeting_notes",
          ],
        }),
        {
          headers: { "Content-Type": "application/json" },
        }
      );
    }

    return new Response("Not found", { status: 404 });
  },
};
