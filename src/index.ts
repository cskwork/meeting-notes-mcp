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
import subprocess
import os
import tempfile

def convert_to_wav(audio_path):
    """m4a 등 비wav 포맷을 wav로 변환 (ffmpeg 사용)"""
    if audio_path.lower().endswith('.wav'):
        return audio_path, False

    # 임시 wav 파일 생성
    temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name

    # ffmpeg로 변환 (16kHz mono - Whisper 최적)
    cmd = [
        'ffmpeg', '-i', audio_path,
        '-ar', '16000',  # 16kHz sample rate
        '-ac', '1',       # mono
        '-y',             # overwrite
        temp_wav
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return temp_wav, True

# Initialize pipeline with Korean Whisper model
pipe = pipeline(
    "automatic-speech-recognition",
    model="seastar105/whisper-small-komixv2",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Transcribe audio file (pass as argument)
audio_path = sys.argv[1] if len(sys.argv) > 1 else "audio.m4a"

# Convert to wav if needed (m4a, mp3, etc. -> wav)
wav_path, needs_cleanup = convert_to_wav(audio_path)

try:
    result = pipe(wav_path, return_timestamps=True)
finally:
    # Cleanup temp file
    if needs_cleanup and os.path.exists(wav_path):
        os.remove(wav_path)

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

// Progressive transcription for large audio files
const PROGRESSIVE_TRANSCRIBE_INSTRUCTIONS = `# Progressive Transcription (For Long Audio)

긴 오디오 파일을 청크 단위로 나눠 점진적으로 처리합니다.

## Python Script
Save as \`transcribe_progressive.py\`:

\`\`\`python
import torch
from transformers import pipeline
import json
import sys
import subprocess
import os
import tempfile
import argparse
import math

def get_audio_duration(audio_path):
    """ffprobe로 오디오 길이(초) 추출"""
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        audio_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())

def extract_chunk(audio_path, start_sec, duration_sec, output_path):
    """ffmpeg로 오디오 청크 추출 (16kHz mono wav)"""
    cmd = [
        'ffmpeg', '-i', audio_path,
        '-ss', str(start_sec),
        '-t', str(duration_sec),
        '-ar', '16000',
        '-ac', '1',
        '-y',
        output_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)

def format_timestamp(seconds):
    """초를 MM:SS 형식으로 변환"""
    if seconds is None:
        return "??:??"
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"

def main():
    parser = argparse.ArgumentParser(description='Progressive Korean audio transcription')
    parser.add_argument('audio_path', help='Path to audio file')
    parser.add_argument('-o', '--output', default='transcript.json', help='Output JSON file')
    parser.add_argument('--chunk-duration', type=int, default=600, help='Chunk duration in seconds (default: 600 = 10min)')
    parser.add_argument('--chunk-output-dir', help='Directory to save individual chunk files')
    args = parser.parse_args()

    # Create chunk output dir if specified
    if args.chunk_output_dir:
        os.makedirs(args.chunk_output_dir, exist_ok=True)

    # Get total duration
    total_duration = get_audio_duration(args.audio_path)
    num_chunks = math.ceil(total_duration / args.chunk_duration)

    print(f"[INFO] Total duration: {format_timestamp(total_duration)}, Chunks: {num_chunks}", file=sys.stderr)

    # Initialize Whisper pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model="seastar105/whisper-small-komixv2",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    all_segments = []
    full_text = []

    for i in range(num_chunks):
        start_sec = i * args.chunk_duration
        end_sec = min((i + 1) * args.chunk_duration, total_duration)

        # Extract chunk to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            chunk_wav = tmp.name

        try:
            extract_chunk(args.audio_path, start_sec, args.chunk_duration, chunk_wav)

            # Transcribe chunk
            result = pipe(chunk_wav, return_timestamps=True)

            # Adjust timestamps and collect segments
            chunk_segments = []
            for chunk in result.get("chunks", []):
                seg_start = chunk["timestamp"][0]
                seg_end = chunk["timestamp"][1]
                # Offset by chunk start time
                adjusted_start = start_sec + (seg_start if seg_start else 0)
                adjusted_end = start_sec + (seg_end if seg_end else args.chunk_duration)

                segment = {
                    "start": adjusted_start,
                    "end": adjusted_end,
                    "text": chunk["text"]
                }
                all_segments.append(segment)
                chunk_segments.append(segment)

            full_text.append(result["text"])

            # Save chunk file if output dir specified
            if args.chunk_output_dir:
                chunk_filename = f"chunk_{i+1:03d}.txt"
                chunk_path = os.path.join(args.chunk_output_dir, chunk_filename)

                with open(chunk_path, 'w', encoding='utf-8') as f:
                    f.write(f"# Chunk {i+1}/{num_chunks} ({format_timestamp(start_sec)} - {format_timestamp(end_sec)})\\n\\n")
                    for seg in chunk_segments:
                        f.write(f"[{format_timestamp(seg['start'])}] {seg['text']}\\n")

                # Progress marker for Claude Code agent
                print(f"[CHUNK_READY] {chunk_path}", file=sys.stderr)

            print(f"[CHUNK_COMPLETE] {i+1}/{num_chunks}", file=sys.stderr)

        finally:
            if os.path.exists(chunk_wav):
                os.remove(chunk_wav)

    # Final output
    output = {
        "text": " ".join(full_text),
        "segments": all_segments,
        "metadata": {
            "total_duration": total_duration,
            "num_chunks": num_chunks,
            "chunk_duration": args.chunk_duration
        }
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[TRANSCRIPTION_COMPLETE] {args.output}", file=sys.stderr)
    print(json.dumps(output, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
\`\`\`

## Usage Examples

\`\`\`bash
# Basic progressive transcription (10min chunks)
python transcribe_progressive.py recording.m4a -o transcript.json

# Custom chunk size (5min)
python transcribe_progressive.py recording.m4a --chunk-duration 300 -o transcript.json

# Save individual chunk files (for Claude Code agent processing)
python transcribe_progressive.py recording.m4a --chunk-output-dir ./chunks -o transcript.json
\`\`\`

## Progress Markers (stderr output)

- \`[INFO] Total duration: MM:SS, Chunks: N\` - Initial info
- \`[CHUNK_READY] path/to/chunk_001.txt\` - Chunk file saved
- \`[CHUNK_COMPLETE] 1/5\` - Chunk processing done
- \`[TRANSCRIPTION_COMPLETE] transcript.json\` - All done

## Chunk TXT Format

\`\`\`
# Chunk 1/3 (00:00 - 10:00)

[00:00] 첫 번째 세그먼트 텍스트...
[00:15] 두 번째 세그먼트 텍스트...
[00:32] 세 번째 세그먼트 텍스트...
\`\`\`

## Claude Code Agent Workflow

1. **백그라운드에서 트랜스크립션 실행**
   \`\`\`bash
   python transcribe_progressive.py audio.m4a --chunk-output-dir ./chunks -o transcript.json 2>&1 | tee progress.log &
   \`\`\`

2. **각 청크 파일 감지 시 요약**
   - \`[CHUNK_READY]\` 마커 감지 (tail -f progress.log)
   - 청크 txt 읽기 -> Claude로 요약 생성 -> 부분 회의록에 추가

3. **최종 회의록 병합**
   - 모든 청크 요약 결합
   - 전체 Executive Summary 생성
   - 액션 아이템 통합`;

// Auto-summarize transcript using Claude
const SUMMARIZE_TRANSCRIPT_INSTRUCTIONS = `# Auto-Summarize Transcript

Claude를 사용하여 긴 트랜스크립트를 자동으로 회의록으로 변환합니다.

## Python Script
Save as \`summarize_transcript.py\`:

\`\`\`python
import json
import sys
import subprocess
import argparse
import os
from datetime import datetime

def format_timestamp(seconds):
    """초를 HH:MM:SS 형식으로 변환"""
    if seconds is None:
        return "??:??:??"
    h, remainder = divmod(int(seconds), 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def split_by_time(segments, chunk_minutes=10):
    """세그먼트를 시간 단위로 청크 분할"""
    chunk_seconds = chunk_minutes * 60
    chunks = []
    current_chunk = []
    current_start = 0

    for seg in segments:
        seg_start = seg.get("start", 0)
        # New chunk if past boundary
        if seg_start >= current_start + chunk_seconds and current_chunk:
            chunks.append({
                "start": current_start,
                "end": seg_start,
                "segments": current_chunk
            })
            current_chunk = []
            current_start = seg_start - (seg_start % chunk_seconds)

        current_chunk.append(seg)

    # Add remaining
    if current_chunk:
        last_end = current_chunk[-1].get("end", current_start + chunk_seconds)
        chunks.append({
            "start": current_start,
            "end": last_end,
            "segments": current_chunk
        })

    return chunks

def summarize_chunk(chunk_text, chunk_num, total_chunks):
    """Claude CLI로 청크 요약"""
    prompt = f'''다음은 회의 트랜스크립트의 {chunk_num}/{total_chunks} 부분입니다.

이 부분에서 다음을 추출해주세요:
1. 주요 논의 내용 (2-3문장)
2. 결정 사항 (있다면)
3. 액션 아이템 (담당자, 기한 포함)
4. 후속 조치 필요 사항

트랜스크립트:
{chunk_text}

간결하게 한국어로 요약해주세요.'''

    cmd = ['claude', '-p', prompt]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip()

def merge_summaries(chunk_summaries, title=None):
    """청크 요약을 최종 회의록으로 병합"""
    combined = "\\n\\n---\\n\\n".join(chunk_summaries)

    prompt = f'''다음은 회의의 각 부분별 요약입니다. 이를 하나의 완성된 회의록으로 통합해주세요.

요약들:
{combined}

다음 형식으로 최종 회의록을 작성해주세요:

# 회의록{f": {title}" if title else ""}

## 회의 정보
- 일시: {datetime.now().strftime("%Y-%m-%d")}

## Executive Summary
(전체 회의 핵심 내용 3-5문장)

## 주요 논의 사항
(번호 목록)

## 결정 사항
(번호 목록)

## 액션 아이템
| 작업 | 담당자 | 기한 | 우선순위 |
|------|--------|------|----------|

## 후속 조치
(체크리스트 형식)'''

    cmd = ['claude', '-p', prompt]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip()

def main():
    parser = argparse.ArgumentParser(description='Auto-summarize transcript to meeting notes')
    parser.add_argument('transcript', help='Path to transcript JSON file')
    parser.add_argument('-o', '--output', default='meeting-notes.md', help='Output markdown file')
    parser.add_argument('-c', '--chunk-minutes', type=int, default=10, help='Chunk size in minutes')
    parser.add_argument('-t', '--title', help='Meeting title')
    args = parser.parse_args()

    # Load transcript
    with open(args.transcript, 'r', encoding='utf-8') as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if not segments:
        # Fallback: treat full text as single segment
        segments = [{"start": 0, "end": 0, "text": data.get("text", "")}]

    # Split into chunks
    chunks = split_by_time(segments, args.chunk_minutes)
    print(f"[INFO] Processing {len(chunks)} chunks...", file=sys.stderr)

    # Summarize each chunk
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        chunk_text = "\\n".join([
            f"[{format_timestamp(s.get('start'))}] {s.get('text', '')}"
            for s in chunk["segments"]
        ])

        print(f"[SUMMARIZING] Chunk {i+1}/{len(chunks)} ({format_timestamp(chunk['start'])} - {format_timestamp(chunk['end'])})", file=sys.stderr)
        summary = summarize_chunk(chunk_text, i+1, len(chunks))
        chunk_summaries.append(summary)
        print(f"[CHUNK_SUMMARY_DONE] {i+1}/{len(chunks)}", file=sys.stderr)

    # Merge into final meeting notes
    print(f"[MERGING] Creating final meeting notes...", file=sys.stderr)
    final_notes = merge_summaries(chunk_summaries, args.title)

    # Save output
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(final_notes)

    print(f"[COMPLETE] {args.output}", file=sys.stderr)

if __name__ == "__main__":
    main()
\`\`\`

## Usage Examples

\`\`\`bash
# Basic usage (10min chunks)
python summarize_transcript.py transcript.json -o meeting-notes.md

# Custom chunk size (15min)
python summarize_transcript.py transcript.json -c 15 -o meeting-notes.md

# With meeting title
python summarize_transcript.py transcript.json -t "프로젝트 킥오프 미팅" -o meeting-notes.md
\`\`\`

## Prerequisites
- Claude CLI installed (\`npm install -g @anthropic-ai/claude-cli\` or via pipx)
- Transcript JSON with segments array

## Progress Markers

- \`[INFO] Processing N chunks...\` - Initial count
- \`[SUMMARIZING] Chunk 1/5 (00:00 - 10:00)\` - Processing chunk
- \`[CHUNK_SUMMARY_DONE] 1/5\` - Chunk complete
- \`[MERGING] Creating final meeting notes...\` - Merging phase
- \`[COMPLETE] meeting-notes.md\` - Done`;

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
      "Returns instructions for transcribing Korean audio using local Whisper model. Use progressive mode for long audio files (>10 minutes).",
      {
        audioPath: z.string().describe("Path to the audio file to transcribe"),
        outputDir: z.string().optional().describe("Optional output directory for transcript"),
        progressive: z.boolean().optional().describe("Enable progressive/chunked transcription for long audio files"),
        chunkDuration: z.number().optional().describe("Chunk duration in seconds for progressive mode (default: 600 = 10min)"),
        chunkOutputDir: z.string().optional().describe("Directory to save individual chunk files for agent processing"),
      },
      async ({ audioPath, outputDir, progressive, chunkDuration, chunkOutputDir }) => {
        // Progressive mode for long audio
        if (progressive) {
          let response = PROGRESSIVE_TRANSCRIBE_INSTRUCTIONS;

          // Add customized command
          const chunkDur = chunkDuration || 600;
          let cmd = `python transcribe_progressive.py "${audioPath}" -o transcript.json --chunk-duration ${chunkDur}`;
          if (chunkOutputDir) {
            cmd += ` --chunk-output-dir "${chunkOutputDir}"`;
          }

          response += `\n\n## Customized Command\n\`\`\`bash\n${cmd}\n\`\`\``;

          if (outputDir) {
            response += `\n\n## Save Location\nSave output to: \`${outputDir}/transcript.json\``;
          }

          return {
            content: [{ type: "text", text: response }],
          };
        }

        // Standard mode
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

    // Tool 4: Auto-summarize transcript (for long transcripts)
    this.server.tool(
      "summarize_transcript",
      "Returns instructions for auto-summarizing transcript to meeting notes using Claude CLI (for long transcripts)",
      {
        transcriptPath: z.string().describe("Path to transcript JSON file"),
        outputPath: z.string().optional().describe("Output path for meeting notes markdown (default: meeting-notes.md)"),
        chunkMinutes: z.number().optional().describe("Chunk size in minutes for summarization (default: 10)"),
        title: z.string().optional().describe("Meeting title for the final notes"),
      },
      async ({ transcriptPath, outputPath, chunkMinutes, title }) => {
        let response = SUMMARIZE_TRANSCRIPT_INSTRUCTIONS;

        // Build customized command
        const output = outputPath || "meeting-notes.md";
        const chunkMin = chunkMinutes || 10;
        let cmd = `python summarize_transcript.py "${transcriptPath}" -o "${output}" -c ${chunkMin}`;
        if (title) {
          cmd += ` -t "${title}"`;
        }

        response += `\n\n## Customized Command\n\`\`\`bash\n${cmd}\n\`\`\``;

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
            "summarize_transcript",
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
