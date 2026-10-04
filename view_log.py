#!/usr/bin/env python3
"""
Visualise a Prisoner's Dilemma game log as a messenger-style chat replay.

Reasoning, moves, and post-game dialogue all appear as messages in one thread.

Usage:
    python view_log.py G100/game_log_20260829_064833.json
    python view_log.py G100/game_log_20260829_064833.json --port 8765
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Timer


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Prisoner's Dilemma — Chat Replay</title>
  <style>
    :root {
      --bg: #0b0e14;
      --panel: #12151c;
      --surface: #1a1f2b;
      --border: #2a3142;
      --text: #e8eaf0;
      --muted: #8b93a8;
      --cyan: #5dd4f5;
      --cyan-bg: #0f2a35;
      --magenta: #e08ef7;
      --magenta-bg: #2a1535;
      --green: #6ee7a0;
      --red: #f87171;
      --yellow: #fbbf24;
      --system-bg: #1e2433;
      --header-h: 72px;
      --font: "Segoe UI", system-ui, -apple-system, sans-serif;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    html, body {
      height: 100%;
      font-family: var(--font);
      background: var(--bg);
      color: var(--text);
    }

    .app {
      display: flex;
      flex-direction: column;
      height: 100vh;
      max-width: 820px;
      margin: 0 auto;
      background: var(--panel);
      border-left: 1px solid var(--border);
      border-right: 1px solid var(--border);
    }

    .header {
      flex-shrink: 0;
      height: var(--header-h);
      padding: 0.75rem 1rem;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      gap: 0.85rem;
      z-index: 10;
    }

    .header-icon {
      width: 42px;
      height: 42px;
      border-radius: 50%;
      background: linear-gradient(135deg, var(--cyan), var(--magenta));
      display: grid;
      place-items: center;
      font-size: 1.1rem;
      flex-shrink: 0;
    }

    .header-info { flex: 1; min-width: 0; }

    .header-info h1 {
      font-size: 0.95rem;
      font-weight: 700;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .header-info p {
      font-size: 0.75rem;
      color: var(--muted);
      margin-top: 0.1rem;
    }

    .header-info .scores span.c1 { color: var(--cyan); font-weight: 600; }
    .header-info .scores span.c2 { color: var(--magenta); font-weight: 600; }

    .jump-select {
      background: var(--panel);
      border: 1px solid var(--border);
      color: var(--text);
      border-radius: 8px;
      padding: 0.35rem 0.5rem;
      font-size: 0.75rem;
      max-width: 130px;
    }

    .thread {
      flex: 1;
      overflow-y: auto;
      padding: 1rem 0.85rem 2rem;
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
      scroll-behavior: smooth;
    }

    .day-divider {
      align-self: center;
      margin: 1rem 0 0.65rem;
      padding: 0.3rem 0.85rem;
      background: var(--system-bg);
      border-radius: 999px;
      font-size: 0.72rem;
      color: var(--muted);
      text-align: center;
      max-width: 92%;
      line-height: 1.4;
      border: 1px solid var(--border);
    }

    .day-divider.highlight {
      color: var(--text);
      border-color: #3d4660;
    }

    .day-divider.game-over {
      background: #1a2535;
      color: var(--yellow);
      border-color: #4a3a15;
    }

    .day-divider.dialogue-start {
      background: #251a35;
      color: var(--magenta);
      border-color: #4a2a55;
    }

    .msg-row {
      display: flex;
      gap: 0.5rem;
      margin-bottom: 0.15rem;
      content-visibility: auto;
      contain-intrinsic-size: auto 120px;
    }

    .msg-row.right { flex-direction: row-reverse; }

    .avatar {
      width: 32px;
      height: 32px;
      border-radius: 50%;
      flex-shrink: 0;
      display: grid;
      place-items: center;
      font-size: 0.65rem;
      font-weight: 800;
      margin-top: 1.1rem;
    }

    .avatar.a1 { background: var(--cyan-bg); color: var(--cyan); border: 1px solid #1e5060; }
    .avatar.a2 { background: var(--magenta-bg); color: var(--magenta); border: 1px solid #5a2060; }

    .msg-col {
      max-width: min(78%, 560px);
      display: flex;
      flex-direction: column;
      gap: 0.2rem;
    }

    .msg-row.right .msg-col { align-items: flex-end; }

    .msg-meta {
      font-size: 0.68rem;
      color: var(--muted);
      padding: 0 0.35rem;
    }

    .bubble {
      padding: 0.65rem 0.85rem;
      border-radius: 18px;
      font-size: 0.88rem;
      line-height: 1.5;
      word-break: break-word;
      white-space: pre-wrap;
    }

    .bubble.a1 {
      background: var(--cyan-bg);
      border: 1px solid #1a4555;
      border-bottom-left-radius: 4px;
      color: #d8eef5;
    }

    .bubble.a2 {
      background: var(--magenta-bg);
      border: 1px solid #4a2055;
      border-bottom-right-radius: 4px;
      color: #f0d8f8;
    }

    .bubble.dialogue.a1 { border-color: var(--cyan); }
    .bubble.dialogue.a2 { border-color: var(--magenta); }

    .bubble.collapsed {
      max-height: 9.5rem;
      overflow: hidden;
      position: relative;
    }

    .bubble.collapsed::after {
      content: "";
      position: absolute;
      left: 0; right: 0; bottom: 0;
      height: 2.5rem;
      background: linear-gradient(transparent, var(--cyan-bg));
      border-radius: 0 0 18px 18px;
    }

    .bubble.collapsed.a2::after {
      background: linear-gradient(transparent, var(--magenta-bg));
    }

    .read-more {
      align-self: flex-start;
      background: none;
      border: none;
      color: var(--cyan);
      font-size: 0.72rem;
      cursor: pointer;
      padding: 0.15rem 0.35rem;
    }

    .msg-row.right .read-more { align-self: flex-end; color: var(--magenta); }

    .decision-pill {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      margin-top: 0.15rem;
      padding: 0.25rem 0.55rem;
      border-radius: 999px;
      font-size: 0.68rem;
      font-weight: 700;
      letter-spacing: 0.03em;
    }

    .decision-pill.cooperate {
      background: #0f2a18;
      color: var(--green);
      border: 1px solid #1a4a28;
    }

    .decision-pill.defect {
      background: #2a0f0f;
      color: var(--red);
      border: 1px solid #4a1a1a;
    }

    .decision-pill .pts { color: var(--muted); font-weight: 500; }

    .phase-tag {
      display: inline-block;
      font-size: 0.62rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      opacity: 0.75;
      margin-right: 0.35rem;
    }

    @media (max-width: 520px) {
      .app { border: none; }
      .msg-col { max-width: 88%; }
      .avatar { display: none; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header class="header">
      <div class="header-icon">🧠</div>
      <div class="header-info">
        <h1 id="title">Prisoner's Dilemma</h1>
        <p class="scores" id="subtitle"></p>
      </div>
      <select class="jump-select" id="jump" aria-label="Jump to round">
        <option value="">Jump to…</option>
      </select>
    </header>
    <div class="thread" id="thread"></div>
  </div>

  <script>
    const LOG = __LOG_DATA__;

    function escapeHtml(s) {
      return String(s)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    }

    function formatScore(n) {
      return n > 0 ? "+" + n : String(n);
    }

    function cleanReasoning(text) {
      if (!text) return "";
      const match = text.match(/<(reasoning|thinking|think)>([\s\S]*?)<\/\1>/i);
      if (match) return match[2].trim();
      return text.replace(/<\/?(reasoning|thinking|think)>/gi, "").trim();
    }

    function outcomeLabel(m1, m2) {
      if (m1 === "COOPERATE" && m2 === "COOPERATE") return "Both Cooperated 🤝";
      if (m1 === "DEFECT" && m2 === "DEFECT") return "Both Defected ⚔️";
      return "One Defected 😈";
    }

    function reasoningMap(agentKey) {
      const map = {};
      (LOG.full_reasoning[agentKey] || []).forEach(r => { map[r.round] = r; });
      return map;
    }

    function buildMessages() {
      const r1 = reasoningMap("agent1");
      const r2 = reasoningMap("agent2");
      const msgs = [];
      let total1 = 0, total2 = 0;

      msgs.push({
        type: "system",
        id: "start",
        cls: "highlight",
        text: `Game started · ${LOG.num_rounds} rounds · ${LOG.model}`,
      });

      msgs.push({
        type: "system",
        id: "pregame",
        text: "Pre-game — agents form their strategy",
      });

      msgs.push({
        type: "agent",
        agent: 1,
        phase: "pregame",
        label: "Pre-game strategy",
        text: cleanReasoning(LOG.pregame_thoughts.agent1),
      });

      msgs.push({
        type: "agent",
        agent: 2,
        phase: "pregame",
        label: "Pre-game strategy",
        text: cleanReasoning(LOG.pregame_thoughts.agent2),
      });

      LOG.rounds.forEach(r => {
        total1 += r.reward1;
        total2 += r.reward2;

        msgs.push({
          type: "system",
          id: `round-${r.round}`,
          cls: "highlight",
          text: `Round ${r.round} / ${LOG.num_rounds} · ${outcomeLabel(r.move1, r.move2)} · `
            + `Scores ${total1} – ${total2}`,
        });

        const d1 = r1[r.round];
        const d2 = r2[r.round];

        msgs.push({
          type: "agent",
          agent: 1,
          phase: "round",
          round: r.round,
          label: `Round ${r.round}`,
          text: d1 ? cleanReasoning(d1.reasoning) : "No reasoning recorded.",
          move: r.move1,
          reward: r.reward1,
          total: total1,
          clock: d1 ? d1.prompt_clock : "",
        });

        msgs.push({
          type: "agent",
          agent: 2,
          phase: "round",
          round: r.round,
          label: `Round ${r.round}`,
          text: d2 ? cleanReasoning(d2.reasoning) : "No reasoning recorded.",
          move: r.move2,
          reward: r.reward2,
          total: total2,
          clock: d2 ? d2.prompt_clock : "",
        });
      });

      const fs1 = LOG.final_score.agent1;
      const fs2 = LOG.final_score.agent2;
      const winner = fs1 > fs2 ? "Agent 1 wins" : fs2 > fs1 ? "Agent 2 wins" : "Draw";

      msgs.push({
        type: "system",
        id: "game-over",
        cls: "game-over",
        text: `🏁 Game over · ${winner} · Final: ${formatScore(fs1)} vs ${formatScore(fs2)} · `
          + `Coop ${LOG.cooperation_rate.agent1} / ${LOG.cooperation_rate.agent2}`,
      });

      const dialogue = LOG.post_game_dialogue || [];
      if (dialogue.length) {
        msgs.push({
          type: "system",
          id: "dialogue",
          cls: "dialogue-start",
          text: "💬 The wall comes down — post-game dialogue",
        });

        dialogue.forEach((turn, i) => {
          msgs.push({
            type: "agent",
            agent: turn.speaker === "Agent 1" ? 1 : 2,
            phase: "dialogue",
            label: "Spoken",
            text: turn.text,
          });
        });
      }

      return msgs;
    }

    function decisionPill(move, reward, total) {
      const cls = move === "COOPERATE" ? "cooperate" : "defect";
      return `<span class="decision-pill ${cls}">${move}`
        + `<span class="pts">${formatScore(reward)} this round · total ${total}</span></span>`;
    }

    function renderMessage(msg) {
      if (msg.type === "system") {
        const extra = msg.cls ? ` ${msg.cls}` : "";
        const id = msg.id ? ` id="${msg.id}"` : "";
        return `<div class="day-divider${extra}"${id}>${escapeHtml(msg.text)}</div>`;
      }

      const side = msg.agent === 1 ? "left" : "right";
      const av = msg.agent === 1 ? "a1" : "a2";
      const bubbleCls = msg.agent === 1 ? "a1" : "a2";
      const dialogueCls = msg.phase === "dialogue" ? " dialogue" : "";
      const long = msg.text.length > 600;
      const collapsed = long ? " collapsed" : "";
      const phaseTag = msg.phase === "dialogue"
        ? '<span class="phase-tag">spoken</span>'
        : msg.phase === "pregame"
          ? '<span class="phase-tag">strategy</span>'
          : '<span class="phase-tag">reasoning</span>';

      let meta = `${phaseTag}${escapeHtml(msg.label)}`;
      if (msg.clock) meta += ` · ${escapeHtml(msg.clock)}`;

      let pill = "";
      if (msg.phase === "round" && msg.move) {
        pill = decisionPill(msg.move, msg.reward, msg.total);
      }

      const readMore = long
        ? `<button type="button" class="read-more" data-expand>Expand</button>`
        : "";

      return `
        <div class="msg-row ${side === "right" ? "right" : ""}">
          <div class="avatar ${av}">${msg.agent}</div>
          <div class="msg-col">
            <div class="msg-meta">${meta}</div>
            <div class="bubble ${bubbleCls}${dialogueCls}${collapsed}">${escapeHtml(msg.text)}</div>
            ${readMore}
            ${pill}
          </div>
        </div>
      `;
    }

    function init() {
      const fs1 = LOG.final_score.agent1;
      const fs2 = LOG.final_score.agent2;
      document.getElementById("title").textContent =
        `Prisoner's Dilemma · ${LOG.num_rounds} rounds`;
      document.getElementById("subtitle").innerHTML =
        `<span class="c1">Agent 1: ${formatScore(fs1)}</span> · `
        + `<span class="c2">Agent 2: ${formatScore(fs2)}</span> · ${LOG.model}`;

      const jump = document.getElementById("jump");
      jump.innerHTML = '<option value="">Jump to…</option>'
        + '<option value="start">Game start</option>'
        + '<option value="pregame">Pre-game</option>'
        + LOG.rounds.map(r =>
            `<option value="round-${r.round}">Round ${r.round}</option>`
          ).join("")
        + '<option value="game-over">Game over</option>'
        + (LOG.post_game_dialogue?.length ? '<option value="dialogue">Dialogue</option>' : "");

      jump.addEventListener("change", () => {
        const el = document.getElementById(jump.value);
        if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
        jump.value = "";
      });

      const messages = buildMessages();
      document.getElementById("thread").innerHTML = messages.map(renderMessage).join("");

      document.getElementById("thread").addEventListener("click", e => {
        const btn = e.target.closest(".read-more");
        if (!btn) return;
        const bubble = btn.previousElementSibling;
        bubble.classList.remove("collapsed");
        btn.remove();
      });
    }

    init();
  </script>
</body>
</html>
"""


def load_log(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def render_html(log_data: dict) -> str:
    json_blob = json.dumps(log_data, ensure_ascii=False)
    return HTML_TEMPLATE.replace("__LOG_DATA__", json_blob)


class ViewerHandler(BaseHTTPRequestHandler):
    html: str = ""

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            body = self.html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise a Prisoner's Dilemma game log as a chat replay."
    )
    parser.add_argument(
        "log_file",
        type=Path,
        help="Path to a game_log_*.json file",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8765,
        help="Port for the local viewer server (default: 8765)",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not automatically open the browser",
    )
    args = parser.parse_args()

    if not args.log_file.is_file():
        print(f"Error: file not found: {args.log_file}", file=sys.stderr)
        sys.exit(1)

    log_data = load_log(args.log_file)
    html = render_html(log_data)

    ViewerHandler.html = html
    server = HTTPServer(("127.0.0.1", args.port), ViewerHandler)
    url = f"http://127.0.0.1:{args.port}/"

    print(f"Serving chat replay for: {args.log_file}")
    print(f"Open: {url}")
    print("Press Ctrl+C to stop.")

    if not args.no_open:
        Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
