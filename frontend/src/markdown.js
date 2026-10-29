/** Escape HTML then apply a small safe Markdown subset. */

function escapeHtml(text) {
  return String(text || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function inlineMarkdown(escaped) {
  let out = escaped;
  out = out.replace(/`([^`]+)`/g, "<code>$1</code>");
  out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  out = out.replace(/\*([^*\n]+)\*/g, "<em>$1</em>");
  out = out.replace(/_([^_\n]+)_/g, "<em>$1</em>");
  out = out.replace(
    /\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>',
  );
  return out;
}

/**
 * Convert Markdown to trusted HTML (input is escaped first).
 * Supports: headings, paragraphs, lists, hr, fenced code, bold/italic/code/links.
 */
export function renderMarkdown(md) {
  const raw = String(md || "").replace(/\r\n/g, "\n");
  if (!raw.trim()) return "";

  const lines = raw.split("\n");
  const blocks = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (/^```/.test(line)) {
      const code = [];
      i += 1;
      while (i < lines.length && !/^```/.test(lines[i])) {
        code.push(lines[i]);
        i += 1;
      }
      i += 1;
      blocks.push(`<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`);
      continue;
    }

    if (/^---+$/.test(line.trim()) || /^\*\*\*+$/.test(line.trim())) {
      blocks.push("<hr />");
      i += 1;
      continue;
    }

    const heading = /^(#{1,3})\s+(.+)$/.exec(line);
    if (heading) {
      const level = heading[1].length;
      blocks.push(`<h${level}>${inlineMarkdown(escapeHtml(heading[2].trim()))}</h${level}>`);
      i += 1;
      continue;
    }

    if (/^\s*[-*]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        items.push(
          `<li>${inlineMarkdown(escapeHtml(lines[i].replace(/^\s*[-*]\s+/, "")))}</li>`,
        );
        i += 1;
      }
      blocks.push(`<ul>${items.join("")}</ul>`);
      continue;
    }

    if (/^\s*\d+\.\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        items.push(
          `<li>${inlineMarkdown(escapeHtml(lines[i].replace(/^\s*\d+\.\s+/, "")))}</li>`,
        );
        i += 1;
      }
      blocks.push(`<ol>${items.join("")}</ol>`);
      continue;
    }

    if (!line.trim()) {
      i += 1;
      continue;
    }

    const para = [];
    while (i < lines.length && lines[i].trim()) {
      if (/^#{1,3}\s+/.test(lines[i]) || /^```/.test(lines[i]) || /^\s*[-*]\s+/.test(lines[i])) {
        break;
      }
      para.push(lines[i]);
      i += 1;
    }
    blocks.push(`<p>${inlineMarkdown(escapeHtml(para.join(" ")))}</p>`);
  }

  return blocks.join("\n");
}

export function parseStoryHash(hash = typeof window !== "undefined" ? window.location.hash : "") {
  const match = /^#\/s\/([A-Za-z0-9][A-Za-z0-9\-_]{0,120})$/.exec(hash || "");
  return match ? match[1] : null;
}

export function storyShareUrl(slug) {
  const safe = String(slug || "").trim();
  if (!safe) return "";
  if (typeof window === "undefined") return `#/s/${safe}`;
  const path = window.location.pathname || "/";
  return `${window.location.origin}${path}#/s/${safe}`;
}

function escapeAttr(text) {
  return String(text || "")
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/</g, "&lt;");
}

/** Filename-safe slug from a title. */
export function storyFilename(title, ext = "html") {
  const base = String(title || "story")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 60) || "story";
  return `${base}.${ext}`;
}

/**
 * Self-contained HTML document for download / print — readable article styling.
 */
export function buildStoryHtmlDocument({
  title,
  body_md: bodyMd,
  author_name: authorName = "",
  published_at: publishedAt = "",
  status = "",
} = {}) {
  const safeTitle = escapeAttr(title || "Untitled");
  const bodyHtml = renderMarkdown(bodyMd || "") || "<p><em>No content.</em></p>";
  let meta = [];
  if (authorName) meta.push(escapeAttr(authorName));
  if (publishedAt) {
    try {
      meta.push(
        new Date(publishedAt).toLocaleDateString(undefined, {
          year: "numeric",
          month: "long",
          day: "numeric",
        }),
      );
    } catch {
      meta.push(escapeAttr(publishedAt));
    }
  } else if (status === "draft") {
    meta.push("Draft");
  }
  const metaLine = meta.length ? `<p class="meta">${meta.join(" · ")}</p>` : "";

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>${safeTitle}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet" />
  <style>
    :root {
      --ink: #14201a;
      --muted: #5a6b62;
      --paper: #f4f7f4;
      --accent: #1f6b4a;
      --rule: rgba(20, 32, 26, 0.12);
      --font: "Outfit", "Segoe UI", system-ui, sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: var(--font);
      color: var(--ink);
      background:
        radial-gradient(1200px 600px at 10% -10%, rgba(31, 107, 74, 0.12), transparent 55%),
        radial-gradient(900px 500px at 100% 0%, rgba(90, 140, 110, 0.1), transparent 50%),
        var(--paper);
      line-height: 1.7;
    }
    .sheet {
      max-width: 720px;
      margin: 0 auto;
      padding: 48px 28px 80px;
    }
    .brand {
      font-size: 12px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 28px;
    }
    h1 {
      font-family: var(--font);
      font-weight: 600;
      font-size: clamp(2rem, 5vw, 2.75rem);
      line-height: 1.15;
      margin: 0 0 12px;
      letter-spacing: -0.02em;
    }
    .meta {
      margin: 0 0 28px;
      color: var(--muted);
      font-size: 0.95rem;
    }
    .rule {
      height: 1px;
      background: var(--rule);
      margin: 0 0 28px;
      border: 0;
    }
    article { font-size: 1.08rem; font-family: var(--font); }
    article h1, article h2, article h3 {
      font-family: var(--font);
      line-height: 1.25;
      margin: 1.5em 0 0.5em;
    }
    article h1 { font-size: 1.6rem; }
    article h2 { font-size: 1.35rem; }
    article h3 { font-size: 1.15rem; }
    article p { margin: 0 0 1.1em; }
    article ul, article ol { margin: 0 0 1.1em; padding-left: 1.35em; }
    article a { color: var(--accent); }
    article code {
      font-size: 0.92em;
      padding: 0.12em 0.4em;
      border-radius: 6px;
      background: rgba(20, 32, 26, 0.06);
    }
    article pre {
      padding: 16px 18px;
      overflow: auto;
      border-radius: 12px;
      background: rgba(20, 32, 26, 0.06);
      border: 1px solid var(--rule);
    }
    article pre code { padding: 0; background: none; }
    article hr {
      border: 0;
      border-top: 1px solid var(--rule);
      margin: 2em 0;
    }
    footer {
      margin-top: 48px;
      padding-top: 16px;
      border-top: 1px solid var(--rule);
      font-size: 12px;
      color: var(--muted);
    }
    @media print {
      body { background: #fff; }
      .sheet { padding: 0; max-width: none; }
    }
  </style>
</head>
<body>
  <div class="sheet">
    <p class="brand">Lexicon Gate</p>
    <h1>${safeTitle}</h1>
    ${metaLine}
    <hr class="rule" />
    <article>
${bodyHtml}
    </article>
    <footer>Exported from Lexicon Gate</footer>
  </div>
</body>
</html>
`;
}

/** Trigger a browser download of the styled HTML story. */
export function downloadStoryHtml(story) {
  const html = buildStoryHtmlDocument(story);
  const blob = new Blob([html], { type: "text/html;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = storyFilename(story?.title, "html");
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
