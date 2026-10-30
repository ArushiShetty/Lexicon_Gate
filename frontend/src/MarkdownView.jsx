import React, { useMemo } from "react";

/**
 * Safely parse inline Markdown constructs into React elements:
 * - Citations: [1], [2], [14] -> interactive citation button
 * - Bold: **text** or __text__ -> <strong>
 * - Italic: *text* or _text_ -> <em>
 * - Inline code: `code` -> <code>
 * - Links: [label](https://...) -> <a>
 */
export function parseInline(text, onCite) {
  if (!text) return "";
  const str = String(text);

  // Regex to match:
  // 1. Citations: \[(?<cite>\d+)\]
  // 2. Links: \[(?<label>[^\]]+)\]\((?<url>https?:\/\/[^\s)]+)\)
  // 3. Bold: \*\*(?<bold>[^*]+)\*\*|__(?<bold2>[^_]+)__
  // 4. Inline code: `(?<code>[^`]+)`
  // 5. Italic: \*(?<italic>[^*]+)\*|_(?<italic2>[^_]+)_
  const tokenRegex = /(\[(\d+)\])|(\[([^\]]+)\]\((https?:\/\/[^\s)]+)\))|(\*\*([^*]+)\*\*|__([^_]+)__)|(`([^`]+)`)|(\*([^*]+)\*|_([^_]+)_)/g;

  const elements = [];
  let lastIndex = 0;
  let match;
  let keyIdx = 0;

  while ((match = tokenRegex.exec(str)) !== null) {
    // Plain text before token
    if (match.index > lastIndex) {
      elements.push(str.substring(lastIndex, match.index));
    }

    if (match[1]) {
      // Citation: [1]
      const citeId = Number(match[2]);
      elements.push(
        <button
          key={`cite-${keyIdx++}`}
          type="button"
          className="cite-link"
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            onCite?.(citeId);
          }}
          title={`View source passage [${citeId}]`}
          aria-label={`Source citation ${citeId}`}
        >
          [{citeId}]
        </button>
      );
    } else if (match[3]) {
      // Link: [label](url)
      const label = match[4];
      const url = match[5];
      elements.push(
        <a
          key={`link-${keyIdx++}`}
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          className="markdown-link"
        >
          {label}
        </a>
      );
    } else if (match[6]) {
      // Bold: **text** or __text__
      const boldText = match[7] || match[8];
      elements.push(
        <strong key={`bold-${keyIdx++}`}>
          {parseInline(boldText, onCite)}
        </strong>
      );
    } else if (match[9]) {
      // Inline code: `code`
      const codeText = match[10];
      elements.push(
        <code key={`code-${keyIdx++}`} className="markdown-inline-code">
          {codeText}
        </code>
      );
    } else if (match[11]) {
      // Italic: *text* or _text_
      const italicText = match[12] || match[13];
      elements.push(
        <em key={`italic-${keyIdx++}`}>
          {parseInline(italicText, onCite)}
        </em>
      );
    }

    lastIndex = tokenRegex.lastIndex;
  }

  if (lastIndex < str.length) {
    elements.push(str.substring(lastIndex));
  }

  return elements.length === 1 && typeof elements[0] === "string"
    ? elements[0]
    : elements;
}

/**
 * Parses full markdown string into structured React blocks:
 * Headings, lists, tables, code blocks, blockquotes, and paragraphs.
 */
export function MarkdownView({ content, onCite, className = "markdown-body" }) {
  const blocks = useMemo(() => {
    if (!content) return [];
    const lines = String(content).replace(/\r\n/g, "\n").split("\n");
    const result = [];
    let i = 0;
    let blockId = 0;

    while (i < lines.length) {
      const line = lines[i];

      // Empty line
      if (!line.trim()) {
        i++;
        continue;
      }

      // Code blocks: ```
      if (/^```/.test(line)) {
        const langMatch = line.match(/^```(\w+)?/);
        const lang = langMatch ? langMatch[1] : "";
        const codeLines = [];
        i++;
        while (i < lines.length && !/^```/.test(lines[i])) {
          codeLines.push(lines[i]);
          i++;
        }
        if (i < lines.length) i++; // skip closing ```
        result.push(
          <pre key={`code-${blockId++}`} className={`markdown-pre ${lang ? `lang-${lang}` : ""}`}>
            <code>{codeLines.join("\n")}</code>
          </pre>
        );
        continue;
      }

      // Horizontal rule: --- or ***
      if (/^(?:---+|\*\*\*+)$/.test(line.trim())) {
        result.push(<hr key={`hr-${blockId++}`} className="markdown-hr" />);
        i++;
        continue;
      }

      // Headings: #, ##, ###, ####
      const headingMatch = line.match(/^(#{1,4})\s+(.+)$/);
      if (headingMatch) {
        const level = headingMatch[1].length;
        const text = headingMatch[2].trim();
        const Tag = `h${level}`;
        result.push(
          <Tag key={`h-${blockId++}`} className={`markdown-h${level}`}>
            {parseInline(text, onCite)}
          </Tag>
        );
        i++;
        continue;
      }

      // Blockquote / Alert callouts: > [!WARNING] or > text
      if (/^>\s*/.test(line)) {
        const quoteLines = [];
        while (i < lines.length && /^>\s*/.test(lines[i])) {
          quoteLines.push(lines[i].replace(/^>\s?/, ""));
          i++;
        }
        let alertType = null;
        if (/^\[!(WARNING|CAUTION|ALERT)\]/i.test(quoteLines[0] || "")) {
          alertType = "warning";
          quoteLines.shift();
        } else if (/^\[!(NOTE|INFO|TIP)\]/i.test(quoteLines[0] || "")) {
          alertType = "note";
          quoteLines.shift();
        }

        if (alertType) {
          const alertContent = quoteLines.join("\n");
          result.push(
            <div key={`alert-${blockId++}`} className={`markdown-alert markdown-alert-${alertType}`}>
              <div className="markdown-alert-icon">{alertType === "warning" ? "⚠️" : "ℹ️"}</div>
              <div className="markdown-alert-body">
                {alertContent.split("\n\n").map((para, pIdx) => (
                  <p key={pIdx}>{parseInline(para.replace(/\n/g, " "), onCite)}</p>
                ))}
              </div>
            </div>
          );
        } else {
          result.push(
            <blockquote key={`quote-${blockId++}`} className="markdown-quote">
              <p>{parseInline(quoteLines.join(" "), onCite)}</p>
            </blockquote>
          );
        }
        continue;
      }

      // Markdown Table: | Col 1 | Col 2 |
      if (line.trim().startsWith("|") && line.trim().endsWith("|") && i + 1 < lines.length && /^\|\s*[-:]+[-| :]*\|$/.test(lines[i + 1].trim())) {
        const headerCells = line.split("|").slice(1, -1).map((c) => c.trim());
        i += 2; // skip header and delimiter row
        const rows = [];
        while (i < lines.length && lines[i].trim().startsWith("|") && lines[i].trim().endsWith("|")) {
          const cells = lines[i].split("|").slice(1, -1).map((c) => c.trim());
          rows.push(cells);
          i++;
        }
        result.push(
          <div key={`table-${blockId++}`} className="markdown-table-wrapper">
            <table className="markdown-table">
              <thead>
                <tr>
                  {headerCells.map((th, thIdx) => (
                    <th key={`th-${thIdx}`}>{parseInline(th, onCite)}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row, rIdx) => (
                  <tr key={`tr-${rIdx}`}>
                    {row.map((td, cIdx) => (
                      <td key={`td-${rIdx}-${cIdx}`}>{parseInline(td, onCite)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
        continue;
      }

      // Unordered list: - or *
      if (/^\s*[-*]\s+/.test(line)) {
        const items = [];
        while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
          const itemText = lines[i].replace(/^\s*[-*]\s+/, "");
          items.push(itemText);
          i++;
        }
        result.push(
          <ul key={`ul-${blockId++}`} className="markdown-ul">
            {items.map((item, itemIdx) => (
              <li key={`li-${itemIdx}`}>{parseInline(item, onCite)}</li>
            ))}
          </ul>
        );
        continue;
      }

      // Ordered list: 1. 2.
      if (/^\s*\d+\.\s+/.test(line)) {
        const items = [];
        while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
          const itemText = lines[i].replace(/^\s*\d+\.\s+/, "");
          items.push(itemText);
          i++;
        }
        result.push(
          <ol key={`ol-${blockId++}`} className="markdown-ol">
            {items.map((item, itemIdx) => (
              <li key={`oli-${itemIdx}`}>{parseInline(item, onCite)}</li>
            ))}
          </ol>
        );
        continue;
      }

      // Paragraph: accumulate lines until blank line or new block marker
      const paraLines = [];
      while (
        i < lines.length &&
        lines[i].trim() &&
        !/^#{1,4}\s+/.test(lines[i]) &&
        !/^\s*[-*]\s+/.test(lines[i]) &&
        !/^\s*\d+\.\s+/.test(lines[i]) &&
        !/^```/.test(lines[i]) &&
        !/^>\s+/.test(lines[i]) &&
        !/^(?:---+|\*\*\*+)$/.test(lines[i].trim()) &&
        !(lines[i].trim().startsWith("|") && lines[i].trim().endsWith("|"))
      ) {
        paraLines.push(lines[i]);
        i++;
      }

      if (paraLines.length > 0) {
        result.push(
          <p key={`p-${blockId++}`} className="markdown-p">
            {parseInline(paraLines.join(" "), onCite)}
          </p>
        );
      }
    }

    return result;
  }, [content, onCite]);

  return <div className={className}>{blocks}</div>;
}
