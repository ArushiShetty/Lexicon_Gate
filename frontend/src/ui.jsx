/**
 * Shared Lexicon Gate UI primitives — match styles.css tokens, no new deps.
 */
import { useEffect, useId, useRef } from "react";

export function Breadcrumb({ items = [] }) {
  if (!items.length) return null;
  return (
    <nav className="breadcrumb" aria-label="Breadcrumb">
      <ol className="breadcrumb-list">
        {items.map((item, index) => {
          const last = index === items.length - 1;
          return (
            <li key={`${item.label}-${index}`} className="breadcrumb-item">
              {index > 0 ? <span className="breadcrumb-sep" aria-hidden="true">/</span> : null}
              {last || !item.onClick ? (
                <span className="breadcrumb-current" aria-current={last ? "page" : undefined}>
                  {item.label}
                </span>
              ) : (
                <button type="button" className="breadcrumb-link" onClick={item.onClick}>
                  {item.label}
                </button>
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

export function EmptyState({ title, children, action = null }) {
  return (
    <div className="empty-panel" role="status">
      {title ? <h3 className="empty-panel-title">{title}</h3> : null}
      {typeof children === "string" ? <p className="muted">{children}</p> : children}
      {action}
    </div>
  );
}

export function StatusBanner({ tone = "info", children, role }) {
  if (!children) return null;
  const r = role || (tone === "error" ? "alert" : "status");
  return (
    <p className={`status-banner status-banner-${tone}`} role={r}>
      {children}
    </p>
  );
}

const ROLE_HINTS = {
  owner: "Full control — members, sources, manuscript, delete",
  editor: "Can edit sources, evidence, notes, and manuscript sections",
  reviewer: "Can review, comment, and update checklist items",
  viewer: "Read-only — Ask/Compare over linked sources",
};

export function RoleBadge({ role }) {
  if (!role) return null;
  return (
    <span className={`status-pill role-badge role-${role}`} title={ROLE_HINTS[role] || ""}>
      {role}
    </span>
  );
}

export function PermissionHint({ role }) {
  const hint = ROLE_HINTS[role];
  if (!hint) return null;
  return <p className="permission-hint muted">{hint}</p>;
}

export function SaveStatus({ dirty, saving, savedLabel = "", error = "" }) {
  let text = "All changes saved";
  let tone = "saved";
  if (error) {
    text = error;
    tone = "error";
  } else if (saving) {
    text = "Saving…";
    tone = "saving";
  } else if (dirty) {
    text = "Unsaved changes";
    tone = "dirty";
  } else if (savedLabel) {
    text = savedLabel;
    tone = "saved";
  }
  return (
    <span className={`save-status save-status-${tone}`} role="status" aria-live="polite">
      {text}
    </span>
  );
}

export function AiAdvisory({ children }) {
  return (
    <aside className="ai-advisory" role="note" aria-label="AI advisory">
      <strong className="ai-advisory-label">AI advisory</strong>
      <div className="ai-advisory-body">{children}</div>
    </aside>
  );
}

export function ConfirmDialog({
  open,
  title,
  message,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  danger = false,
  busy = false,
  onConfirm,
  onCancel,
}) {
  const titleId = useId();
  const descId = useId();
  const confirmRef = useRef(null);

  useEffect(() => {
    if (!open) return undefined;
    const prev = document.activeElement;
    confirmRef.current?.focus();
    function onKey(e) {
      if (e.key === "Escape" && !busy) onCancel?.();
    }
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
      if (prev && typeof prev.focus === "function") prev.focus();
    };
  }, [open, busy, onCancel]);

  if (!open) return null;

  return (
    <div className="confirm-backdrop" role="presentation" onClick={() => !busy && onCancel?.()}>
      <div
        className="confirm-dialog"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={descId}
        onClick={(e) => e.stopPropagation()}
      >
        <h3 id={titleId}>{title}</h3>
        <p id={descId} className="muted">
          {message}
        </p>
        <div className="row-actions confirm-actions">
          <button type="button" className="ghost" disabled={busy} onClick={onCancel}>
            {cancelLabel}
          </button>
          <button
            ref={confirmRef}
            type="button"
            className={danger ? "danger" : ""}
            disabled={busy}
            onClick={onConfirm}
          >
            {busy ? "Working…" : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
