import { useState } from "react"

export default function UrlInput({ onAnalyze, loading }) {
  const [url, setUrl] = useState("")

  function handleSubmit() {
    if (!url.trim()) return
    onAnalyze(url.trim())
  }

  function handleKeyDown(e) {
    if (e.key === "Enter") handleSubmit()
  }

  const exampleUrls = [
    "https://www.bbc.com/news/world",
    "https://timesofindia.indiatimes.com",
  ]

  return (
    <div className="card">
      <div className="label">Article URL</div>
      <div style={{ display: "flex", gap: 10 }}>
        <input
          type="text"
          placeholder="Paste a news article URL..."
          value={url}
          onChange={e => setUrl(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
        />
        <button
          className="btn"
          onClick={handleSubmit}
          disabled={loading || !url.trim()}
          style={{ whiteSpace: "nowrap" }}
        >
          {loading ? "Analyzing..." : "Analyze"}
        </button>
      </div>

      <div style={{ marginTop: 12, display: "flex", gap: 8, flexWrap: "wrap" }}>
        <span style={{ fontSize: 12, color: "var(--muted)" }}>Try:</span>
        {exampleUrls.map(u => (
          <button
            key={u}
            className="btn-outline"
            onClick={() => setUrl(u)}
            disabled={loading}
          >
            {new URL(u).hostname}
          </button>
        ))}
      </div>
    </div>
  )
}