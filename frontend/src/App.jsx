import { useState } from "react"
import UrlInput      from "./UrlInput"
import ResultCard    from "./ResultCard"
import TextHighlight from "./TextHighlight"
import GradCamOverlay from "./GradCamOverlay"

export default function App() {
  const [loading, setLoading]   = useState(false)
  const [result,  setResult]    = useState(null)
  const [error,   setError]     = useState("")

  async function handleAnalyze(url) {
    setLoading(true)
    setError("")
    setResult(null)

    try {
      const res = await fetch("/predict", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ url })
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || "Something went wrong")
      }

      const data = await res.json()
      setResult(data)

    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <div className="app-header">
        <h1>Fake News Detector</h1>
        <p>Multimodal AI — BERT + ViT fusion model</p>
      </div>

      <UrlInput onAnalyze={handleAnalyze} loading={loading} />

      {error && (
        <div className="card" style={{ borderColor: "rgba(248,113,113,0.3)" }}>
          <p style={{ color: "#f87171", fontSize: 14 }}>{error}</p>
        </div>
      )}

      {loading && (
        <div className="card" style={{ textAlign: "center", padding: 40 }}>
          <div className="spinner" />
          <p style={{ color: "var(--muted)", marginTop: 16, fontSize: 14 }}>
            Scraping article, running model...
          </p>
        </div>
      )}

      {result && !loading && (
        <>
          <ResultCard result={result} />
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            <TextHighlight
              words={result.attention_weights}
              label={result.label}
            />
            <GradCamOverlay
              imageUrl={result.image_url}
              gradcamImage={result.gradcam_image}
            />
          </div>
        </>
      )}
    </div>
  )
}