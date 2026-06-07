export default function ResultCard({ result }) {
  const isFake = result.label === "FAKE"

  const barColor     = isFake ? "var(--fake-color)" : "var(--real-color)"
  const fakeWidth    = `${result.fake_probability}%`
  const realWidth    = `${result.real_probability}%`

  return (
    <div
      className="card"
      style={{
        borderColor: isFake
          ? "rgba(248,113,113,0.3)"
          : "rgba(52,211,153,0.3)"
      }}
    >
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 16 }}>
        <div>
          <div style={{ fontSize: 13, color: "var(--muted)", marginBottom: 6 }}>
            {result.title?.slice(0, 80)}{result.title?.length > 80 ? "..." : ""}
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontSize: 32, fontWeight: 700, color: barColor }}>
              {result.label}
            </span>
            <span className={isFake ? "fake-badge" : "real-badge"}>
              {result.confidence}% confident
            </span>
          </div>
        </div>
      </div>

      <hr className="divider" />

      <div className="label">Probability breakdown</div>
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>

        <div>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 4 }}>
            <span style={{ color: "var(--fake-color)" }}>Fake</span>
            <span style={{ color: "var(--muted)" }}>{result.fake_probability}%</span>
          </div>
          <div style={{ background: "var(--bg3)", borderRadius: 4, height: 6 }}>
            <div style={{
              width:        fakeWidth,
              height:       "100%",
              background:   "var(--fake-color)",
              borderRadius: 4,
              transition:   "width 0.6s ease"
            }} />
          </div>
        </div>

        <div>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 4 }}>
            <span style={{ color: "var(--real-color)" }}>Real</span>
            <span style={{ color: "var(--muted)" }}>{result.real_probability}%</span>
          </div>
          <div style={{ background: "var(--bg3)", borderRadius: 4, height: 6 }}>
            <div style={{
              width:        realWidth,
              height:       "100%",
              background:   "var(--real-color)",
              borderRadius: 4,
              transition:   "width 0.6s ease"
            }} />
          </div>
        </div>

      </div>

      {result.text_preview && (
        <>
          <hr className="divider" />
          <div className="label">Article preview</div>
          <p style={{ fontSize: 13, color: "var(--muted)", lineHeight: 1.7 }}>
            {result.text_preview}
          </p>
        </>
      )}
    </div>
  )
}