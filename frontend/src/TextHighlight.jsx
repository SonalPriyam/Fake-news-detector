export default function TextHighlight({ words, label }) {
  if (!words || words.length === 0) return null

  const isFake   = label === "FAKE"
  const highColor = isFake ? "248, 113, 113" : "52, 211, 153"

  return (
    <div className="card">
      <div className="label">Text attention</div>
      <p style={{ fontSize: 12, color: "var(--muted)", marginBottom: 14 }}>
        Highlighted words influenced the model most
      </p>

      <div style={{ lineHeight: 2.2, fontSize: 14 }}>
        {words.map((item, i) => {
          const opacity = 0.15 + item.score * 0.75
          return (
            <span
              key={i}
              title={`Attention score: ${item.score}`}
              style={{
                background:    `rgba(${highColor}, ${opacity})`,
                borderRadius:  3,
                padding:       "2px 4px",
                marginRight:   4,
                marginBottom:  4,
                display:       "inline-block",
                cursor:        "default",
                transition:    "background 0.2s",
                color:         item.score > 0.6
                  ? `rgb(${highColor})`
                  : "var(--text)"
              }}
            >
              {item.word}
            </span>
          )
        })}
      </div>

      <div style={{
        marginTop:   16,
        display:     "flex",
        alignItems:  "center",
        gap:         8,
        fontSize:    11,
        color:       "var(--muted)"
      }}>
        <div style={{
          width:        60,
          height:       6,
          borderRadius: 3,
          background:   `linear-gradient(to right, rgba(${highColor}, 0.15), rgba(${highColor}, 0.9))`
        }} />
        <span>low → high attention</span>
      </div>
    </div>
  )
}