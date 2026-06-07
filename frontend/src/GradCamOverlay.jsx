export default function GradCamOverlay({ imageUrl, gradcamImage }) {
  if (!imageUrl && !gradcamImage) return null

  return (
    <div className="card">
      <div className="label">Image analysis</div>
      <p style={{ fontSize: 12, color: "var(--muted)", marginBottom: 14 }}>
        Grad-CAM heatmap — regions that influenced the prediction
      </p>

      {gradcamImage ? (
        <img
          src={gradcamImage}
          alt="Grad-CAM heatmap"
          style={{
            width:        "100%",
            borderRadius: 8,
            border:       "0.5px solid var(--border2)"
          }}
        />
      ) : imageUrl ? (
        <img
          src={imageUrl}
          alt="Article image"
          style={{
            width:        "100%",
            borderRadius: 8,
            border:       "0.5px solid var(--border2)"
          }}
        />
      ) : (
        <div style={{
          height:        120,
          background:    "var(--bg3)",
          borderRadius:  8,
          display:       "flex",
          alignItems:    "center",
          justifyContent:"center",
          fontSize:      13,
          color:         "var(--muted)"
        }}>
          No image available
        </div>
      )}
    </div>
  )
}