import { useState } from "react";

const API_BASE = "http://localhost:8000";

export default function App() {
  const [form, setForm] = useState({
    age: 35,
    monthly_spend: 49.99,
    tenure_months: 10,
    plan_type: "pro",
    region: "west"
  });
  const [result, setResult] = useState(null);

  const update = (key, value) => setForm((prev) => ({ ...prev, [key]: value }));

  async function onSubmit(e) {
    e.preventDefault();
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...form,
        age: Number(form.age),
        monthly_spend: Number(form.monthly_spend),
        tenure_months: Number(form.tenure_months)
      })
    });
    const data = await res.json();
    setResult(data);
  }

  return (
    <div style={{ fontFamily: "sans-serif", maxWidth: 540, margin: "30px auto" }}>
      <h2>Customer Churn Risk Checker</h2>
      <p>Demo UI calling a scikit-learn model API.</p>

      <form onSubmit={onSubmit} style={{ display: "grid", gap: 10 }}>
        <input type="number" value={form.age} onChange={(e) => update("age", e.target.value)} placeholder="Age" />
        <input type="number" step="0.01" value={form.monthly_spend} onChange={(e) => update("monthly_spend", e.target.value)} placeholder="Monthly Spend" />
        <input type="number" value={form.tenure_months} onChange={(e) => update("tenure_months", e.target.value)} placeholder="Tenure (months)" />

        <select value={form.plan_type} onChange={(e) => update("plan_type", e.target.value)}>
          <option value="basic">basic</option>
          <option value="pro">pro</option>
          <option value="enterprise">enterprise</option>
        </select>

        <select value={form.region} onChange={(e) => update("region", e.target.value)}>
          <option value="north">north</option>
          <option value="south">south</option>
          <option value="east">east</option>
          <option value="west">west</option>
        </select>

        <button type="submit">Predict</button>
      </form>

      {result && (
        <div style={{ marginTop: 20, padding: 12, border: "1px solid #ddd" }}>
          <div><strong>Prediction:</strong> {result.prediction === 1 ? "Likely churn" : "Likely retain"}</div>
          <div><strong>Churn probability:</strong> {result.churn_probability}</div>
        </div>
      )}
    </div>
  );
}