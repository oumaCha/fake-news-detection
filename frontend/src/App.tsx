import { useState } from 'react'

function App() {
  const [article, setArticle] = useState("");
  const [prediction , setPrediction] = useState("");
  const [confidence , setConfidence] = useState<number | null>(null);


  async function handleAnalyze(){
    const response = await fetch("http://127.0.0.1:8000/api/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text: article,
      }),
    });

    const data = await response.json();
    setPrediction(data.prediction);
    setConfidence(data.confidence);
  
  }
  return (
   <div>
    <h1>Fake News Detector</h1>

    <p>Pastea news article below and the system will analyze whether
      it is likely real or fake
    </p>

    <textarea placeholder='Paste your news article here...' rows={12} cols={70} value={article} onChange={(event) => setArticle(event.target.value)} />

    <br/>

    <button onClick={handleAnalyze}>Analyze Article</button>

    {prediction && (
      <div>
        <h2>Result</h2>

        <p>Prediction:{prediction}</p>

        <p>
          Confidence:{" "}
          {confidence !== null 
          ? `${Math.round(confidence * 100)}%`:""}
        </p>
      </div>
    )}

   </div>
  );
}

export default App
