"use client";

import { useState } from "react";

export default function Home() {
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setResponse("");

    try {
      const res = await fetch("http://localhost:8000/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question }),
      });

      const data = await res.json();
      setResponse(data.answer);
    } catch (error) {
      console.error("Error fetching chatbot response:", error);
      setResponse("⚠️ Failed to get response from chatbot.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 to-white p-6">
      <div className="max-w-3xl mx-auto bg-white shadow-xl rounded-2xl p-8 mt-12 border border-gray-200">
        <h1 className="text-3xl font-bold text-center text-blue-700 mb-6">
          🤖 Chat with your PDF
        </h1>
        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <input
            type="text"
            placeholder="Ask me anything about your PDF..."
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            className="px-4 py-3 border rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-300 text-gray-800"
            required
          />
          <button
            type="submit"
            className={`w-full py-3 font-semibold rounded-lg transition duration-200 text-white ${
              loading ? "bg-blue-400" : "bg-blue-600 hover:bg-blue-700"
            }`}
            disabled={loading}
          >
            {loading ? "Thinking..." : "Ask Question"}
          </button>
        </form>

        {response && (
          <div className="mt-8 bg-blue-50 border border-blue-200 p-4 rounded-lg shadow-inner">
            <h2 className="text-lg font-semibold text-blue-800 mb-2">
              📬 Chatbot Response:
            </h2>
            <p className="text-gray-700 whitespace-pre-line">{response}</p>
          </div>
        )}
      </div>
    </main>
  );
}