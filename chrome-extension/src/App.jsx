import { useState } from 'react';
import { PlayCircle, BarChart3, Loader2, AlertCircle, ThumbsUp, ThumbsDown, Minus } from 'lucide-react';

function App() {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const analyzeComments = async () => {
    try {
      setLoading(true);
      setError(null);
      setResults(null);
      
      // 1. Get current active tab
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (!tab.url.includes("youtube.com/watch")) {
        throw new Error("Please open a YouTube video page to analyze comments.");
      }

      // 2. Extract Video ID
      const urlParams = new URL(tab.url).searchParams;
      const videoId = urlParams.get('v');
      if (!videoId) throw new Error("Could not find a valid YouTube video ID.");

      // 3. Fetch Comments from YouTube Data API
      const apiKey = import.meta.env.VITE_YOUTUBE_API_KEY;
      const ytResponse = await fetch(`https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId=${videoId}&key=${apiKey}&maxResults=50`);
      const ytData = await ytResponse.json();
      
      if (ytData.error) throw new Error(ytData.error.message);
      if (!ytData.items || ytData.items.length === 0) throw new Error("No comments found on this video.");

      // Parse comments
      const comments = ytData.items.map(item => item.snippet.topLevelComment.snippet.textOriginal);

      // 4. Send to FastAPI Backend
      const backendUrl = import.meta.env.VITE_BACKEND_URL;
      const apiResponse = await fetch(`${backendUrl}/api/v1/predict/batch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comments })
      });
      
      const apiData = await apiResponse.json();
      if (!apiResponse.ok) throw new Error("Failed to analyze sentiment from backend.");

      // 5. Aggregate Results
      let positive = 0, negative = 0, neutral = 0;
      apiData.predictions.forEach(p => {
        if (p.sentiment === 'positive') positive++;
        else if (p.sentiment === 'negative') negative++;
        else neutral++;
      });
      
      const total = apiData.predictions.length;
      setResults({
        total,
        positive: Math.round((positive / total) * 100),
        negative: Math.round((negative / total) * 100),
        neutral: Math.round((neutral / total) * 100),
      });

    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center min-h-[500px] w-[400px] bg-slate-950 text-slate-50 relative overflow-hidden">
      
      {/* Background Decorators */}
      <div className="absolute top-[-50px] left-[-50px] w-48 h-48 bg-red-500 rounded-full blur-[100px] opacity-20"></div>
      <div className="absolute bottom-[-50px] right-[-50px] w-48 h-48 bg-blue-500 rounded-full blur-[100px] opacity-20"></div>

      {/* Header */}
      <div className="w-full flex items-center justify-between p-5 border-b border-white/10 bg-white/5 backdrop-blur-md z-10">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-red-500/20 rounded-xl">
            <PlayCircle className="w-6 h-6 text-red-500" />
          </div>
          <h1 className="text-lg font-bold tracking-tight bg-gradient-to-br from-white to-slate-400 bg-clip-text text-transparent">
            SentyTube
          </h1>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 w-full p-6 flex flex-col items-center justify-center gap-6 z-10">
        
        {!loading && !results && !error && (
          <div className="flex flex-col items-center text-center gap-4 animate-in fade-in zoom-in duration-500">
            <div className="p-4 bg-slate-800/50 rounded-full border border-white/5 shadow-xl mb-2">
              <BarChart3 className="w-10 h-10 text-blue-400" />
            </div>
            <h2 className="text-xl font-semibold text-white">Vibe Check!</h2>
            <p className="text-sm text-slate-400 leading-relaxed px-4">
              Click the button below to instantly scan the top comments of the current video and analyze the audience sentiment.
            </p>
          </div>
        )}

        {loading && (
          <div className="flex flex-col items-center gap-4 animate-in fade-in duration-300">
            <Loader2 className="w-12 h-12 text-blue-500 animate-spin" />
            <p className="text-sm font-medium text-slate-300 animate-pulse">Running Neural Networks...</p>
          </div>
        )}

        {error && (
          <div className="w-full p-4 bg-red-500/10 border border-red-500/20 rounded-2xl flex flex-col items-center text-center gap-2 animate-in slide-in-from-bottom-4 duration-300">
            <AlertCircle className="w-8 h-8 text-red-400" />
            <p className="text-sm text-red-200">{error}</p>
          </div>
        )}

        {results && (
          <div className="w-full flex flex-col gap-5 animate-in slide-in-from-bottom-6 duration-500">
            <div className="text-center">
              <p className="text-xs text-slate-400 uppercase tracking-wider font-semibold mb-1">Analysis Complete</p>
              <h2 className="text-3xl font-bold text-white">{results.total} <span className="text-lg text-slate-500 font-normal">Comments</span></h2>
            </div>

            <div className="flex flex-col gap-3">
              {/* Positive */}
              <div className="flex flex-col gap-1.5">
                <div className="flex justify-between items-center text-sm font-medium">
                  <div className="flex items-center gap-2 text-emerald-400"><ThumbsUp className="w-4 h-4" /> Positive</div>
                  <span className="text-emerald-400">{results.positive}%</span>
                </div>
                <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                  <div className="h-full bg-emerald-500 transition-all duration-1000 ease-out" style={{ width: `${results.positive}%` }}></div>
                </div>
              </div>

              {/* Neutral */}
              <div className="flex flex-col gap-1.5 mt-2">
                <div className="flex justify-between items-center text-sm font-medium">
                  <div className="flex items-center gap-2 text-slate-400"><Minus className="w-4 h-4" /> Neutral</div>
                  <span className="text-slate-400">{results.neutral}%</span>
                </div>
                <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                  <div className="h-full bg-slate-500 transition-all duration-1000 ease-out" style={{ width: `${results.neutral}%` }}></div>
                </div>
              </div>

              {/* Negative */}
              <div className="flex flex-col gap-1.5 mt-2">
                <div className="flex justify-between items-center text-sm font-medium">
                  <div className="flex items-center gap-2 text-rose-400"><ThumbsDown className="w-4 h-4" /> Negative</div>
                  <span className="text-rose-400">{results.negative}%</span>
                </div>
                <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                  <div className="h-full bg-rose-500 transition-all duration-1000 ease-out" style={{ width: `${results.negative}%` }}></div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Footer / Action */}
      <div className="w-full p-6 pt-0 z-10">
        <button 
          onClick={analyzeComments} 
          disabled={loading}
          className="w-full py-3.5 px-4 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white rounded-xl font-semibold shadow-lg shadow-blue-900/50 transition-all active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {loading ? 'Analyzing...' : (results ? 'Analyze Again' : 'Analyze Video')}
        </button>
      </div>

    </div>
  );
}

export default App;
