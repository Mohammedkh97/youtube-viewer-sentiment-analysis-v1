import { useState } from 'react';
import { PlayCircle, Sparkles, Loader2, AlertTriangle, ThumbsUp, ThumbsDown, Minus, Activity } from 'lucide-react';

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
    <div className="flex flex-col items-center min-h-[550px] w-[400px] bg-[#09090b] text-zinc-50 relative overflow-hidden font-sans select-none">
      
      {/* Background Decorators - Glowing Orbs */}
      <div className="absolute top-[-20%] left-[-20%] w-64 h-64 bg-violet-600 rounded-full mix-blend-screen filter blur-[120px] opacity-40 animate-pulse"></div>
      <div className="absolute bottom-[-20%] right-[-20%] w-64 h-64 bg-fuchsia-600 rounded-full mix-blend-screen filter blur-[120px] opacity-30 animate-pulse" style={{ animationDelay: '2s' }}></div>
      <div className="absolute top-[40%] right-[-10%] w-48 h-48 bg-blue-600 rounded-full mix-blend-screen filter blur-[100px] opacity-20 animate-pulse" style={{ animationDelay: '1s' }}></div>

      {/* Header */}
      <div className="w-full flex items-center justify-between p-5 border-b border-white/5 bg-[#09090b]/60 backdrop-blur-xl z-10 shadow-lg">
        <div className="flex items-center gap-3">
          <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-tr from-violet-500/20 to-fuchsia-500/20 border border-white/10 shadow-[0_0_15px_rgba(139,92,246,0.3)]">
            <Activity className="w-5 h-5 text-violet-400" />
          </div>
          <h1 className="text-xl font-extrabold tracking-tight bg-gradient-to-r from-white via-zinc-200 to-zinc-400 bg-clip-text text-transparent drop-shadow-sm">
            SentyTube
          </h1>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 w-full p-6 flex flex-col items-center justify-center gap-6 z-10">
        
        {!loading && !results && !error && (
          <div className="flex flex-col items-center text-center gap-5 animate-in fade-in zoom-in duration-700">
            <div className="relative group cursor-default">
              <div className="absolute -inset-1 bg-gradient-to-r from-violet-600 to-fuchsia-600 rounded-full blur opacity-25 group-hover:opacity-50 transition duration-1000 group-hover:duration-200"></div>
              <div className="relative p-5 bg-[#09090b] ring-1 ring-white/10 rounded-full shadow-2xl">
                <Sparkles className="w-12 h-12 text-fuchsia-400 drop-shadow-[0_0_15px_rgba(232,121,249,0.5)]" />
              </div>
            </div>
            <div className="space-y-2">
              <h2 className="text-2xl font-bold tracking-tight text-white drop-shadow-md">AI Sentiment Radar</h2>
              <p className="text-[13px] text-zinc-400 leading-relaxed max-w-[280px] mx-auto font-medium">
                Tap below to run our neural network on the top comments. Discover exactly how the audience feels right now.
              </p>
            </div>
          </div>
        )}

        {loading && (
          <div className="flex flex-col items-center gap-6 animate-in fade-in duration-500">
            <div className="relative flex items-center justify-center">
              <div className="absolute w-20 h-20 border-4 border-violet-500/30 rounded-full animate-ping"></div>
              <Loader2 className="w-12 h-12 text-fuchsia-400 animate-spin drop-shadow-[0_0_10px_rgba(232,121,249,0.8)]" />
            </div>
            <div className="space-y-1 text-center">
              <p className="text-base font-bold text-transparent bg-clip-text bg-gradient-to-r from-violet-400 to-fuchsia-400 animate-pulse">
                Analyzing Audience...
              </p>
              <p className="text-xs text-zinc-500">Extracting emotional nuances</p>
            </div>
          </div>
        )}

        {error && (
          <div className="w-full p-5 bg-rose-500/5 border border-rose-500/20 rounded-2xl flex flex-col items-center text-center gap-3 shadow-[0_0_30px_rgba(244,63,94,0.1)] backdrop-blur-md animate-in slide-in-from-bottom-4 duration-300">
            <AlertTriangle className="w-10 h-10 text-rose-400 drop-shadow-[0_0_10px_rgba(244,63,94,0.5)]" />
            <p className="text-sm font-medium text-rose-200/90 leading-snug">{error}</p>
          </div>
        )}

        {results && (
          <div className="w-full flex flex-col gap-6 animate-in slide-in-from-bottom-8 duration-700">
            
            {/* Summary Banner */}
            <div className="relative overflow-hidden rounded-2xl p-5 border border-white/5 bg-white/[0.02] shadow-2xl flex items-center justify-between">
              <div className="absolute inset-0 bg-gradient-to-r from-violet-600/10 to-fuchsia-600/10 mix-blend-overlay"></div>
              <div className="z-10 flex flex-col">
                <p className="text-[11px] text-zinc-400 uppercase tracking-widest font-bold mb-1">Audience Vibe</p>
                <h2 className="text-3xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-white to-zinc-400 drop-shadow-sm">
                  {results.positive > results.negative ? 'Positive' : (results.negative > results.positive ? 'Negative' : 'Mixed')}
                </h2>
              </div>
              <div className="z-10 flex flex-col items-end">
                <span className="text-2xl font-bold text-white">{results.total}</span>
                <span className="text-[10px] text-zinc-500 uppercase font-semibold">Comments</span>
              </div>
            </div>

            {/* Progress Bars */}
            <div className="flex flex-col gap-5 px-1">
              
              {/* Positive */}
              <div className="flex flex-col gap-2">
                <div className="flex justify-between items-center text-sm font-bold">
                  <div className="flex items-center gap-2 text-emerald-400">
                    <ThumbsUp className="w-4 h-4" /> 
                    <span className="tracking-wide">Positive</span>
                  </div>
                  <span className="text-emerald-300 text-lg drop-shadow-[0_0_5px_rgba(52,211,153,0.5)]">{results.positive}%</span>
                </div>
                <div className="w-full h-3 bg-[#18181b] rounded-full overflow-hidden border border-white/5 shadow-inner">
                  <div className="h-full bg-gradient-to-r from-emerald-600 to-emerald-400 transition-all duration-1000 ease-out shadow-[0_0_10px_rgba(52,211,153,0.8)] rounded-full" style={{ width: `${results.positive}%` }}></div>
                </div>
              </div>

              {/* Neutral */}
              <div className="flex flex-col gap-2">
                <div className="flex justify-between items-center text-sm font-bold">
                  <div className="flex items-center gap-2 text-zinc-400">
                    <Minus className="w-4 h-4" /> 
                    <span className="tracking-wide">Neutral</span>
                  </div>
                  <span className="text-zinc-300 text-lg">{results.neutral}%</span>
                </div>
                <div className="w-full h-3 bg-[#18181b] rounded-full overflow-hidden border border-white/5 shadow-inner">
                  <div className="h-full bg-gradient-to-r from-zinc-600 to-zinc-400 transition-all duration-1000 ease-out rounded-full" style={{ width: `${results.neutral}%` }}></div>
                </div>
              </div>

              {/* Negative */}
              <div className="flex flex-col gap-2">
                <div className="flex justify-between items-center text-sm font-bold">
                  <div className="flex items-center gap-2 text-rose-400">
                    <ThumbsDown className="w-4 h-4" /> 
                    <span className="tracking-wide">Negative</span>
                  </div>
                  <span className="text-rose-300 text-lg drop-shadow-[0_0_5px_rgba(2fb,113,133,0.5)]">{results.negative}%</span>
                </div>
                <div className="w-full h-3 bg-[#18181b] rounded-full overflow-hidden border border-white/5 shadow-inner">
                  <div className="h-full bg-gradient-to-r from-rose-600 to-rose-400 transition-all duration-1000 ease-out shadow-[0_0_10px_rgba(2fb,113,133,0.8)] rounded-full" style={{ width: `${results.negative}%` }}></div>
                </div>
              </div>

            </div>
          </div>
        )}
      </div>

      {/* Footer / Action */}
      <div className="w-full p-6 pt-2 pb-8 z-10">
        <button 
          onClick={analyzeComments} 
          disabled={loading}
          className="relative group w-full disabled:opacity-60 disabled:cursor-not-allowed"
        >
          {/* Animated Glow Behind Button */}
          {!loading && <div className="absolute -inset-0.5 bg-gradient-to-r from-violet-600 via-fuchsia-600 to-blue-600 rounded-2xl blur opacity-60 group-hover:opacity-100 transition duration-500 group-hover:duration-200 animate-gradient-xy"></div>}
          
          <div className="relative w-full py-4 px-6 bg-[#09090b] border border-white/10 group-hover:border-white/20 rounded-2xl flex items-center justify-center gap-3 transition-all duration-300 group-active:scale-[0.98]">
            {loading ? (
              <span className="text-sm font-bold tracking-wide text-zinc-300">Processing Data...</span>
            ) : (
              <>
                <span className="text-[15px] font-bold tracking-wide text-transparent bg-clip-text bg-gradient-to-r from-white to-zinc-300">
                  {results ? 'Scan Another Video' : 'Analyze Comments'}
                </span>
                <PlayCircle className="w-5 h-5 text-fuchsia-400" />
              </>
            )}
          </div>
        </button>
      </div>

    </div>
  );
}

export default App;
