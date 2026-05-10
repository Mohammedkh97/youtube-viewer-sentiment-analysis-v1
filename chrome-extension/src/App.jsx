import { useState } from 'react';
import { PlayCircle, Sparkles, Loader2, AlertTriangle, ThumbsUp, ThumbsDown, Minus, Activity, RefreshCw, MessageSquare } from 'lucide-react';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from 'recharts';

// ── helpers ──────────────────────────────────────────────────────────────────

const SENTIMENT_CFG = {
  positive: { color: '#10b981', glow: 'rgba(16,185,129,0.5)',  icon: ThumbsUp,   label: 'Positive', bg: 'bg-emerald-500/10', border: 'border-emerald-500/20', text: 'text-emerald-400' },
  neutral:  { color: '#71717a', glow: 'rgba(113,113,122,0.3)', icon: Minus,       label: 'Neutral',  bg: 'bg-zinc-500/10',   border: 'border-zinc-500/20',   text: 'text-zinc-400'   },
  negative: { color: '#f43f5e', glow: 'rgba(244,63,94,0.5)',   icon: ThumbsDown,  label: 'Negative', bg: 'bg-rose-500/10',   border: 'border-rose-500/20',   text: 'text-rose-400'   },
};

const CustomTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const { name, value } = payload[0];
    const cfg = SENTIMENT_CFG[name.toLowerCase()];
    return (
      <div className="px-3 py-1.5 rounded-lg text-xs font-bold" style={{ background: '#18181b', border: `1px solid ${cfg.color}40`, color: cfg.color }}>
        {cfg.label}: {value}%
      </div>
    );
  }
  return null;
};

// ── component ─────────────────────────────────────────────────────────────────

export default function App() {
  const [loading, setLoading]   = useState(false);
  const [results, setResults]   = useState(null);
  const [error, setError]       = useState(null);

  const analyzeComments = async () => {
    try {
      setLoading(true);
      setError(null);
      setResults(null);

      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (!tab.url.includes('youtube.com/watch'))
        throw new Error('Please open a YouTube video page to analyze comments.');

      const videoId = new URL(tab.url).searchParams.get('v');
      if (!videoId) throw new Error('Could not find a valid YouTube video ID.');

      const apiKey = import.meta.env.VITE_YOUTUBE_API_KEY;
      const ytRes  = await fetch(
        `https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId=${videoId}&key=${apiKey}&maxResults=50`
      );
      const ytData = await ytRes.json();
      if (ytData.error) throw new Error(ytData.error.message);
      if (!ytData.items?.length) throw new Error('No comments found on this video.');

      const rawComments = ytData.items.map(i => i.snippet.topLevelComment.snippet.textOriginal);

      const backendUrl = import.meta.env.VITE_BACKEND_URL;
      const apiRes = await fetch(`${backendUrl}/api/v1/predict/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ comments: rawComments }),
      });
      const apiData = await apiRes.json();
      if (!apiRes.ok) throw new Error('Failed to analyze sentiment from backend.');

      let pos = 0, neg = 0, neu = 0;
      apiData.predictions.forEach(p => {
        if (p.sentiment === 'positive') pos++;
        else if (p.sentiment === 'negative') neg++;
        else neu++;
      });

      const total = apiData.predictions.length;
      const topComments = apiData.predictions
        .sort((a, b) => b.confidence - a.confidence)
        .slice(0, 5)
        .map((p, i) => ({ text: rawComments[apiData.predictions.indexOf(p)] || rawComments[i], sentiment: p.sentiment, confidence: p.confidence }));

      setResults({
        total,
        positive: Math.round((pos / total) * 100),
        negative: Math.round((neg / total) * 100),
        neutral:  Math.round((neu / total) * 100),
        counts: { positive: pos, negative: neg, neutral: neu },
        topComments,
        overallVibe: pos > neg ? 'positive' : neg > pos ? 'negative' : 'neutral',
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const pieData = results ? [
    { name: 'Positive', value: results.positive },
    { name: 'Neutral',  value: results.neutral  },
    { name: 'Negative', value: results.negative },
  ] : [];

  return (
    <div className="flex flex-col min-h-[580px] w-[400px] bg-[#09090b] text-zinc-50 relative overflow-hidden font-sans select-none">

      {/* ambient orbs */}
      <div className="absolute top-[-15%] left-[-15%] w-56 h-56 bg-violet-600 rounded-full mix-blend-screen blur-[100px] opacity-30 animate-pulse" />
      <div className="absolute bottom-[-15%] right-[-15%] w-56 h-56 bg-fuchsia-600 rounded-full mix-blend-screen blur-[100px] opacity-25 animate-pulse" style={{ animationDelay: '2s' }} />
      <div className="absolute top-[45%] right-[-8%] w-40 h-40 bg-blue-600 rounded-full mix-blend-screen blur-[90px] opacity-15 animate-pulse" style={{ animationDelay: '1s' }} />

      {/* ── header ── */}
      <header className="flex items-center justify-between px-5 py-4 border-b border-white/5 bg-[#09090b]/70 backdrop-blur-xl z-10 shrink-0">
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-9 h-9 rounded-xl bg-gradient-to-tr from-violet-500/20 to-fuchsia-500/20 border border-white/10 shadow-[0_0_12px_rgba(139,92,246,0.35)]">
            <Activity className="w-4 h-4 text-violet-400" />
          </div>
          <span className="text-lg font-extrabold tracking-tight bg-gradient-to-r from-white via-zinc-200 to-zinc-400 bg-clip-text text-transparent">
            SentyTube
          </span>
        </div>
        {results && (
          <button onClick={analyzeComments} className="flex items-center gap-1.5 text-xs text-zinc-400 hover:text-white transition-colors px-3 py-1.5 rounded-lg hover:bg-white/5 border border-transparent hover:border-white/10">
            <RefreshCw className="w-3.5 h-3.5" /> Rescan
          </button>
        )}
      </header>

      {/* ── main content ── */}
      <main className="flex-1 overflow-y-auto z-10 scrollbar-thin">

        {/* idle state */}
        {!loading && !results && !error && (
          <div className="flex flex-col items-center justify-center text-center gap-5 p-8 h-full min-h-[440px]">
            <div className="relative group cursor-default">
              <div className="absolute -inset-1.5 bg-gradient-to-r from-violet-600 to-fuchsia-600 rounded-full blur opacity-30 group-hover:opacity-55 transition duration-700" />
              <div className="relative p-5 bg-[#09090b] ring-1 ring-white/10 rounded-full shadow-2xl">
                <Sparkles className="w-12 h-12 text-fuchsia-400 drop-shadow-[0_0_14px_rgba(232,121,249,0.55)]" />
              </div>
            </div>
            <div className="space-y-2">
              <h2 className="text-2xl font-bold text-white">AI Sentiment Radar</h2>
              <p className="text-[13px] text-zinc-400 leading-relaxed max-w-[270px] mx-auto">
                Run our neural network on the top 50 comments to instantly discover how the audience really feels.
              </p>
            </div>
          </div>
        )}

        {/* loading state */}
        {loading && (
          <div className="flex flex-col items-center justify-center gap-6 p-8 min-h-[440px]">
            <div className="relative flex items-center justify-center">
              <div className="absolute w-20 h-20 border-4 border-violet-500/25 rounded-full animate-ping" />
              <Loader2 className="w-11 h-11 text-fuchsia-400 animate-spin drop-shadow-[0_0_10px_rgba(232,121,249,0.8)]" />
            </div>
            <div className="space-y-1 text-center">
              <p className="text-base font-bold bg-clip-text text-transparent bg-gradient-to-r from-violet-400 to-fuchsia-400 animate-pulse">
                Analyzing Audience...
              </p>
              <p className="text-xs text-zinc-500">Extracting emotional nuances</p>
            </div>
          </div>
        )}

        {/* error state */}
        {error && (
          <div className="m-5 p-5 bg-rose-500/5 border border-rose-500/20 rounded-2xl flex flex-col items-center text-center gap-3 shadow-[0_0_25px_rgba(244,63,94,0.08)] backdrop-blur-md">
            <AlertTriangle className="w-10 h-10 text-rose-400 drop-shadow-[0_0_8px_rgba(244,63,94,0.5)]" />
            <p className="text-sm font-medium text-rose-200/90 leading-snug">{error}</p>
          </div>
        )}

        {/* results */}
        {results && (
          <div className="flex flex-col gap-5 p-5">

            {/* ── vibe banner ── */}
            {(() => {
              const cfg = SENTIMENT_CFG[results.overallVibe];
              const Icon = cfg.icon;
              return (
                <div className={`relative overflow-hidden rounded-2xl p-4 border ${cfg.border} ${cfg.bg} flex items-center justify-between gap-3`}>
                  <div className="absolute inset-0 opacity-5" style={{ background: `radial-gradient(circle at 20% 50%, ${cfg.color}, transparent 70%)` }} />
                  <div className="flex flex-col z-10">
                    <span className="text-[10px] text-zinc-400 uppercase tracking-widest font-bold mb-0.5">Overall Vibe</span>
                    <span className={`text-2xl font-extrabold ${cfg.text} capitalize drop-shadow-[0_0_8px_${cfg.glow}]`}>
                      {cfg.label}
                    </span>
                  </div>
                  <div className="z-10 flex flex-col items-end">
                    <span className="text-2xl font-bold text-white">{results.total}</span>
                    <span className="text-[10px] text-zinc-500 uppercase font-semibold">Comments</span>
                  </div>
                  <Icon className={`absolute right-14 top-1/2 -translate-y-1/2 w-16 h-16 opacity-5 ${cfg.text}`} />
                </div>
              );
            })()}

            {/* ── donut chart + stat cards ── */}
            <div className="flex items-center gap-4">

              {/* donut */}
              <div className="relative shrink-0 w-[130px] h-[130px]">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={pieData}
                      cx="50%" cy="50%"
                      innerRadius={38} outerRadius={58}
                      paddingAngle={3}
                      dataKey="value"
                      strokeWidth={0}
                      animationBegin={0}
                      animationDuration={900}
                    >
                      {pieData.map((entry) => (
                        <Cell key={entry.name} fill={SENTIMENT_CFG[entry.name.toLowerCase()].color} />
                      ))}
                    </Pie>
                    <Tooltip content={<CustomTooltip />} />
                  </PieChart>
                </ResponsiveContainer>
                {/* center label */}
                <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                  <span className="text-lg font-extrabold text-white leading-none">{results.positive}%</span>
                  <span className="text-[9px] text-zinc-500 uppercase tracking-wider mt-0.5">positive</span>
                </div>
              </div>

              {/* stat cards column */}
              <div className="flex-1 flex flex-col gap-2">
                {['positive', 'neutral', 'negative'].map(key => {
                  const cfg = SENTIMENT_CFG[key];
                  const Icon = cfg.icon;
                  return (
                    <div key={key} className={`flex items-center justify-between px-3 py-2 rounded-xl border ${cfg.border} ${cfg.bg}`}>
                      <div className={`flex items-center gap-2 text-xs font-bold ${cfg.text}`}>
                        <Icon className="w-3.5 h-3.5" />
                        {cfg.label}
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] text-zinc-500">{results.counts[key]}</span>
                        <span className={`text-sm font-extrabold ${cfg.text}`}>{results[key]}%</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* ── progress bars ── */}
            <div className="flex flex-col gap-3 bg-white/[0.02] rounded-2xl p-4 border border-white/5">
              <p className="text-[10px] uppercase tracking-widest font-bold text-zinc-500 mb-1">Distribution</p>
              {['positive', 'neutral', 'negative'].map(key => {
                const cfg = SENTIMENT_CFG[key];
                return (
                  <div key={key} className="flex flex-col gap-1">
                    <div className="flex justify-between text-[11px] font-semibold">
                      <span className={cfg.text}>{cfg.label}</span>
                      <span className="text-zinc-400">{results[key]}%</span>
                    </div>
                    <div className="w-full h-2 bg-[#18181b] rounded-full overflow-hidden border border-white/5">
                      <div
                        className="h-full rounded-full transition-all duration-1000 ease-out"
                        style={{ width: `${results[key]}%`, background: `linear-gradient(90deg, ${cfg.color}88, ${cfg.color})`, boxShadow: `0 0 8px ${cfg.glow}` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>

            {/* ── top comments ── */}
            <div className="flex flex-col gap-3 bg-white/[0.02] rounded-2xl p-4 border border-white/5">
              <div className="flex items-center gap-2 mb-1">
                <MessageSquare className="w-3.5 h-3.5 text-zinc-500" />
                <p className="text-[10px] uppercase tracking-widest font-bold text-zinc-500">Top Comments</p>
              </div>
              {results.topComments.map((c, i) => {
                const cfg = SENTIMENT_CFG[c.sentiment];
                return (
                  <div key={i} className="flex items-start gap-2.5">
                    <div className={`mt-0.5 shrink-0 w-2 h-2 rounded-full`} style={{ background: cfg.color, boxShadow: `0 0 5px ${cfg.glow}` }} />
                    <p className="text-[12px] text-zinc-300 leading-relaxed line-clamp-2 flex-1">{c.text}</p>
                    <span className={`shrink-0 text-[10px] font-bold px-2 py-0.5 rounded-full border ${cfg.border} ${cfg.bg} ${cfg.text}`}>
                      {Math.round(c.confidence * 100)}%
                    </span>
                  </div>
                );
              })}
            </div>

          </div>
        )}

      </main>

      {/* ── CTA button ── */}
      {!results && (
        <footer className="w-full p-5 z-10 shrink-0">
          <button
            onClick={analyzeComments}
            disabled={loading}
            className="relative group w-full disabled:opacity-60 disabled:cursor-not-allowed"
          >
            {!loading && (
              <div className="absolute -inset-0.5 bg-gradient-to-r from-violet-600 via-fuchsia-600 to-blue-600 rounded-2xl blur opacity-60 group-hover:opacity-90 transition duration-500 group-hover:duration-200" />
            )}
            <div className="relative w-full py-3.5 px-6 bg-[#09090b] border border-white/10 group-hover:border-white/20 rounded-2xl flex items-center justify-center gap-3 transition-all duration-300 group-active:scale-[0.98]">
              {loading ? (
                <span className="text-sm font-bold tracking-wide text-zinc-300">Processing Data...</span>
              ) : (
                <>
                  <span className="text-[15px] font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-zinc-300">
                    Analyze Comments
                  </span>
                  <PlayCircle className="w-5 h-5 text-fuchsia-400" />
                </>
              )}
            </div>
          </button>
        </footer>
      )}

    </div>
  );
}
