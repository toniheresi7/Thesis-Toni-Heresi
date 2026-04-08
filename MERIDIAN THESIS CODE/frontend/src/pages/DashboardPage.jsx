import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api';
import CredBadge from '../components/CredBadge';

// ── Signal strength: rank-based 1-5 stars ─────────────────────────────────────
// Rank 1 → 5 stars, last rank → 1 star. Works regardless of adj_score scale.
function signalStrength(rank, total) {
  if (!total || total <= 1) return 5;
  const norm = 1 - (rank - 1) / (total - 1);
  return Math.max(1, Math.round(norm * 4) + 1);
}

function SignalStars({ rank, total }) {
  const stars = signalStrength(rank, total);
  const pct   = total > 1 ? (1 - (rank - 1) / (total - 1)) * 100 : 100;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 5, minWidth: 110 }}>
      {/* Stars */}
      <div style={{ display: 'flex', gap: 3 }}>
        {[1, 2, 3, 4, 5].map(n => (
          <span key={n} style={{
            fontSize: 13, lineHeight: 1,
            color: n <= stars ? '#f59e0b' : 'var(--border-b)',
            filter: n <= stars ? 'drop-shadow(0 0 4px rgba(245,158,11,0.5))' : 'none',
            transition: 'color 0.2s',
          }}>★</span>
        ))}
      </div>
      {/* Bar */}
      <div style={{ height: 3, background: 'var(--border)', borderRadius: 2, overflow: 'hidden', width: 80 }}>
        <div style={{
          height: '100%', borderRadius: 2,
          width: `${Math.max(4, pct)}%`,
          background: 'linear-gradient(90deg, #f59e0b, #fbbf24)',
          boxShadow: '0 0 6px rgba(245,158,11,0.4)',
        }} />
      </div>
    </div>
  );
}

// ── Sentiment: FinBERT → bar + word label ─────────────────────────────────────
function sentimentLabel(tone) {
  if (tone >  0.05) return { word: 'Bullish', color: 'var(--green)' };
  if (tone < -0.05) return { word: 'Bearish', color: 'var(--red)'   };
  return                    { word: 'Neutral', color: 'var(--text-3)' };
}

function SentimentCell({ tone }) {
  const { word, color } = sentimentLabel(tone);
  // Centre = 50%, positive expands right, negative expands left
  const pct = Math.abs(tone) / 0.35 * 50;   // cap at 35% tone → full half
  const clipped = Math.min(50, pct);
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 5, minWidth: 120 }}>
      <span style={{ fontSize: 12, fontWeight: 600, color }}>{word}</span>
      {/* Centred bar */}
      <div style={{ position: 'relative', height: 4, background: 'var(--border)', borderRadius: 2, width: 90 }}>
        {/* Centre tick */}
        <div style={{
          position: 'absolute', left: '50%', top: 0,
          width: 1, height: '100%', background: 'var(--border-b)',
        }} />
        {tone >= 0 ? (
          <div style={{
            position: 'absolute', left: '50%', top: 0,
            height: '100%', width: `${clipped}%`,
            background: 'var(--green)', borderRadius: '0 2px 2px 0',
          }} />
        ) : (
          <div style={{
            position: 'absolute', right: '50%', top: 0,
            height: '100%', width: `${clipped}%`,
            background: 'var(--red)', borderRadius: '2px 0 0 2px',
          }} />
        )}
      </div>
    </div>
  );
}

// ── Buzz: attn_shock σ → plain-English label ──────────────────────────────────
function buzzConfig(attn) {
  if (attn >= 2.0)  return { icon: '▲',  label: 'High',   color: '#fb923c', bg: 'rgba(251,146,60,0.1)',   border: 'rgba(251,146,60,0.3)'  };
  if (attn >= 0.5)  return { icon: '↑',  label: 'Rising', color: 'var(--blue)', bg: 'rgba(59,130,246,0.1)', border: 'rgba(59,130,246,0.3)' };
  if (attn >= -0.5) return { icon: '—',  label: 'Normal', color: 'var(--text-3)', bg: 'transparent',        border: 'var(--border)'        };
  return                   { icon: '↓',  label: 'Quiet',  color: 'var(--text-3)', bg: 'transparent',        border: 'var(--border)'        };
}

function BuzzPill({ attn }) {
  const { icon, label, color, bg, border } = buzzConfig(attn);
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 5,
      padding: '3px 10px', borderRadius: 20,
      fontSize: 12, fontWeight: 600, color,
      background: bg, border: `1px solid ${border}`,
      whiteSpace: 'nowrap',
    }}>
      <span style={{ fontSize: label === 'High' ? 12 : 13 }}>{icon}</span>
      {label}
    </span>
  );
}

// ── Market mood from shortlist aggregate ──────────────────────────────────────
function marketMood(shortlist) {
  const avgTone = shortlist.reduce((s, r) => s + r.tone_finbert, 0) / shortlist.length;
  if (avgTone >  0.04) return { label: 'Bullish',  color: 'var(--green)', icon: '▲' };
  if (avgTone < -0.04) return { label: 'Bearish',  color: 'var(--red)',   icon: '▼' };
  return                      { label: 'Mixed',    color: 'var(--amber)', icon: '◆' };
}

// ── Stat card ─────────────────────────────────────────────────────────────────
function StatCard({ label, value, sub, accent, mono = true }) {
  return (
    <div className="card" style={{ padding: '16px 20px', flex: 1, minWidth: 140 }}>
      <div style={{
        fontSize: 10, fontWeight: 700, letterSpacing: '0.12em',
        color: 'var(--text-3)', fontFamily: 'var(--mono)', marginBottom: 6,
      }}>
        {label}
      </div>
      <div style={{
        fontSize: 22, fontWeight: 700, lineHeight: 1,
        fontFamily: mono ? 'var(--mono)' : 'var(--sans)',
        color: accent || 'var(--text-1)',
      }}>
        {value}
      </div>
      {sub && <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 5 }}>{sub}</div>}
    </div>
  );
}

function TopSignalCard({ shortlist }) {
  const top = shortlist[0];
  const mood = marketMood(shortlist);
  const highCount = shortlist.filter(s => s.credibility_label === 'high').length;
  const stars = signalStrength(top.rank, shortlist.length);
  return (
    <div className="card" style={{
      padding: '16px 20px', flex: 2, minWidth: 240,
      background: 'linear-gradient(135deg, #0b1120 0%, #0d1a2e 100%)',
      border: '1px solid rgba(0,212,184,0.2)',
      boxShadow: '0 0 24px rgba(0,212,184,0.06)',
    }}>
      <div style={{
        fontSize: 10, fontWeight: 700, letterSpacing: '0.12em',
        color: 'var(--text-3)', fontFamily: 'var(--mono)', marginBottom: 8,
      }}>
        TODAY'S TOP SIGNAL
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <span style={{
          fontFamily: 'var(--mono)', fontSize: 28, fontWeight: 700,
          color: 'var(--accent)', letterSpacing: '0.04em',
        }}>
          {top.ticker}
        </span>
        <div style={{ display: 'flex', gap: 2 }}>
          {[1,2,3,4,5].map(n => (
            <span key={n} style={{ fontSize: 14, color: n <= stars ? '#f59e0b' : 'var(--border-b)' }}>★</span>
          ))}
        </div>
        <CredBadge label={top.credibility_label} score={top.credibility_score} size="sm" />
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 6, lineHeight: 1.5 }}>
        {top.explanation}
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginTop: 10 }}>
        <span style={{ fontSize: 11, color: 'var(--text-3)' }}>
          Market mood:&nbsp;
          <span style={{ color: mood.color, fontWeight: 600 }}>{mood.icon} {mood.label}</span>
        </span>
        <span style={{ fontSize: 11, color: 'var(--text-3)' }}>
          {highCount} high-confidence signal{highCount !== 1 ? 's' : ''} today
        </span>
      </div>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function DashboardPage() {
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter]   = useState('all');
  const navigate = useNavigate();

  useEffect(() => {
    api.shortlist()
      .then(d => { setData(d); setLoading(false); })
      .catch(console.error);
  }, []);

  if (loading) return <LoadingScreen />;
  if (!data)   return <ErrorScreen />;

  const { shortlist } = data;
  const highCount = shortlist.filter(s => s.credibility_label === 'high').length;
  const medCount  = shortlist.filter(s => s.credibility_label === 'medium').length;

  const filtered = filter === 'all'
    ? shortlist
    : shortlist.filter(s => s.credibility_label === filter);

  return (
    <div style={S.root}>
      {/* Page header */}
      <div style={S.header}>
        <div>
          <div style={S.pageTitle}>Today's Shortlist</div>
          <div style={S.pageDate}>
            Signal date: <span className="mono" style={{ color: 'var(--text-2)' }}>{data.date}</span>
            &nbsp;·&nbsp;
            <span style={{ color: 'var(--text-3)' }}>Social signal intelligence · 5 subreddits · 500 tickers screened</span>
          </div>
        </div>
        <button onClick={() => window.location.reload()} className="btn btn-ghost btn-sm">
          ↻ REFRESH
        </button>
      </div>

      {/* Stat cards */}
      <div style={S.statsRow}>
        <TopSignalCard shortlist={shortlist} />
        <StatCard
          label="SCREENED"
          value="500"
          sub="S&amp;P 500 tickers monitored"
          accent="var(--accent)"
        />
        <StatCard
          label="HIGH CONFIDENCE"
          value={highCount}
          sub={`${highCount} strong signals today`}
          accent="var(--green)"
        />
        <StatCard
          label="MODERATE"
          value={medCount}
          sub={`${medCount} developing signals`}
          accent="var(--amber)"
        />
        <StatCard
          label="TOTAL MENTIONS"
          value={shortlist.reduce((s, r) => s + r.n_posts, 0).toLocaleString()}
          sub="posts captured today"
          mono
        />
      </div>

      {/* Filter bar */}
      <div style={S.filterRow}>
        <span style={S.filterLabel}>CONFIDENCE:</span>
        {[
          { key: 'all',    label: 'All'      },
          { key: 'high',   label: '★ High'   },
          { key: 'medium', label: '◆ Moderate' },
          { key: 'low',    label: '▽ Low'    },
        ].map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setFilter(key)}
            style={{ ...S.filterBtn, ...(filter === key ? S.filterBtnActive : {}) }}
          >
            {label}
          </button>
        ))}
        <span style={{ marginLeft: 'auto', fontSize: 11, color: 'var(--text-3)' }}>
          {filtered.length} ticker{filtered.length !== 1 ? 's' : ''}
        </span>
      </div>

      {/* Shortlist table */}
      <div className="card" style={{ overflow: 'hidden' }}>
        <table style={S.table}>
          <thead>
            <tr style={S.thead}>
              <th style={S.th}>RANK</th>
              <th style={S.th}>TICKER</th>
              <th style={S.th}>CONFIDENCE</th>
              <th style={S.th}>SIGNAL</th>
              <th style={S.th}>SENTIMENT</th>
              <th style={S.th}>BUZZ</th>
              <th style={S.th}>MENTIONS</th>
              <th style={S.th}>WHY IT'S RANKED</th>
              <th style={S.th} />
            </tr>
          </thead>
          <tbody>
            {filtered.map((row, idx) => (
              <tr
                key={row.ticker}
                style={{ ...S.tr, ...(idx % 2 === 0 ? S.trEven : {}) }}
                onMouseEnter={e => e.currentTarget.style.background = 'var(--bg-hover)'}
                onMouseLeave={e => e.currentTarget.style.background = idx % 2 === 0 ? '#0c1424' : ''}
              >
                {/* Rank */}
                <td style={S.tdRank}>
                  <span style={{ ...S.rankNum }}>{row.rank}</span>
                </td>

                {/* Ticker */}
                <td style={S.td}>
                  <span style={S.ticker}>{row.ticker}</span>
                </td>

                {/* Confidence (credibility) */}
                <td style={S.td}>
                  <CredBadge label={row.credibility_label} score={row.credibility_score} />
                </td>

                {/* Signal strength — stars + bar */}
                <td style={S.td}>
                  <SignalStars rank={row.rank} total={shortlist.length} />
                </td>

                {/* Sentiment — word + centred bar */}
                <td style={S.td}>
                  <SentimentCell tone={row.tone_finbert} />
                </td>

                {/* Buzz */}
                <td style={S.td}>
                  <BuzzPill attn={row.attn_shock} />
                </td>

                {/* Mentions */}
                <td style={S.td}>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                    <span style={{ fontFamily: 'var(--mono)', fontSize: 15, fontWeight: 700, color: 'var(--text-1)' }}>
                      {row.n_posts}
                    </span>
                    <span style={{ fontSize: 10, color: 'var(--text-3)' }}>posts</span>
                  </div>
                </td>

                {/* Explanation */}
                <td style={{ ...S.td, maxWidth: 280 }}>
                  <span style={S.explanation}>{row.explanation}</span>
                </td>

                {/* Detail button */}
                <td style={S.td}>
                  <button
                    className="btn btn-ghost btn-sm"
                    onClick={() => navigate(`/stock/${row.ticker}`)}
                  >
                    Detail →
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Disclaimer */}
      <div style={S.disclaimer}>
        For monitoring purposes only · Not financial advice · Past signal performance does not guarantee future results
      </div>
    </div>
  );
}

// ── Utility screens ───────────────────────────────────────────────────────────
function LoadingScreen() {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '60vh' }}>
      <div style={{ textAlign: 'center' }}>
        <div style={{ fontFamily: 'var(--mono)', color: 'var(--accent)', fontSize: 13, letterSpacing: '0.1em' }}>
          LOADING SIGNAL DATA…
        </div>
        <div style={{ height: 2, width: 200, background: 'var(--border)', borderRadius: 1, overflow: 'hidden', margin: '12px auto 0' }}>
          <div style={{ height: '100%', background: 'var(--accent)', borderRadius: 1, width: '40%' }} />
        </div>
      </div>
    </div>
  );
}

function ErrorScreen() {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '60vh' }}>
      <div style={{ textAlign: 'center', color: 'var(--red)', fontFamily: 'var(--mono)', fontSize: 12 }}>
        SIGNAL FEED UNAVAILABLE — CHECK BACKEND SERVER
      </div>
    </div>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────
const S = {
  root: {
    padding: '24px 28px', maxWidth: 1400, margin: '0 auto',
    display: 'flex', flexDirection: 'column', gap: 20,
  },
  header: { display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' },
  pageTitle: { fontSize: 20, fontWeight: 700, color: 'var(--text-1)', letterSpacing: '-0.01em' },
  pageDate: { fontSize: 12, color: 'var(--text-3)', marginTop: 4 },
  statsRow: { display: 'flex', gap: 12, flexWrap: 'wrap' },
  filterRow: {
    display: 'flex', alignItems: 'center', gap: 8,
    padding: '10px 16px',
    background: 'var(--bg-card)', border: '1px solid var(--border)',
    borderRadius: 'var(--r-md)',
  },
  filterLabel: {
    fontSize: 10, fontWeight: 700, letterSpacing: '0.12em',
    color: 'var(--text-3)', fontFamily: 'var(--mono)',
  },
  filterBtn: {
    background: 'none', border: '1px solid transparent',
    color: 'var(--text-3)', fontSize: 12, fontWeight: 600,
    letterSpacing: '0.04em', padding: '4px 12px',
    borderRadius: 4, cursor: 'pointer',
    transition: 'all 0.15s',
  },
  filterBtnActive: {
    color: 'var(--accent)', border: '1px solid rgba(0,212,184,0.3)',
    background: 'rgba(0,212,184,0.08)',
  },
  table: { width: '100%', borderCollapse: 'collapse' },
  thead: { borderBottom: '1px solid var(--border)' },
  th: {
    padding: '10px 14px', textAlign: 'left',
    fontSize: 10, fontWeight: 700, letterSpacing: '0.1em',
    color: 'var(--text-3)', fontFamily: 'var(--mono)',
    whiteSpace: 'nowrap',
  },
  tr: { borderBottom: '1px solid var(--border)', transition: 'background 0.1s' },
  trEven: { background: '#0c1424' },
  td: { padding: '14px 14px', verticalAlign: 'middle' },
  tdRank: { padding: '14px 14px', verticalAlign: 'middle', width: 52 },
  rankNum: {
    fontFamily: 'var(--mono)', fontSize: 17, fontWeight: 700, color: 'var(--text-3)',
  },
  ticker: {
    fontFamily: 'var(--mono)', fontSize: 15, fontWeight: 700,
    color: 'var(--text-1)', letterSpacing: '0.04em',
  },
  explanation: {
    fontSize: 12, color: 'var(--text-2)', lineHeight: 1.5,
    display: '-webkit-box', WebkitLineClamp: 2,
    WebkitBoxOrient: 'vertical', overflow: 'hidden',
  },
  disclaimer: {
    textAlign: 'center', fontSize: 11,
    color: 'var(--text-3)', padding: '4px 0',
  },
};
