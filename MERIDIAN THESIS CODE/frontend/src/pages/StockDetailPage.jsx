import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  AreaChart, Area, BarChart, Bar, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine, Cell,
} from 'recharts';
import { api } from '../api';
import CredBadge from '../components/CredBadge';

/* ── Shared tooltip ──────────────────────────────────────────────────────────*/
function TT({ active, payload, label, unit = '' }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: '#0f1729', border: '1px solid var(--border-b)',
      borderRadius: 6, padding: '8px 12px', fontFamily: 'var(--mono)', fontSize: 12,
    }}>
      <div style={{ color: 'var(--text-3)', marginBottom: 4 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color || 'var(--accent)' }}>
          {p.name}: {typeof p.value === 'number' ? p.value.toFixed(4) : p.value}{unit}
        </div>
      ))}
    </div>
  );
}

/* ── Section wrapper ─────────────────────────────────────────────────────────*/
function ChartCard({ title, sub, children, style }) {
  return (
    <div className="card" style={{ padding: 24, ...style }}>
      <div style={{ marginBottom: 16 }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-1)' }}>{title}</div>
        {sub && <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 2 }}>{sub}</div>}
      </div>
      {children}
    </div>
  );
}

/* ── Header metric pill ──────────────────────────────────────────────────────*/
function MetricPill({ label, value, color }) {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', gap: 4,
      padding: '12px 20px',
      background: 'var(--bg-card-alt)',
      border: '1px solid var(--border)',
      borderRadius: 8, minWidth: 110,
    }}>
      <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.1em', color: 'var(--text-3)', fontFamily: 'var(--mono)' }}>
        {label}
      </div>
      <div style={{ fontSize: 20, fontWeight: 700, fontFamily: 'var(--mono)', color: color || 'var(--accent)' }}>
        {value}
      </div>
    </div>
  );
}

/* ── 5-day Sentiment Trend ───────────────────────────────────────────────────*/
function SentimentTrend({ data }) {
  const last      = data[data.length - 1]?.tone_finbert ?? 0;
  const lineColor = last >= 0 ? 'var(--green)' : 'var(--red)';
  const fillStop  = last >= 0 ? '#10b981' : '#ef4444';
  const gradId    = `sentGrad_${last >= 0 ? 'g' : 'r'}`;

  return (
    <ResponsiveContainer width="100%" height={160}>
      <AreaChart data={data} margin={{ top: 8, right: 8, left: -28, bottom: 0 }}>
        <defs>
          <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor={fillStop} stopOpacity={0.25} />
            <stop offset="95%" stopColor={fillStop} stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis dataKey="date" tick={{ fontSize: 10, fill: 'var(--text-3)' }}
          tickFormatter={d => d.slice(5)} />
        <YAxis tick={{ fontSize: 10, fill: 'var(--text-3)' }} domain={[-1, 1]} />
        <ReferenceLine y={0} stroke="var(--border-b)" strokeWidth={1.5} />
        <Tooltip content={<TT />} />
        <Area dataKey="tone_finbert" name="Sentiment"
          stroke={lineColor} strokeWidth={2}
          fill={`url(#${gradId})`}
          dot={{ r: 4, fill: lineColor, strokeWidth: 0 }}
          activeDot={{ r: 5 }}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

/* ── 5-day Post Volume ───────────────────────────────────────────────────────*/
function VolumeBars({ data }) {
  return (
    <ResponsiveContainer width="100%" height={160}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: -28, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis dataKey="date" tick={{ fontSize: 10, fill: 'var(--text-3)' }}
          tickFormatter={d => d.slice(5)} />
        <YAxis tick={{ fontSize: 10, fill: 'var(--text-3)' }} />
        <Tooltip content={<TT />} />
        <Bar dataKey="n_posts" name="Posts" radius={[3, 3, 0, 0]}>
          {data.map((entry, i) => (
            <Cell key={i} fill={
              entry.attn_shock > 2    ? 'var(--amber)'
              : entry.attn_shock > 0  ? 'var(--accent)'
              : 'var(--text-3)'
            } />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/* ── 30-day Attention Shock ──────────────────────────────────────────────────*/
function AttnShockChart({ data }) {
  return (
    <ResponsiveContainer width="100%" height={160}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: -28, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis dataKey="date" tick={{ fontSize: 10, fill: 'var(--text-3)' }}
          tickFormatter={d => d.slice(5)} interval={4} />
        <YAxis tick={{ fontSize: 10, fill: 'var(--text-3)' }} />
        <ReferenceLine y={0}  stroke="var(--border-b)" />
        <ReferenceLine y={2}  stroke="var(--amber)" strokeDasharray="4 4"
          label={{ value: '+2σ', fill: 'var(--amber)', fontSize: 10 }} />
        <ReferenceLine y={-2} stroke="var(--red)" strokeDasharray="4 4" />
        <Tooltip content={<TT unit="σ" />} />
        <Bar dataKey="attn_shock" name="Attn Shock" radius={[2, 2, 0, 0]}>
          {data.map((entry, i) => (
            <Cell key={i} fill={
              Math.abs(entry.attn_shock) > 2 ? 'var(--amber)'
              : entry.attn_shock > 0         ? 'var(--accent)'
              : 'var(--text-3)'
            } />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/* ── Credibility Breakdown ───────────────────────────────────────────────────*/
function PenaltyBar({ label, weight, rawValue, penalty, color }) {
  const pct = Math.min(100, (rawValue / 0.55) * 100);
  return (
    <div style={{ paddingBottom: 14, borderBottom: '1px solid var(--border)', marginBottom: 14 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 12, color: 'var(--text-2)', fontFamily: 'var(--mono)' }}>{label}</span>
          <span style={{
            fontSize: 10, color: 'var(--text-3)', fontFamily: 'var(--mono)',
            padding: '1px 6px', border: '1px solid var(--border)', borderRadius: 3,
          }}>
            {weight}
          </span>
        </div>
        <div style={{ textAlign: 'right' }}>
          <span style={{ fontSize: 12, fontFamily: 'var(--mono)', color }}>
            −{(penalty * 100).toFixed(1)}%
          </span>
          <span style={{ fontSize: 10, color: 'var(--text-3)', fontFamily: 'var(--mono)', marginLeft: 8 }}>
            raw {rawValue.toFixed(3)}
          </span>
        </div>
      </div>
      <div style={{ height: 5, background: 'var(--border)', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{
          height: '100%', width: `${pct}%`, background: color, borderRadius: 3,
          transition: 'width 0.4s ease',
        }} />
      </div>
    </div>
  );
}

function CredibilityBreakdown({ bd }) {
  if (!bd) return null;
  const scoreColor = bd.composite >= 0.75 ? 'var(--green)'
    : bd.composite >= 0.60 ? 'var(--amber)' : 'var(--red)';
  const netPct   = (1 - bd.total_penalty) * 100;
  const dupPct   = bd.dup_penalty   * 100;
  const herfPct  = bd.herf_penalty  * 100;
  const burstPct = bd.burst_penalty * 100;

  return (
    <div className="card" style={{ padding: 24 }}>
      <div style={{ marginBottom: 20 }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-1)' }}>Credibility Breakdown</div>
        <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 2 }}>
          Cred = 1 − α·DupRate − β·Herfindahl − γ·Burstiness
        </div>
      </div>

      {/* Big composite score */}
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, marginBottom: 24 }}>
        <span style={{ fontSize: 48, fontWeight: 700, fontFamily: 'var(--mono)', color: scoreColor, lineHeight: 1 }}>
          {(bd.composite * 100).toFixed(1)}%
        </span>
        <div style={{ fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--mono)', lineHeight: 1.6 }}>
          composite<br />credibility
        </div>
      </div>

      {/* Stacked bar */}
      <div style={{ display: 'flex', height: 8, borderRadius: 4, overflow: 'hidden', marginBottom: 8 }}>
        <div style={{ width: `${netPct}%`,   background: scoreColor,     opacity: 0.85 }} />
        <div style={{ width: `${dupPct}%`,   background: 'var(--amber)', opacity: 0.8  }} />
        <div style={{ width: `${herfPct}%`,  background: '#f97316',      opacity: 0.8  }} />
        <div style={{ width: `${burstPct}%`, background: 'var(--red)',   opacity: 0.8  }} />
      </div>
      <div style={{ display: 'flex', gap: 14, marginBottom: 24, flexWrap: 'wrap' }}>
        {[
          { label: 'Net score',  color: scoreColor       },
          { label: 'Dup rate',   color: 'var(--amber)'   },
          { label: 'Herf.',      color: '#f97316'         },
          { label: 'Burstiness', color: 'var(--red)'     },
        ].map(({ label, color }) => (
          <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <div style={{ width: 8, height: 8, borderRadius: 2, background: color }} />
            <span style={{ fontSize: 10, color: 'var(--text-3)', fontFamily: 'var(--mono)' }}>{label}</span>
          </div>
        ))}
      </div>

      {/* Per-component penalty bars */}
      <PenaltyBar label="Duplication Rate"     weight="α = 0.33"
        rawValue={bd.dup_rate}   penalty={bd.dup_penalty}   color="var(--amber)" />
      <PenaltyBar label="Author Concentration" weight="β = 0.33"
        rawValue={bd.herfindahl} penalty={bd.herf_penalty}  color="#f97316" />
      <PenaltyBar label="Burstiness"           weight="γ = 0.34"
        rawValue={bd.burstiness} penalty={bd.burst_penalty} color="var(--red)" />

      {/* Formula row */}
      <div style={{
        padding: '10px 14px', background: 'var(--bg-card-alt)',
        borderRadius: 6, border: '1px solid var(--border)',
        fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--text-3)',
      }}>
        1.000 &nbsp;−&nbsp;
        <span style={{ color: 'var(--amber)'  }}>{dupPct.toFixed(1)}%</span>
        &nbsp;−&nbsp;
        <span style={{ color: '#f97316'        }}>{herfPct.toFixed(1)}%</span>
        &nbsp;−&nbsp;
        <span style={{ color: 'var(--red)'    }}>{burstPct.toFixed(1)}%</span>
        &nbsp;=&nbsp;
        <span style={{ color: scoreColor, fontWeight: 700 }}>{(bd.composite * 100).toFixed(1)}%</span>
      </div>
    </div>
  );
}

/* ── Top Reddit Posts ────────────────────────────────────────────────────────*/
const SENT_STYLE = {
  bullish: { color: 'var(--green)', bg: 'rgba(16,185,129,0.1)',  border: 'rgba(16,185,129,0.25)' },
  bearish: { color: 'var(--red)',   bg: 'rgba(239,68,68,0.1)',   border: 'rgba(239,68,68,0.25)'  },
  neutral: { color: 'var(--text-3)',bg: 'rgba(148,163,184,0.07)',border: 'rgba(148,163,184,0.2)' },
};

function PostCard({ post }) {
  const sentiment = post.sentiment || 'neutral';
  const st        = SENT_STYLE[sentiment] || SENT_STYLE.neutral;
  const author    = post.author || 'unknown';
  const initial   = author.replace('u/', '').slice(0, 2).toUpperCase();
  return (
    <div style={{
      padding: '14px 16px',
      background: 'var(--bg-card-alt)',
      border: '1px solid var(--border)',
      borderRadius: 8,
    }}>
      {/* Author row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
        <div style={{
          width: 30, height: 30, borderRadius: '50%', flexShrink: 0,
          background: 'rgba(0,212,184,0.12)', border: '1px solid rgba(0,212,184,0.2)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontFamily: 'var(--mono)', fontSize: 10, fontWeight: 700, color: 'var(--accent)',
        }}>
          {initial}
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--text-2)',
            whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {author}
          </div>
          <div style={{ fontSize: 10, color: 'var(--text-3)', marginTop: 1 }}>
            {post.subreddit}{post.time_ago ? ` · ${post.time_ago}` : ''}
          </div>
        </div>
        <span style={{
          padding: '2px 8px', borderRadius: 4,
          border: `1px solid ${st.border}`, background: st.bg, color: st.color,
          fontFamily: 'var(--mono)', fontSize: 9, fontWeight: 700, letterSpacing: '0.08em',
          whiteSpace: 'nowrap',
        }}>
          {sentiment.toUpperCase()}
        </span>
      </div>

      {/* Post title */}
      {post.title && (
        <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-1)', lineHeight: 1.5, marginBottom: 6 }}>
          {post.title}
        </div>
      )}

      {/* Post body */}
      {post.text && (
        <div style={{ fontSize: 12, color: 'var(--text-2)', lineHeight: 1.6, marginBottom: 10 }}>
          {post.text.length > 200 ? post.text.slice(0, 200) + '…' : post.text}
        </div>
      )}

      {/* Upvote row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <span style={{ fontSize: 10, color: 'var(--text-3)', fontFamily: 'var(--mono)' }}>▲</span>
        <span style={{ fontSize: 11, fontFamily: 'var(--mono)', color: 'var(--text-2)', fontWeight: 600 }}>
          {(post.score || 0).toLocaleString()}
        </span>
        <span style={{ fontSize: 10, color: 'var(--text-3)', fontFamily: 'var(--mono)' }}>upvotes</span>
      </div>
    </div>
  );
}

function TopPostsCard({ posts }) {
  const isSpecific = posts?.length > 0 && posts[0]?.is_specific !== false;
  return (
    <div className="card" style={{ padding: 24, display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div style={{ marginBottom: 4 }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-1)' }}>
          {isSpecific ? 'Top Driving Posts' : 'Related Market Discussion'}
        </div>
        <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 2 }}>
          {isSpecific
            ? "Highest-impact Reddit posts in today\u2019s signal window"
            : 'No ticker-specific posts found for this date \u2014 showing highest-upvoted related posts'}
        </div>
      </div>
      {posts?.length
        ? posts.map((p, i) => <PostCard key={i} post={p} />)
        : (
          <div style={{
            padding: '24px 0', textAlign: 'center',
            color: 'var(--text-3)', fontSize: 12,
            fontFamily: 'var(--mono)', fontStyle: 'italic',
          }}>
            No individual posts available for this date.
          </div>
        )
      }
    </div>
  );
}

/* ── Feature contribution bar ────────────────────────────────────────────────*/
function ContribBar({ label, value, contribution, maxContrib }) {
  const pct = Math.abs(contribution) / maxContrib * 100;
  const pos = contribution >= 0;
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '7px 0', borderBottom: '1px solid var(--border)' }}>
      <div style={{ width: 160, fontSize: 12, color: 'var(--text-2)', fontFamily: 'var(--mono)', flexShrink: 0 }}>{label}</div>
      <div style={{ flex: 1, height: 6, background: 'var(--border)', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{ height: '100%', borderRadius: 3, width: `${pct}%`,
          background: pos ? 'var(--green)' : 'var(--red)' }} />
      </div>
      <div style={{ width: 72, fontFamily: 'var(--mono)', fontSize: 12, textAlign: 'right',
        color: pos ? 'var(--green)' : 'var(--red)' }}>
        {contribution >= 0 ? '+' : ''}{contribution.toFixed(4)}
      </div>
      <div style={{ width: 64, fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--text-3)', textAlign: 'right' }}>
        {value.toFixed(4)}
      </div>
    </div>
  );
}

/* ── Main page ───────────────────────────────────────────────────────────────*/
export default function StockDetailPage() {
  const { ticker }          = useParams();
  const navigate            = useNavigate();
  const [data, setData]     = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.stock(ticker)
      .then(d => { setData(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, [ticker]);

  if (loading) return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '60vh' }}>
      <span style={{ fontFamily: 'var(--mono)', color: 'var(--accent)', fontSize: 13, letterSpacing: '0.1em' }}>
        LOADING {ticker}…
      </span>
    </div>
  );
  if (!data) return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '60vh' }}>
      <span style={{ fontFamily: 'var(--mono)', color: 'var(--red)', fontSize: 12 }}>
        TICKER NOT FOUND — {ticker}
      </span>
    </div>
  );

  const {
    history,
    feature_contributions: contribs,
    credibility_breakdown: credBd,
    top_posts: topPosts,
  } = data;

  const recent5    = history.slice(-5);
  const maxContrib = Math.max(...contribs.map(c => Math.abs(c.contribution)));
  const radarData  = contribs.map(c => ({
    feature:  c.label.replace(' ', '\n'),
    value:    Math.max(0, Math.min(1, (c.value + 1) / 2)),
    fullMark: 1,
  }));

  return (
    <div style={S.root}>
      {/* Back + breadcrumb */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <button className="btn btn-ghost btn-sm" onClick={() => navigate('/')}>← Dashboard</button>
        <span style={{ color: 'var(--text-3)', fontSize: 12 }}>→</span>
        <span style={{ fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--text-2)' }}>{ticker}</span>
      </div>

      {/* Stock header */}
      <div className="card" style={{ padding: '24px 28px' }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 24, flexWrap: 'wrap' }}>
          <div style={{ flex: 1 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 8 }}>
              <span style={{ fontFamily: 'var(--mono)', fontSize: 32, fontWeight: 700,
                color: 'var(--text-1)', letterSpacing: '0.04em' }}>
                {ticker}
              </span>
              <span style={{ fontSize: 12, color: 'var(--text-3)', fontFamily: 'var(--mono)',
                padding: '3px 10px', border: '1px solid var(--border)', borderRadius: 4 }}>
                {data.sector}
              </span>
              <CredBadge label={data.credibility_label} score={data.credibility_score} />
            </div>
            <div style={{ fontSize: 13, color: 'var(--text-2)', lineHeight: 1.6, maxWidth: 600 }}>
              {data.explanation}
            </div>
          </div>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            <MetricPill label="RANK"      value={`#${data.rank}`}          color="var(--accent)" />
            <MetricPill label="ADJ SCORE" value={data.adj_score.toFixed(4)} color="var(--accent)" />
            <MetricPill label="SENTIMENT"
              value={data.tone_finbert >= 0
                ? `+${data.tone_finbert.toFixed(3)}`
                : data.tone_finbert.toFixed(3)}
              color={data.tone_finbert >= 0 ? 'var(--green)' : 'var(--red)'} />
            <MetricPill label="ATTN σ"
              value={`${data.attn_shock > 0 ? '+' : ''}${data.attn_shock.toFixed(2)}σ`}
              color={Math.abs(data.attn_shock) > 1 ? 'var(--amber)' : 'var(--text-2)'} />
            <MetricPill label="POSTS" value={data.n_posts} color="var(--text-1)" />
          </div>
        </div>
      </div>

      {/* Row 1: 5-day Sentiment + 5-day Volume */}
      <div style={S.grid2}>
        <ChartCard
          title="Sentiment Trend — last 5 days"
          sub="FinBERT score per day  (P(positive) − P(negative)  ∈ [−1, +1])"
        >
          <SentimentTrend data={recent5} />
        </ChartCard>

        <ChartCard
          title="Post Volume — last 5 days"
          sub="Bar colour: teal = normal · amber = attention spike (attn_shock > 2σ)"
        >
          <VolumeBars data={recent5} />
        </ChartCard>
      </div>

      {/* Row 2: Attention Shock 30-day full width */}
      <ChartCard
        title="Attention Shock (σ) — 30-day history"
        sub="Rolling z-score of daily post volume  (window = 20 days) · amber = |shock| > 2σ"
      >
        <AttnShockChart data={history} />
      </ChartCard>

      {/* Row 3: Credibility Breakdown + Top Posts */}
      <div style={S.grid2}>
        <CredibilityBreakdown bd={credBd} />
        <TopPostsCard posts={topPosts} />
      </div>

      {/* Row 4: Feature Contributions + Radar */}
      <div style={S.bottomGrid}>
        <div className="card" style={{ padding: 24, flex: 2 }}>
          <div style={{ marginBottom: 16 }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-1)' }}>Feature Contributions</div>
            <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 2 }}>
              Normalised feature × model coefficient — larger bar = stronger driver of today's rank
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 10,
            padding: '0 0 8px', borderBottom: '1px solid var(--border)' }}>
            <div style={{ width: 160, fontSize: 10, fontWeight: 700, letterSpacing: '0.1em', color: 'var(--text-3)', fontFamily: 'var(--mono)' }}>FEATURE</div>
            <div style={{ flex: 1,   fontSize: 10, fontWeight: 700, letterSpacing: '0.1em', color: 'var(--text-3)', fontFamily: 'var(--mono)' }}>CONTRIBUTION</div>
            <div style={{ width: 72, fontSize: 10, fontWeight: 700, letterSpacing: '0.1em', color: 'var(--text-3)', fontFamily: 'var(--mono)', textAlign: 'right' }}>SCORE</div>
            <div style={{ width: 64, fontSize: 10, fontWeight: 700, letterSpacing: '0.1em', color: 'var(--text-3)', fontFamily: 'var(--mono)', textAlign: 'right' }}>VALUE</div>
          </div>
          {[...contribs]
            .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
            .map(c => (
              <ContribBar key={c.feature}
                label={c.label} value={c.value}
                contribution={c.contribution} maxContrib={maxContrib}
              />
            ))}
        </div>

        <div className="card" style={{ padding: 24, flex: 1, minWidth: 280 }}>
          <div style={{ marginBottom: 8 }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-1)' }}>Signal Profile</div>
            <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 2 }}>Feature values (normalised 0–1)</div>
          </div>
          <ResponsiveContainer width="100%" height={240}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="var(--border)" />
              <PolarAngleAxis dataKey="feature" tick={{ fontSize: 10, fill: 'var(--text-3)' }} />
              <Radar dataKey="value" name={ticker}
                stroke="var(--accent)" fill="var(--accent)" fillOpacity={0.15}
                strokeWidth={2}
              />
              <Tooltip contentStyle={{ background: '#0f1729', border: '1px solid var(--border-b)',
                borderRadius: 6, fontFamily: 'var(--mono)', fontSize: 12 }} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

const S = {
  root:       { padding: '24px 28px', maxWidth: 1400, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 20 },
  grid2:      { display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 },
  bottomGrid: { display: 'flex', gap: 16, flexWrap: 'wrap' },
};
