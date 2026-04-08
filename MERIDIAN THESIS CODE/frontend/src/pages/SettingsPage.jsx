import { useState, useCallback, useRef } from 'react';

// ── Defaults ────────────────────────────────────────────────────────────────
const DEFAULTS = {
  // Signal Preferences
  subreddits: {
    wallstreetbets:  true,
    stocks:          true,
    investing:       true,
    SecurityAnalysis: false,
    StockMarket:     true,
  },
  minPosts:   3,
  shortlistK: 10,

  // Notifications
  dailyAlert:      false,
  alertEmail:      '',
  lowCredWarning:  true,

  // Display
  showAdjScore:  true,
  showAttnShock: true,
  showPosts:     true,
  density:       'comfortable',   // 'compact' | 'comfortable'
};

// ── Primitive controls ───────────────────────────────────────────────────────

function Toggle({ value, onChange, disabled = false }) {
  return (
    <button
      onClick={() => !disabled && onChange(!value)}
      aria-checked={value}
      role="switch"
      style={{
        width: 44, height: 24, borderRadius: 12, flexShrink: 0,
        background: value ? 'var(--accent)' : 'var(--border-b)',
        border: 'none', position: 'relative',
        transition: 'background 0.22s ease',
        cursor: disabled ? 'not-allowed' : 'pointer',
        opacity: disabled ? 0.4 : 1,
        boxShadow: value ? '0 0 10px rgba(0,212,184,0.25)' : 'none',
      }}
    >
      <span style={{
        position: 'absolute',
        top: 3, left: value ? 23 : 3,
        width: 18, height: 18, borderRadius: '50%',
        background: '#fff',
        transition: 'left 0.22s cubic-bezier(0.34,1.56,0.64,1)',
        display: 'block',
        boxShadow: '0 1px 4px rgba(0,0,0,0.4)',
      }} />
    </button>
  );
}

function Slider({ value, onChange, min, max, step = 1 }) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
      <div style={{ position: 'relative', flex: 1 }}>
        {/* Custom track fill */}
        <div style={{
          position: 'absolute', top: '50%', left: 0,
          height: 4, borderRadius: 2,
          width: `${pct}%`,
          background: 'var(--accent)',
          transform: 'translateY(-50%)',
          pointerEvents: 'none', zIndex: 1,
          boxShadow: '0 0 8px rgba(0,212,184,0.4)',
        }} />
        <input
          type="range" min={min} max={max} step={step} value={value}
          onChange={e => onChange(Number(e.target.value))}
          style={{
            width: '100%', height: 4,
            appearance: 'none', WebkitAppearance: 'none',
            background: 'var(--border)',
            borderRadius: 2, outline: 'none',
            position: 'relative', zIndex: 2,
            cursor: 'pointer',
          }}
        />
      </div>
      <span style={{
        fontFamily: 'var(--mono)', fontSize: 16, fontWeight: 700,
        color: 'var(--accent)', width: 28, textAlign: 'right', flexShrink: 0,
      }}>
        {value}
      </span>
    </div>
  );
}

function SegmentedControl({ value, onChange, options }) {
  return (
    <div style={{
      display: 'inline-flex',
      background: 'var(--bg-card-alt)',
      border: '1px solid var(--border)',
      borderRadius: 8, padding: 3, gap: 2,
    }}>
      {options.map(o => {
        const active = value === o.value;
        return (
          <button
            key={o.value}
            onClick={() => onChange(o.value)}
            style={{
              padding: '6px 18px', borderRadius: 6, border: 'none',
              fontFamily: 'var(--mono)', fontSize: 13, fontWeight: 600,
              letterSpacing: '0.04em', cursor: 'pointer',
              transition: 'all 0.18s ease',
              background: active ? 'var(--accent)' : 'transparent',
              color: active ? '#000' : 'var(--text-3)',
              boxShadow: active ? '0 0 12px rgba(0,212,184,0.3)' : 'none',
            }}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

function EmailInput({ value, onChange, disabled }) {
  const valid = !value || /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value);
  return (
    <div style={{ position: 'relative' }}>
      <input
        type="email"
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder="analyst@institution.com"
        disabled={disabled}
        style={{
          width: 280, padding: '9px 14px 9px 36px',
          background: disabled ? '#07101e' : 'var(--bg-input)',
          border: `1px solid ${!valid ? 'var(--red)' : value ? 'var(--accent-dim)' : 'var(--border-b)'}`,
          borderRadius: 6, color: disabled ? 'var(--text-3)' : 'var(--text-1)',
          fontSize: 13, fontFamily: 'var(--mono)', outline: 'none',
          transition: 'border-color 0.15s',
          cursor: disabled ? 'not-allowed' : 'text',
        }}
      />
      <span style={{
        position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)',
        fontSize: 14, opacity: disabled ? 0.3 : 0.6,
      }}>✉</span>
      {!valid && value && (
        <span style={{
          position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)',
          fontSize: 11, color: 'var(--red)', fontFamily: 'var(--mono)',
        }}>invalid</span>
      )}
    </div>
  );
}

// ── Layout primitives ────────────────────────────────────────────────────────

function Section({ id, title, icon, sub, dirty, children }) {
  return (
    <div id={id} className="card" style={{ overflow: 'hidden' }}>
      {/* Section header */}
      <div style={{
        padding: '18px 24px',
        borderBottom: '1px solid var(--border)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        background: 'rgba(255,255,255,0.01)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontSize: 16 }}>{icon}</span>
          <div>
            <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-1)', letterSpacing: '-0.01em' }}>{title}</div>
            {sub && <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 1 }}>{sub}</div>}
          </div>
        </div>
        {dirty && (
          <span style={{
            fontSize: 10, fontWeight: 700, letterSpacing: '0.1em',
            color: 'var(--amber)', fontFamily: 'var(--mono)',
            padding: '2px 8px', borderRadius: 3,
            background: 'rgba(245,158,11,0.1)',
            border: '1px solid rgba(245,158,11,0.25)',
          }}>
            UNSAVED
          </span>
        )}
      </div>
      <div style={{ padding: '20px 24px', display: 'flex', flexDirection: 'column', gap: 20 }}>
        {children}
      </div>
    </div>
  );
}

function Row({ label, sub, children, borderless = false }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center',
      justifyContent: 'space-between', gap: 24,
      paddingBottom: borderless ? 0 : 18,
      borderBottom: borderless ? 'none' : '1px solid var(--border)',
    }}>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-1)' }}>{label}</div>
        {sub && <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 3, lineHeight: 1.5 }}>{sub}</div>}
      </div>
      <div style={{ flexShrink: 0 }}>{children}</div>
    </div>
  );
}

function Divider() {
  return <div style={{ height: 1, background: 'var(--border)', margin: '4px 0' }} />;
}

// ── Subreddit tile ───────────────────────────────────────────────────────────

const SUB_META = {
  wallstreetbets:   { abbr: 'r/wsb',  label: 'r/wallstreetbets',   desc: 'Retail momentum & meme stocks',      posts: '~2,400/day' },
  stocks:           { abbr: 'r/stk',  label: 'r/stocks',           desc: 'Fundamental analysis & earnings',     posts: '~680/day'   },
  investing:        { abbr: 'r/inv',  label: 'r/investing',         desc: 'Long-term equity discussion',         posts: '~520/day'   },
  SecurityAnalysis: { abbr: 'r/sa',   label: 'r/SecurityAnalysis',  desc: 'Deep-dive valuation research',        posts: '~140/day'   },
  StockMarket:      { abbr: 'r/sm',   label: 'r/StockMarket',       desc: 'News-driven market commentary',       posts: '~310/day'   },
};

function SubredditTile({ name, enabled, onChange }) {
  const m = SUB_META[name];
  return (
    <button
      onClick={() => onChange(!enabled)}
      style={{
        background: enabled ? 'rgba(0,212,184,0.06)' : 'var(--bg-card-alt)',
        border: `1px solid ${enabled ? 'rgba(0,212,184,0.35)' : 'var(--border)'}`,
        borderRadius: 8, padding: '12px 14px',
        cursor: 'pointer', textAlign: 'left',
        transition: 'all 0.2s ease',
        display: 'flex', flexDirection: 'column', gap: 6,
        boxShadow: enabled ? '0 0 16px rgba(0,212,184,0.07)' : 'none',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span style={{
          fontFamily: 'var(--mono)', fontSize: 12, fontWeight: 700,
          color: enabled ? 'var(--accent)' : 'var(--text-3)',
          letterSpacing: '0.04em',
        }}>
          {m.label}
        </span>
        <span style={{
          width: 8, height: 8, borderRadius: '50%', flexShrink: 0,
          background: enabled ? 'var(--accent)' : 'var(--border-b)',
          boxShadow: enabled ? '0 0 6px rgba(0,212,184,0.6)' : 'none',
          transition: 'all 0.2s',
        }} />
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-3)', lineHeight: 1.4 }}>{m.desc}</div>
      <div style={{
        fontSize: 10, fontFamily: 'var(--mono)',
        color: enabled ? 'var(--accent-dim)' : 'var(--text-3)',
        letterSpacing: '0.06em',
      }}>
        {m.posts}
      </div>
    </button>
  );
}

// ── Column visibility chip ───────────────────────────────────────────────────

function ColChip({ label, visible, onChange }) {
  return (
    <button
      onClick={() => onChange(!visible)}
      style={{
        display: 'inline-flex', alignItems: 'center', gap: 6,
        padding: '6px 14px', borderRadius: 6, cursor: 'pointer',
        fontFamily: 'var(--mono)', fontSize: 12, fontWeight: 700,
        letterSpacing: '0.06em', border: 'none',
        transition: 'all 0.18s ease',
        background: visible ? 'rgba(0,212,184,0.1)' : 'rgba(74,88,120,0.15)',
        color: visible ? 'var(--accent)' : 'var(--text-3)',
        outline: `1px solid ${visible ? 'rgba(0,212,184,0.3)' : 'var(--border)'}`,
      }}
    >
      <span style={{ fontSize: 10, opacity: 0.8 }}>{visible ? '◉' : '○'}</span>
      {label}
    </button>
  );
}

// ── Density preview ──────────────────────────────────────────────────────────

function DensityPicker({ value, onChange }) {
  const opts = [
    { value: 'compact',     label: 'Compact',      rowH: 8  },
    { value: 'comfortable', label: 'Comfortable',   rowH: 14 },
  ];
  return (
    <div style={{ display: 'flex', gap: 10 }}>
      {opts.map(o => (
        <button
          key={o.value}
          onClick={() => onChange(o.value)}
          style={{
            padding: '10px 16px', borderRadius: 8, cursor: 'pointer',
            border: `1px solid ${value === o.value ? 'rgba(0,212,184,0.4)' : 'var(--border)'}`,
            background: value === o.value ? 'rgba(0,212,184,0.07)' : 'var(--bg-card-alt)',
            display: 'flex', flexDirection: 'column', gap: 8,
            transition: 'all 0.18s',
            minWidth: 110,
          }}
        >
          {/* Mini table preview */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: o.rowH === 8 ? 3 : 6 }}>
            {[0.8, 0.55, 0.65].map((w, i) => (
              <div key={i} style={{
                height: 4, borderRadius: 2,
                background: i === 0 ? 'rgba(0,212,184,0.5)' : 'var(--border-b)',
                width: `${w * 100}%`,
              }} />
            ))}
          </div>
          <div style={{
            fontSize: 11, fontFamily: 'var(--mono)', fontWeight: 700,
            letterSpacing: '0.06em',
            color: value === o.value ? 'var(--accent)' : 'var(--text-3)',
          }}>
            {o.label.toUpperCase()}
          </div>
        </button>
      ))}
    </div>
  );
}

// ── Save toast ───────────────────────────────────────────────────────────────

function SaveToast({ state }) {
  // state: 'idle' | 'saving' | 'saved'
  if (state === 'idle') return null;
  return (
    <div style={{
      position: 'fixed', bottom: 28, right: 28, zIndex: 999,
      padding: '12px 20px', borderRadius: 8,
      display: 'flex', alignItems: 'center', gap: 10,
      background: state === 'saved' ? 'rgba(16,185,129,0.12)' : 'rgba(0,212,184,0.1)',
      border: `1px solid ${state === 'saved' ? 'rgba(16,185,129,0.4)' : 'rgba(0,212,184,0.3)'}`,
      color: state === 'saved' ? 'var(--green)' : 'var(--accent)',
      fontFamily: 'var(--mono)', fontSize: 12, fontWeight: 700,
      letterSpacing: '0.08em',
      boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
      transition: 'all 0.2s ease',
    }}>
      {state === 'saving'
        ? <><Spinner />SAVING…</>
        : <><span style={{ fontSize: 14 }}>✓</span> SETTINGS SAVED</>
      }
    </div>
  );
}

function Spinner() {
  return (
    <span style={{
      width: 12, height: 12, borderRadius: '50%',
      border: '2px solid rgba(0,212,184,0.3)',
      borderTopColor: 'var(--accent)',
      display: 'inline-block',
      animation: 'spin 0.75s linear infinite',
    }} />
  );
}

// ── About data source card ───────────────────────────────────────────────────

function DataSourceCard({ icon, name, detail, url }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'flex-start', gap: 12,
      padding: '14px 16px',
      background: 'var(--bg-card-alt)',
      border: '1px solid var(--border)',
      borderRadius: 8,
    }}>
      <span style={{ fontSize: 20, lineHeight: 1, flexShrink: 0, marginTop: 1 }}>{icon}</span>
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-1)', marginBottom: 3 }}>{name}</div>
        <div style={{ fontSize: 11, color: 'var(--text-3)', lineHeight: 1.5 }}>{detail}</div>
      </div>
      {url && (
        <span style={{
          fontSize: 10, fontFamily: 'var(--mono)', fontWeight: 700,
          color: 'var(--accent)', letterSpacing: '0.06em',
          padding: '3px 8px', border: '1px solid rgba(0,212,184,0.25)',
          borderRadius: 4, flexShrink: 0, marginTop: 2,
          cursor: 'default',
        }}>
          {url}
        </span>
      )}
    </div>
  );
}

// ── Nav sidebar ──────────────────────────────────────────────────────────────

function SideNav({ sections, activeSec }) {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', gap: 2,
      position: 'sticky', top: 24, alignSelf: 'flex-start',
      width: 180, flexShrink: 0,
    }}>
      <div style={{
        fontSize: 10, fontWeight: 700, letterSpacing: '0.12em',
        color: 'var(--text-3)', fontFamily: 'var(--mono)',
        padding: '0 10px 8px', borderBottom: '1px solid var(--border)',
        marginBottom: 4,
      }}>
        SECTIONS
      </div>
      {sections.map(s => (
        <a
          key={s.id}
          href={`#${s.id}`}
          style={{
            display: 'flex', alignItems: 'center', gap: 8,
            padding: '8px 10px', borderRadius: 6,
            fontSize: 12, fontWeight: 500,
            color: activeSec === s.id ? 'var(--accent)' : 'var(--text-3)',
            background: activeSec === s.id ? 'rgba(0,212,184,0.08)' : 'transparent',
            textDecoration: 'none', transition: 'all 0.15s',
          }}
        >
          <span style={{ fontSize: 14 }}>{s.icon}</span>
          {s.title}
        </a>
      ))}
    </div>
  );
}

// ── Main page ────────────────────────────────────────────────────────────────

const SECTIONS = [
  { id: 'signal',  icon: '◆', title: 'Signal Prefs'   },
  { id: 'notifs',  icon: '◇', title: 'Notifications'  },
  { id: 'display', icon: '◧',  title: 'Display'        },
  { id: 'about',   icon: '◈',  title: 'About'          },
];

export default function SettingsPage() {
  const [toastState, setToastState] = useState('idle');   // 'idle'|'saving'|'saved'
  const toastTimer = useRef(null);

  // ── State ──────────────────────────────────────────────────────────────────
  const [subreddits,    setSubreddits]    = useState(DEFAULTS.subreddits);
  const [minPosts,      setMinPosts]      = useState(DEFAULTS.minPosts);
  const [shortlistK,    setShortlistK]    = useState(DEFAULTS.shortlistK);
  const [signalDirty,   setSignalDirty]   = useState(false);

  const [dailyAlert,    setDailyAlert]    = useState(DEFAULTS.dailyAlert);
  const [alertEmail,    setAlertEmail]    = useState(DEFAULTS.alertEmail);
  const [lowCredWarn,   setLowCredWarn]   = useState(DEFAULTS.lowCredWarning);
  const [notifDirty,    setNotifDirty]    = useState(false);

  const [showAdjScore,  setShowAdjScore]  = useState(DEFAULTS.showAdjScore);
  const [showAttnShock, setShowAttnShock] = useState(DEFAULTS.showAttnShock);
  const [showPosts,     setShowPosts]     = useState(DEFAULTS.showPosts);
  const [density,       setDensity]       = useState(DEFAULTS.density);
  const [displayDirty,  setDisplayDirty]  = useState(false);

  // Dirty helpers
  const markSignal  = useCallback(() => setSignalDirty(true),  []);
  const markNotif   = useCallback(() => setNotifDirty(true),   []);
  const markDisplay = useCallback(() => setDisplayDirty(true), []);

  function set(setter, dirtier) {
    return v => { setter(v); dirtier(); };
  }

  // ── Save ───────────────────────────────────────────────────────────────────
  function handleSave() {
    clearTimeout(toastTimer.current);
    setToastState('saving');
    setTimeout(() => {
      setToastState('saved');
      setSignalDirty(false);
      setNotifDirty(false);
      setDisplayDirty(false);
      toastTimer.current = setTimeout(() => setToastState('idle'), 2400);
    }, 600);
  }

  const anyDirty = signalDirty || notifDirty || displayDirty;
  const enabledSubCount = Object.values(subreddits).filter(Boolean).length;

  return (
    <div style={S.page}>
      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        input[type=range]::-webkit-slider-thumb {
          -webkit-appearance: none; appearance: none;
          width: 16px; height: 16px; border-radius: 50%;
          background: var(--accent);
          border: 2px solid #050810;
          cursor: pointer;
          box-shadow: 0 0 8px rgba(0,212,184,0.5);
        }
        input[type=range]::-moz-range-thumb {
          width: 16px; height: 16px; border-radius: 50%;
          background: var(--accent);
          border: 2px solid #050810;
          cursor: pointer;
        }
        a[href^="#"]:hover { color: var(--accent) !important; }
      `}</style>

      {/* Top bar */}
      <div style={S.topBar}>
        <div>
          <div style={S.pageTitle}>Settings</div>
          <div style={S.pageSub}>
            Platform configuration · changes persist for this session
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          {anyDirty && (
            <span style={S.unsavedBadge}>UNSAVED CHANGES</span>
          )}
          <button
            className="btn btn-primary"
            onClick={handleSave}
            style={{ opacity: anyDirty ? 1 : 0.55, transition: 'opacity 0.2s' }}
          >
            SAVE CHANGES
          </button>
        </div>
      </div>

      <div style={S.body}>
        {/* Sidebar nav */}
        <SideNav sections={SECTIONS} activeSec={null} />

        {/* Sections */}
        <div style={S.sections}>

          {/* ── 1. Signal Preferences ─────────────────────────────────────── */}
          <Section
            id="signal"
            icon="◆"
            title="Signal Preferences"
            sub="Controls which data sources feed into the daily shortlist"
            dirty={signalDirty}
          >
            {/* Subreddits */}
            <div>
              <div style={S.fieldLabel}>
                SUBREDDITS
                <span style={S.fieldBadge}>{enabledSubCount} / 5 active</span>
              </div>
              <div style={S.subGrid}>
                {Object.keys(subreddits).map(name => (
                  <SubredditTile
                    key={name}
                    name={name}
                    enabled={subreddits[name]}
                    onChange={v => {
                      setSubreddits(prev => ({ ...prev, [name]: v }));
                      markSignal();
                    }}
                  />
                ))}
              </div>
              {enabledSubCount === 0 && (
                <div style={S.inlineWarn}>
                  At least one subreddit must be enabled for signal generation.
                </div>
              )}
            </div>

            <Divider />

            {/* Min posts */}
            <div>
              <Row
                label="Minimum posts threshold"
                sub="Tickers with fewer posts than this on a given day are excluded from ranking (minimum evidence rule)."
                borderless
              >
                <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--text-3)', letterSpacing: '0.06em' }}>
                  1 ←→ 10
                </span>
              </Row>
              <div style={{ marginTop: 12 }}>
                <Slider
                  value={minPosts}
                  onChange={set(setMinPosts, markSignal)}
                  min={1} max={10}
                />
                <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 6 }}>
                  {[1,2,3,4,5,6,7,8,9,10].map(n => (
                    <span key={n} style={{
                      fontFamily: 'var(--mono)', fontSize: 10,
                      color: n === minPosts ? 'var(--accent)' : 'var(--text-3)',
                      fontWeight: n === minPosts ? 700 : 400,
                    }}>{n}</span>
                  ))}
                </div>
              </div>
            </div>

            <Divider />

            {/* Shortlist size */}
            <Row
              label="Shortlist size (K)"
              sub="Number of tickers surfaced in the daily Top-K shortlist."
              borderless
            >
              <SegmentedControl
                value={shortlistK}
                onChange={set(setShortlistK, markSignal)}
                options={[5, 10, 15, 20].map(n => ({ value: n, label: String(n) }))}
              />
            </Row>
          </Section>

          {/* ── 2. Notifications ───────────────────────────────────────────── */}
          <Section
            id="notifs"
            icon="◇"
            title="Notifications"
            sub="Alert delivery at market close — requires email configuration"
            dirty={notifDirty}
          >
            <Row
              label="Daily digest at market close"
              sub="Sends the Top-K shortlist to your email at 16:00 ET (NYSE close) on every trading day."
            >
              <Toggle value={dailyAlert} onChange={set(setDailyAlert, markNotif)} />
            </Row>

            {/* Email input — animated reveal */}
            <div style={{
              overflow: 'hidden',
              maxHeight: dailyAlert ? 100 : 0,
              opacity: dailyAlert ? 1 : 0,
              transition: 'max-height 0.3s ease, opacity 0.25s ease',
            }}>
              <div style={{
                padding: '14px 16px',
                background: 'rgba(0,212,184,0.04)',
                border: '1px solid rgba(0,212,184,0.15)',
                borderRadius: 8, marginTop: -4,
              }}>
                <div style={S.fieldLabel}>DELIVERY ADDRESS</div>
                <div style={{ marginTop: 8 }}>
                  <EmailInput
                    value={alertEmail}
                    onChange={set(setAlertEmail, markNotif)}
                    disabled={!dailyAlert}
                  />
                </div>
              </div>
            </div>

            <Divider />

            <Row
              label="Low-credibility warnings"
              sub="Show a visual indicator on the dashboard when the shortlist contains tickers with LOW credibility scores."
              borderless
            >
              <Toggle value={lowCredWarn} onChange={set(setLowCredWarn, markNotif)} />
            </Row>

            {/* Info note */}
            <div style={S.infoNote}>
              <span style={{ fontSize: 14 }}>ℹ</span>
              <span>
                Notification delivery requires SMTP credentials configured in the backend.
                This prototype saves preferences for future integration.
              </span>
            </div>
          </Section>

          {/* ── 3. Display ─────────────────────────────────────────────────── */}
          <Section
            id="display"
            icon="◧"
            title="Display"
            sub="Column visibility and layout density on the shortlist dashboard"
            dirty={displayDirty}
          >
            {/* Column visibility */}
            <div>
              <div style={{ ...S.fieldLabel, marginBottom: 10 }}>
                VISIBLE COLUMNS
                <span style={S.fieldBadge}>
                  {[showAdjScore, showAttnShock, showPosts].filter(Boolean).length} / 3 shown
                </span>
              </div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                <ColChip
                  label="ADJ SCORE"
                  visible={showAdjScore}
                  onChange={set(setShowAdjScore, markDisplay)}
                />
                <ColChip
                  label="ATTN SHOCK"
                  visible={showAttnShock}
                  onChange={set(setShowAttnShock, markDisplay)}
                />
                <ColChip
                  label="POSTS"
                  visible={showPosts}
                  onChange={set(setShowPosts, markDisplay)}
                />
              </div>
              <div style={{ marginTop: 10, fontSize: 11, color: 'var(--text-3)' }}>
                Click a column chip to toggle it. TICKER, CREDIBILITY, and SIGNAL are always visible.
              </div>
            </div>

            <Divider />

            {/* Density */}
            <div>
              <div style={{ ...S.fieldLabel, marginBottom: 12 }}>ROW DENSITY</div>
              <DensityPicker value={density} onChange={set(setDensity, markDisplay)} />
            </div>

            {/* Live preview strip */}
            <div style={{
              borderRadius: 8, overflow: 'hidden',
              border: '1px solid var(--border)',
            }}>
              <div style={{
                padding: '8px 14px',
                background: 'var(--bg-card-alt)',
                fontSize: 10, fontWeight: 700,
                letterSpacing: '0.1em', color: 'var(--text-3)',
                fontFamily: 'var(--mono)',
                borderBottom: '1px solid var(--border)',
              }}>
                PREVIEW
              </div>
              {/* Header row */}
              <div style={{
                display: 'grid',
                gridTemplateColumns: `40px 80px 110px ${showAdjScore ? '100px' : ''} ${showAttnShock ? '110px' : ''} ${showPosts ? '60px' : ''} 1fr`,
                padding: '6px 14px', gap: 8,
                borderBottom: '1px solid var(--border)',
                background: 'var(--bg-card)',
              }}>
                {['RANK','TICKER','CRED', showAdjScore && 'ADJ SCORE', showAttnShock && 'ATTN σ', showPosts && 'POSTS', 'SIGNAL'].filter(Boolean).map(h => (
                  <div key={h} style={{ fontSize: 9, fontFamily: 'var(--mono)', fontWeight: 700, letterSpacing: '0.1em', color: 'var(--text-3)' }}>{h}</div>
                ))}
              </div>
              {/* Sample rows */}
              {[
                { rank: 1, ticker: 'NFLX', cred: 'high',   score: '0.0026', attn: '+2.41σ', posts: 28, signal: 'Unusual spike with high agreement…' },
                { rank: 2, ticker: 'TSLA', cred: 'high',   score: '0.0024', attn: '+1.84σ', posts: 31, signal: 'Strong positive sentiment…' },
                { rank: 3, ticker: 'NVDA', cred: 'medium', score: '0.0022', attn: '+3.17σ', posts: 47, signal: 'Volume spike detected…' },
              ].map((r, i) => (
                <div key={r.rank} style={{
                  display: 'grid',
                  gridTemplateColumns: `40px 80px 110px ${showAdjScore ? '100px' : ''} ${showAttnShock ? '110px' : ''} ${showPosts ? '60px' : ''} 1fr`,
                  padding: density === 'compact' ? '6px 14px' : '11px 14px',
                  gap: 8, alignItems: 'center',
                  background: i % 2 === 1 ? 'rgba(255,255,255,0.015)' : 'transparent',
                  transition: 'padding 0.25s ease',
                }}>
                  <span style={{ fontFamily: 'var(--mono)', fontSize: 12, fontWeight: 700, color: i === 0 ? 'var(--accent)' : 'var(--text-3)' }}>{r.rank}</span>
                  <span style={{ fontFamily: 'var(--mono)', fontSize: 13, fontWeight: 700, color: 'var(--text-1)' }}>{r.ticker}</span>
                  <span style={{
                    fontSize: 11, fontFamily: 'var(--mono)', fontWeight: 700,
                    color: r.cred === 'high' ? 'var(--green)' : 'var(--amber)',
                    padding: '2px 6px', borderRadius: 3,
                    background: r.cred === 'high' ? 'rgba(16,185,129,0.1)' : 'rgba(245,158,11,0.1)',
                  }}>{r.cred.toUpperCase()}</span>
                  {showAdjScore  && <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--accent)' }}>{r.score}</span>}
                  {showAttnShock && <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--amber)' }}>{r.attn}</span>}
                  {showPosts     && <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--text-2)' }}>{r.posts}</span>}
                  <span style={{ fontSize: 11, color: 'var(--text-3)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.signal}</span>
                </div>
              ))}
            </div>
          </Section>

          {/* ── 4. About ───────────────────────────────────────────────────── */}
          <Section
            id="about"
            icon="◈"
            title="About Meridian"
          >
            {/* Version */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
              <div style={{
                width: 48, height: 48, borderRadius: 12,
                background: 'rgba(0,212,184,0.08)',
                border: '1px solid rgba(0,212,184,0.25)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 22, flexShrink: 0,
              }}>▸</div>
              <div>
                <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-1)', fontFamily: 'var(--mono)' }}>
                  Meridian
                  <span style={{
                    marginLeft: 8, fontSize: 11,
                    color: 'var(--accent)', fontWeight: 600,
                    padding: '1px 6px', border: '1px solid rgba(0,212,184,0.3)',
                    borderRadius: 3,
                  }}>v1.0-beta</span>
                </div>
                <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 3 }}>
                  Social Signal Intelligence Platform · Built on academic methodology
                </div>
              </div>
            </div>

            <Divider />

            {/* Data sources */}
            <div>
              <div style={{ ...S.fieldLabel, marginBottom: 10 }}>DATA SOURCES</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                <DataSourceCard
                  icon="◉"
                  name="Reddit API (PRAW)"
                  detail="Live post collection from 5 financial subreddits. Signal window: previous market close → current market close (16:00 ET). Even-time sampled at ≤500 posts/day."
                  url="reddit.com/dev/api"
                />
                <DataSourceCard
                  icon="◈"
                  name="FinBERT (ProsusAI)"
                  detail="Financial domain-specific BERT model for per-post sentiment scoring. Output: P(positive) − P(negative) ∈ [−1, +1]. Trimmed mean (10%) per ticker-day."
                  url="HuggingFace"
                />
                <DataSourceCard
                  icon="◧"
                  name="Loughran-McDonald Lexicon"
                  detail="Domain-validated financial sentiment word list used as a validation check against FinBERT. Positive/negative word ratios computed per post."
                  url="SRAF Notre Dame"
                />
                <DataSourceCard
                  icon="◫"
                  name="Yahoo Finance (yfinance)"
                  detail="Adjusted close prices for T+1 and T+5 return computation during model training and evaluation. Used only retrospectively — never for ranking."
                  url="evaluation only"
                />
              </div>
            </div>

            <Divider />

            {/* Methodology */}
            <div>
              <div style={{ ...S.fieldLabel, marginBottom: 10 }}>METHODOLOGY</div>
              <div style={{
                padding: '14px 16px', borderRadius: 8,
                background: 'var(--bg-card-alt)',
                border: '1px solid var(--border)',
                fontFamily: 'var(--mono)', fontSize: 12,
                color: 'var(--text-2)', lineHeight: 1.8,
              }}>
                <div style={{ marginBottom: 8 }}>
                  <span style={{ color: 'var(--accent)' }}>▸</span> Signal = FinBERT tone + attention shock + author breadth + tone agreement
                </div>
                <div style={{ marginBottom: 8 }}>
                  <span style={{ color: 'var(--accent)' }}>▸</span> Credibility = 1 − α·DupRate − β·Herfindahl − γ·Burstiness
                </div>
                <div style={{ marginBottom: 8 }}>
                  <span style={{ color: 'var(--accent)' }}>▸</span> AdjScore = ŷ_{'{'}i,t+1{'}'} × Credibility_{'{'}i,t{'}'}
                </div>
                <div style={{ marginBottom: 8 }}>
                  <span style={{ color: 'var(--accent)' }}>▸</span> Model: LightGBM · trained on 65% of trading days
                </div>
                <div>
                  <span style={{ color: 'var(--accent)' }}>▸</span> Evaluated at T+1 and T+5 horizons vs. momentum and attention-only baselines
                </div>
              </div>
            </div>

            <Divider />

            {/* Disclaimer */}
            <div style={{
              padding: '16px 18px',
              background: 'rgba(239,68,68,0.05)',
              border: '1px solid rgba(239,68,68,0.2)',
              borderRadius: 8,
            }}>
              <div style={{
                fontSize: 11, fontWeight: 700, letterSpacing: '0.1em',
                color: 'var(--red)', fontFamily: 'var(--mono)', marginBottom: 8,
              }}>
                ! LEGAL DISCLAIMER
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-3)', lineHeight: 1.7 }}>
                Meridian is a <strong style={{ color: 'var(--text-2)' }}>research prototype</strong> for
                monitoring social media sentiment signals only.
                It does <strong style={{ color: 'var(--text-2)' }}>not</strong> constitute financial advice,
                investment recommendations, or solicitation to buy or sell any security.
                Past signal performance does not predict future returns.
                Always conduct independent due diligence and consult a qualified financial adviser
                before making investment decisions.
              </div>
            </div>
          </Section>

        </div>
      </div>

      <SaveToast state={toastState} />
    </div>
  );
}

const S = {
  page: {
    padding: '24px 28px',
    maxWidth: 1100,
    margin: '0 auto',
    display: 'flex',
    flexDirection: 'column',
    gap: 20,
  },
  topBar: {
    display: 'flex',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    gap: 16,
  },
  pageTitle: {
    fontSize: 20, fontWeight: 700,
    color: 'var(--text-1)', letterSpacing: '-0.01em',
  },
  pageSub: {
    fontSize: 12, color: 'var(--text-3)', marginTop: 4,
  },
  unsavedBadge: {
    fontSize: 10, fontWeight: 700, letterSpacing: '0.1em',
    color: 'var(--amber)', fontFamily: 'var(--mono)',
    padding: '4px 10px', borderRadius: 4,
    background: 'rgba(245,158,11,0.1)',
    border: '1px solid rgba(245,158,11,0.3)',
  },
  body: {
    display: 'flex',
    gap: 24,
    alignItems: 'flex-start',
  },
  sections: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    gap: 16,
  },
  fieldLabel: {
    fontSize: 10, fontWeight: 700, letterSpacing: '0.12em',
    color: 'var(--text-3)', fontFamily: 'var(--mono)',
    display: 'flex', alignItems: 'center', gap: 10,
  },
  fieldBadge: {
    fontSize: 10, color: 'var(--accent)', fontWeight: 600,
    padding: '1px 7px', borderRadius: 3,
    background: 'rgba(0,212,184,0.08)',
    border: '1px solid rgba(0,212,184,0.2)',
    letterSpacing: '0.04em',
  },
  subGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
    gap: 8, marginTop: 10,
  },
  inlineWarn: {
    marginTop: 8, padding: '8px 12px', borderRadius: 6,
    background: 'rgba(239,68,68,0.08)',
    border: '1px solid rgba(239,68,68,0.25)',
    fontSize: 11, color: 'var(--red)', fontFamily: 'var(--mono)',
  },
  infoNote: {
    display: 'flex', alignItems: 'flex-start', gap: 10,
    padding: '12px 14px', borderRadius: 8,
    background: 'rgba(59,130,246,0.06)',
    border: '1px solid rgba(59,130,246,0.2)',
    fontSize: 11, color: 'var(--text-3)', lineHeight: 1.6,
  },
};
