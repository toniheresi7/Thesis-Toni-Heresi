import { useState } from 'react';

const SCAN_LINES = Array.from({ length: 18 }, (_, i) => i);

export default function LoginPage({ onLogin }) {
  const [user, setUser]   = useState('');
  const [pass, setPass]   = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  function handleSubmit(e) {
    e.preventDefault();
    if (!user || !pass) { setError('CREDENTIALS REQUIRED'); return; }
    setLoading(true);
    setError('');
    setTimeout(() => {
      setLoading(false);
      onLogin();
    }, 900);
  }

  return (
    <div style={S.root}>
      {/* Animated grid background */}
      <div style={S.grid} aria-hidden />

      {/* Scan lines */}
      <div style={S.scanLines} aria-hidden>
        {SCAN_LINES.map(i => <div key={i} style={S.scanLine} />)}
      </div>

      <div style={S.center}>
        {/* Logo block */}
        <div style={S.logoBlock}>
          <div style={S.logoIcon}>
            <span style={S.iconGlyph}>▸</span>
          </div>
          <div>
            <div style={S.logoName}>Meridian</div>
            <div style={S.logoSub}>SOCIAL SIGNAL INTELLIGENCE PLATFORM</div>
          </div>
        </div>

        {/* Divider */}
        <div style={S.divider}>
          <span style={S.dividerLine} />
          <span style={S.dividerText}>SECURE ACCESS</span>
          <span style={S.dividerLine} />
        </div>

        {/* Login form */}
        <form onSubmit={handleSubmit} style={S.form} autoComplete="off">
          <div style={S.field}>
            <label style={S.label}>USER ID</label>
            <input
              style={S.input}
              value={user}
              onChange={e => setUser(e.target.value)}
              placeholder="analyst@institution.com"
              spellCheck={false}
            />
          </div>

          <div style={S.field}>
            <label style={S.label}>ACCESS KEY</label>
            <input
              style={S.input}
              type="password"
              value={pass}
              onChange={e => setPass(e.target.value)}
              placeholder="••••••••••••"
            />
          </div>

          {error && <div style={S.error}>{error}</div>}

          <button type="submit" style={S.submit} disabled={loading}>
            {loading
              ? <><span style={S.spinner} />AUTHENTICATING…</>
              : 'ENTER PLATFORM →'}
          </button>
        </form>

        {/* Demo hint */}
        <p style={S.hint}>
          Demo mode — enter any credentials to continue
        </p>

        {/* Footer stat strip */}
        <div style={S.stats}>
          {[
            { label: 'SIGNAL DATE', value: '2024-11-11' },
            { label: 'TICKERS',     value: '10 / 500'   },
            { label: 'MODEL',       value: 'LightGBM' },
            { label: 'STATUS',      value: 'LIVE ●'     },
          ].map(({ label, value }) => (
            <div key={label} style={S.stat}>
              <div style={S.statLabel}>{label}</div>
              <div style={S.statValue}>{value}</div>
            </div>
          ))}
        </div>
      </div>

      <div style={S.disclaimer}>
        FOR MONITORING PURPOSES ONLY — NOT FINANCIAL ADVICE
      </div>

      <style>{`
        @keyframes pulse-glow {
          0%, 100% { opacity: 0.6; }
          50% { opacity: 1; }
        }
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}

const S = {
  root: {
    position: 'relative', height: '100vh', width: '100%',
    display: 'flex', flexDirection: 'column',
    alignItems: 'center', justifyContent: 'center',
    background: 'radial-gradient(ellipse at 50% 40%, #091828 0%, #050810 60%)',
    overflow: 'hidden',
  },
  grid: {
    position: 'absolute', inset: 0,
    backgroundImage: `
      linear-gradient(rgba(0,212,184,0.04) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,212,184,0.04) 1px, transparent 1px)
    `,
    backgroundSize: '40px 40px',
    zIndex: 0,
  },
  scanLines: {
    position: 'absolute', inset: 0, zIndex: 0,
    display: 'flex', flexDirection: 'column', justifyContent: 'space-around',
    opacity: 0.03, pointerEvents: 'none',
  },
  scanLine: {
    height: 1, width: '100%',
    background: 'linear-gradient(90deg, transparent, #00d4b8, transparent)',
  },
  center: {
    position: 'relative', zIndex: 1,
    width: 400, display: 'flex', flexDirection: 'column', gap: 28,
    padding: 40,
    background: 'rgba(11,17,32,0.92)',
    border: '1px solid rgba(0,212,184,0.2)',
    borderRadius: 16,
    boxShadow: '0 0 60px rgba(0,212,184,0.06), 0 20px 60px rgba(0,0,0,0.6)',
    backdropFilter: 'blur(8px)',
  },
  logoBlock: {
    display: 'flex', alignItems: 'center', gap: 14,
  },
  logoIcon: {
    width: 44, height: 44, borderRadius: 10,
    background: 'rgba(0,212,184,0.1)',
    border: '1px solid rgba(0,212,184,0.3)',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    flexShrink: 0,
  },
  iconGlyph: {
    color: '#00d4b8', fontSize: 22, lineHeight: 1,
    animation: 'pulse-glow 2.4s ease-in-out infinite',
    display: 'block',
  },
  logoName: {
    fontFamily: 'var(--mono)', fontSize: 22, fontWeight: 600,
    color: '#e2e8f4', letterSpacing: '0.04em',
  },
  logoSub: {
    fontSize: 9, fontWeight: 700, letterSpacing: '0.15em',
    color: 'var(--text-3)', marginTop: 2,
  },
  divider: {
    display: 'flex', alignItems: 'center', gap: 12,
  },
  dividerLine: {
    flex: 1, height: 1,
    background: 'linear-gradient(90deg, transparent, var(--border-b))',
  },
  dividerText: {
    fontSize: 10, fontWeight: 700, letterSpacing: '0.15em',
    color: 'var(--text-3)',
  },
  form: { display: 'flex', flexDirection: 'column', gap: 16 },
  field: { display: 'flex', flexDirection: 'column', gap: 6 },
  label: {
    fontSize: 10, fontWeight: 700, letterSpacing: '0.12em',
    color: 'var(--text-3)', fontFamily: 'var(--mono)',
  },
  input: {
    background: '#07101e', border: '1px solid var(--border)',
    borderRadius: 6, padding: '10px 14px',
    color: 'var(--text-1)', fontSize: 13,
    fontFamily: 'var(--mono)', outline: 'none',
    transition: 'border-color 0.15s',
  },
  error: {
    padding: '8px 12px', borderRadius: 6,
    background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)',
    color: '#ef4444', fontSize: 11, fontWeight: 700,
    letterSpacing: '0.08em', fontFamily: 'var(--mono)',
  },
  submit: {
    width: '100%', padding: '12px',
    background: 'var(--accent)', color: '#000',
    border: 'none', borderRadius: 6,
    fontFamily: 'var(--mono)', fontSize: 13, fontWeight: 700,
    letterSpacing: '0.06em', cursor: 'pointer',
    transition: 'all 0.15s',
    display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
  },
  spinner: {
    width: 12, height: 12, borderRadius: '50%',
    border: '2px solid rgba(0,0,0,0.3)',
    borderTopColor: '#000',
    animation: 'spin 0.8s linear infinite',
    display: 'inline-block',
  },
  hint: {
    textAlign: 'center', fontSize: 11,
    color: 'var(--text-3)', letterSpacing: '0.02em',
  },
  stats: {
    display: 'grid', gridTemplateColumns: '1fr 1fr',
    gap: '1px',
    background: 'var(--border)',
    borderRadius: 8, overflow: 'hidden',
    border: '1px solid var(--border)',
  },
  stat: {
    padding: '10px 14px',
    background: 'rgba(7,16,30,0.8)',
  },
  statLabel: {
    fontSize: 9, fontWeight: 700, letterSpacing: '0.12em',
    color: 'var(--text-3)', fontFamily: 'var(--mono)',
  },
  statValue: {
    fontSize: 12, fontWeight: 600,
    color: 'var(--accent)', fontFamily: 'var(--mono)',
    marginTop: 2,
  },
  disclaimer: {
    position: 'absolute', bottom: 20, left: 0, right: 0,
    textAlign: 'center',
    fontSize: 9, letterSpacing: '0.12em',
    color: 'var(--text-3)', fontFamily: 'var(--mono)',
    zIndex: 1,
  },
};
