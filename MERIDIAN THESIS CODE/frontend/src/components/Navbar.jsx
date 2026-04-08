import { useNavigate, useLocation } from 'react-router-dom';
import { useState, useEffect } from 'react';

const NAV = [
  { path: '/',         label: 'Dashboard' },
  { path: '/settings', label: 'Settings'  },
];

export default function Navbar({ onLogout }) {
  const navigate = useNavigate();
  const location = useLocation();
  const [time, setTime]   = useState(new Date());
  const [server, setServer] = useState('connecting');

  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000);
    fetch('http://localhost:8000/health')
      .then(() => setServer('live'))
      .catch(() => setServer('offline'));
    return () => clearInterval(t);
  }, []);

  const fmt = (d) =>
    d.toLocaleTimeString('en-US', { hour12: false }) + '  ET';

  return (
    <nav style={S.nav}>
      {/* Logo */}
      <div style={S.logo} onClick={() => navigate('/')}>
        <span style={S.logoMark}>▸</span>
        <span style={S.logoText}>Meridian</span>
        <span style={S.logoTag}>BETA</span>
      </div>

      {/* Nav links */}
      <div style={S.links}>
        {NAV.map(({ path, label }) => (
          <button
            key={path}
            onClick={() => navigate(path)}
            style={{
              ...S.link,
              ...(location.pathname === path ? S.linkActive : {}),
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Status bar */}
      <div style={S.right}>
        <div style={S.pill}>
          <span style={{
            ...S.dot,
            background: server === 'live' ? 'var(--green)' : server === 'offline' ? 'var(--red)' : 'var(--amber)',
          }} />
          <span style={S.pillText}>{server.toUpperCase()}</span>
        </div>
        <span style={S.clock} className="mono">{fmt(time)}</span>
        <span style={S.divider} />
        <span style={S.signalDate} className="mono">2024-11-11</span>
        <span style={S.divider} />
        <button style={S.logout} onClick={onLogout}>LOG OUT</button>
      </div>
    </nav>
  );
}

const S = {
  nav: {
    display: 'flex', alignItems: 'center', gap: 32,
    height: 52, padding: '0 24px',
    background: '#070e1c',
    borderBottom: '1px solid var(--border)',
    position: 'sticky', top: 0, zIndex: 100,
    flexShrink: 0,
  },
  logo: {
    display: 'flex', alignItems: 'center', gap: 8,
    cursor: 'pointer', userSelect: 'none',
  },
  logoMark: {
    color: 'var(--accent)', fontSize: 20, lineHeight: 1,
  },
  logoText: {
    fontFamily: 'var(--mono)', fontWeight: 600, fontSize: 15,
    color: 'var(--text-1)', letterSpacing: '0.04em',
  },
  logoTag: {
    fontSize: 9, fontWeight: 700, letterSpacing: '0.12em',
    color: 'var(--accent)', background: 'var(--accent-glow)',
    border: '1px solid var(--accent-dim)', borderRadius: 3,
    padding: '1px 5px', marginLeft: 4,
  },
  links: { display: 'flex', gap: 4 },
  link: {
    background: 'none', border: 'none', color: 'var(--text-3)',
    fontSize: 13, fontWeight: 500, padding: '6px 14px',
    borderRadius: 'var(--r-sm)', transition: 'color 0.15s',
    letterSpacing: '0.02em',
  },
  linkActive: {
    color: 'var(--accent)', background: 'rgba(0,212,184,0.08)',
  },
  right: {
    marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 16,
  },
  pill: {
    display: 'flex', alignItems: 'center', gap: 6,
    padding: '3px 10px', borderRadius: 20,
    border: '1px solid var(--border)', background: 'var(--bg-card)',
  },
  dot: { width: 6, height: 6, borderRadius: '50%' },
  pillText: {
    fontFamily: 'var(--mono)', fontSize: 10, fontWeight: 600,
    color: 'var(--text-2)', letterSpacing: '0.1em',
  },
  clock: { fontSize: 12, color: 'var(--text-3)' },
  signalDate: { fontSize: 12, color: 'var(--text-2)' },
  divider: {
    width: 1, height: 16, background: 'var(--border-b)',
  },
  logout: {
    background: 'none', border: '1px solid var(--border)',
    color: 'var(--text-3)', fontSize: 10, fontWeight: 700,
    letterSpacing: '0.1em', padding: '4px 10px',
    borderRadius: 'var(--r-sm)', transition: 'all 0.15s',
  },
};
