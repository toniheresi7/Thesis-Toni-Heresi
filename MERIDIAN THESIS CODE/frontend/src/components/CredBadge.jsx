const CONFIG = {
  high:   { color: '#10b981', bg: 'rgba(16,185,129,0.12)', border: 'rgba(16,185,129,0.3)', label: 'HIGH' },
  medium: { color: '#f59e0b', bg: 'rgba(245,158,11,0.12)', border: 'rgba(245,158,11,0.3)', label: 'MED'  },
  low:    { color: '#ef4444', bg: 'rgba(239,68,68,0.12)',  border: 'rgba(239,68,68,0.3)',  label: 'LOW'  },
};

export default function CredBadge({ label, score, size = 'md' }) {
  const c = CONFIG[label] || CONFIG.low;
  const small = size === 'sm';
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: small ? 4 : 6,
      padding: small ? '2px 7px' : '3px 10px',
      borderRadius: 4,
      background: c.bg, border: `1px solid ${c.border}`,
      fontFamily: 'var(--mono)', fontWeight: 700,
      fontSize: small ? 10 : 11, letterSpacing: '0.08em',
      color: c.color, whiteSpace: 'nowrap',
    }}>
      <span style={{
        width: small ? 5 : 6, height: small ? 5 : 6,
        borderRadius: '50%', background: c.color,
        flexShrink: 0,
      }} />
      {c.label}
      {score !== undefined && !small && (
        <span style={{ opacity: 0.7, marginLeft: 2 }}>
          {(score * 100).toFixed(0)}%
        </span>
      )}
    </span>
  );
}
