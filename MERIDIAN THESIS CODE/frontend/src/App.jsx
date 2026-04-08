import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { useState } from 'react';
import LoginPage     from './pages/LoginPage';
import DashboardPage from './pages/DashboardPage';
import StockDetail   from './pages/StockDetailPage';
import SettingsPage  from './pages/SettingsPage';
import Navbar        from './components/Navbar';

export default function App() {
  const [auth, setAuth] = useState(false);

  if (!auth) return <LoginPage onLogin={() => setAuth(true)} />;

  return (
    <BrowserRouter>
      <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
        <Navbar onLogout={() => setAuth(false)} />
        <div style={{ flex: 1, overflow: 'auto' }}>
          <Routes>
            <Route path="/"              element={<DashboardPage />} />
            <Route path="/stock/:ticker" element={<StockDetail />} />
            <Route path="/settings"      element={<SettingsPage />} />
            <Route path="*"              element={<Navigate to="/" />} />
          </Routes>
        </div>
      </div>
    </BrowserRouter>
  );
}
