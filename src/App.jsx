import React, { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import BackgroundGlow from './components/BackgroundGlow';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import LandingPage from './pages/LandingPage';
import DashboardPage from './pages/DashboardPage';
import HowItWorksPage from './pages/HowItWorksPage';
import AboutPage from './pages/AboutPage';
import NetworkDirectoryPage from './pages/NetworkDirectoryPage';
import SecurityLabPage from './pages/SecurityLabPage';
import AuthPage from './pages/AuthPage';
import { AuthProvider, useAuth } from './context/AuthContext';
import './App.css';

// Scroll to top on page transition
function ScrollToTop() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);
  return null;
}

function AppContent() {
  const { isAuthenticated } = useAuth();

  // If not authenticated, render secure authentication barrier
  if (!isAuthenticated) {
    return <AuthPage />;
  }

  return (
    <Router>
      <ScrollToTop />
      <div className="relative min-h-screen bg-[#080711] text-slate-100 selection:bg-cyan-500 selection:text-black">
        {/* Background Ambient Glow FX */}
        <BackgroundGlow />

        {/* Sticky Glass Navbar */}
        <Navbar />

        {/* Main Content Viewport */}
        <main className="relative z-10 min-h-[calc(100vh-160px)]">
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/transparency" element={<DashboardPage />} />
            <Route path="/network-directory" element={<NetworkDirectoryPage />} />
            <Route path="/security-lab" element={<SecurityLabPage />} />
            <Route path="/how-it-works" element={<HowItWorksPage />} />
            <Route path="/about" element={<AboutPage />} />
          </Routes>
        </main>

        {/* Footer */}
        <Footer />
      </div>
    </Router>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}
