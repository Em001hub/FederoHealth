import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
  ShieldCheck, Activity, Layers, HelpCircle, ArrowUpRight,
  Bell, Key, Globe, Building2, User, ChevronDown, LogOut, ShieldAlert
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import AuthModal from './AuthModal';
import NotificationsModal from './NotificationsModal';

export default function Navbar() {
  const location = useLocation();
  const { currentHospital, currentUser, notifications, isSuperAdminMode, signOut } = useAuth();
  
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [showNotifications, setShowNotifications] = useState(false);

  const unreadCount = notifications.filter(n => !n.read).length;

  const navLinks = [
    { name: 'Overview', path: '/', icon: Activity },
    { name: 'Live Dashboard', path: '/dashboard', icon: Layers, badge: 'Live AI' },
    { name: 'Network Directory', path: '/network-directory', icon: Globe },
    { name: 'Security Lab', path: '/security-lab', icon: ShieldAlert, badge: 'Live' },
    { name: 'How It Works', path: '/how-it-works', icon: HelpCircle },
  ];

  return (
    <>
      <header className="sticky top-0 z-50 w-full glass-panel border-b border-white/10 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-20 flex items-center justify-between">
          
          {/* Brand Logo */}
          <Link to="/" className="flex items-center gap-3 group">
            <div className="relative w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-400 via-purple-600 to-indigo-600 p-[1px] shadow-lg shadow-cyan-500/20 group-hover:shadow-cyan-500/40 transition-all duration-300">
              <div className="w-full h-full bg-slate-950/90 rounded-[11px] flex items-center justify-center">
                <ShieldCheck className="w-6 h-6 text-cyan-400 group-hover:scale-110 transition-transform duration-300" />
              </div>
            </div>
            <div className="flex flex-col">
              <span className="font-bold text-xl tracking-tight text-white flex items-center gap-1.5 font-heading">
                Federo<span className="text-cyan-400">.Health</span>
              </span>
              <span className="text-[10px] tracking-widest text-slate-400 uppercase font-mono">
                Verified Clinical AI Federation
              </span>
            </div>
          </Link>

          {/* Navigation Links */}
          <nav className="hidden md:flex items-center gap-1.5 p-1.5 rounded-full bg-slate-900/60 border border-white/10 shadow-inner">
            {navLinks.map((link) => {
              const isActive = location.pathname === link.path;
              const Icon = link.icon;
              return (
                <Link
                  key={link.path}
                  to={link.path}
                  className={`relative px-4 py-2 rounded-full text-xs font-medium transition-all duration-300 flex items-center gap-2 ${
                    isActive
                      ? 'text-white bg-gradient-to-r from-cyan-500/20 to-purple-500/20 border border-cyan-400/40 shadow-sm shadow-cyan-500/20'
                      : 'text-slate-300 hover:text-white hover:bg-white/5'
                  }`}
                >
                  <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-cyan-400' : 'text-slate-400'}`} />
                  <span>{link.name}</span>
                  {link.badge && (
                    <span className="ml-1 px-1.5 py-0.5 text-[9px] font-semibold tracking-wider rounded-full bg-cyan-400/10 text-cyan-300 border border-cyan-400/20 uppercase">
                      {link.badge}
                    </span>
                  )}
                </Link>
              );
            })}
          </nav>

          {/* Right Action Bar: Notifications & Institutional Auth Chip */}
          <div className="flex items-center gap-3">
            
            {/* Notification Bell */}
            <button
              onClick={() => setShowNotifications(true)}
              className="relative p-2.5 rounded-xl bg-slate-900/80 hover:bg-slate-800 text-slate-300 hover:text-white border border-white/10 transition-colors"
              title="Platform Notifications"
            >
              <Bell className="w-4 h-4" />
              {unreadCount > 0 && (
                <span className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-cyan-500 text-black text-[9px] font-bold flex items-center justify-center shadow-lg shadow-cyan-500/40">
                  {unreadCount}
                </span>
              )}
            </button>

            {/* Institutional Identity Chip */}
            <button
              onClick={() => setShowAuthModal(true)}
              className="flex items-center gap-2.5 p-1.5 px-3 rounded-xl bg-slate-900/90 hover:bg-slate-850 border border-white/10 hover:border-cyan-400/40 text-left transition-all group"
              title="Switch Hospital Context / Identity"
            >
              <div className="w-7 h-7 rounded-lg bg-cyan-500/10 border border-cyan-400/30 flex items-center justify-center text-cyan-400">
                <Building2 className="w-4 h-4" />
              </div>
              <div className="hidden lg:flex flex-col">
                <span className="text-xs font-bold text-white font-heading truncate max-w-[130px]">
                  {isSuperAdminMode ? 'Super Admin' : currentHospital?.name}
                </span>
                <span className="text-[9px] text-cyan-400 font-mono">
                  {isSuperAdminMode ? 'PLATFORM GOVERNANCE' : currentUser?.role?.replace('_', ' ').toUpperCase()}
                </span>
              </div>
              <ChevronDown className="w-3.5 h-3.5 text-slate-400 group-hover:text-cyan-400 transition-colors" />
            </button>

            {/* Secure Sign Out Button */}
            <button
              onClick={signOut}
              className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-slate-900 hover:bg-rose-950/40 text-slate-400 hover:text-rose-300 border border-white/10 hover:border-rose-400/30 text-xs font-semibold transition-all shadow-sm group"
              title="Lock Session & Sign Out"
            >
              <LogOut className="w-3.5 h-3.5 group-hover:text-rose-400 transition-colors" />
              <span className="hidden sm:inline">Sign Out</span>
            </button>
          </div>

        </div>

        {/* Mobile Navigation Bar */}
        <div className="md:hidden flex items-center justify-around py-2.5 px-2 bg-slate-950/90 border-t border-white/10 text-xs">
          {navLinks.map((link) => {
            const isActive = location.pathname === link.path;
            const Icon = link.icon;
            return (
              <Link
                key={link.path}
                to={link.path}
                className={`flex flex-col items-center gap-1 px-2 py-1 rounded-lg ${
                  isActive ? 'text-cyan-400 font-semibold' : 'text-slate-400'
                }`}
              >
                <Icon className="w-4 h-4" />
                <span className="text-[10px]">{link.name}</span>
              </Link>
            );
          })}
        </div>
      </header>

      {/* Modals */}
      <AuthModal
        isOpen={showAuthModal}
        onClose={() => setShowAuthModal(false)}
      />

      <NotificationsModal
        isOpen={showNotifications}
        onClose={() => setShowNotifications(false)}
      />
    </>
  );
}
