import React, { useState, useRef, useEffect } from 'react';
import { Camera, Upload, RefreshCw, AlertCircle, Activity, X, Zap, ChevronRight, Circle, ArrowRight, Shield, Cpu, BarChart2, Eye, Check } from 'lucide-react';

const AGENT_URL = "https://ahtisham992-ergonomic-posture-agent.hf.space/ergonomic-posture-agent";
const HEALTH_URL = "https://ahtisham992-ergonomic-posture-agent.hf.space/health";

/* ─────────────────────────────────────────────────────────────────────────
   GLOBAL CSS
───────────────────────────────────────────────────────────────────────── */
const GLOBAL_CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:        #070a10;
    --surface:   #0d1117;
    --card:      #111520;
    --card2:     #161b28;
    --border:    rgba(255,255,255,0.06);
    --border-hi: rgba(255,255,255,0.10);
    --accent:    #00e5b4;
    --accent2:   #0066ff;
    --text:      #e8eaf0;
    --muted:     #6b7280;
    --warn:      #f59e0b;
    --danger:    #ef4444;
    --good:      #10b981;
    --font-head: 'Syne', sans-serif;
    --font-body: 'DM Sans', sans-serif;
    --font-mono: 'DM Mono', monospace;
  }

  html, body, #root {
    height: 100%;
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-body);
    font-size: 15px;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
  }

  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 99px; }

  /* ── PAGE TRANSITIONS ── */
  .page-enter {
    animation: fadeUp 0.45s cubic-bezier(0.22, 1, 0.36, 1) both;
  }
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  /* ═══════════════════════════════════════════
     LANDING PAGE
  ═══════════════════════════════════════════ */

  .land-root {
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* Noise overlay */
  .land-root::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.035'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
  }

  /* ── NAV ── */
  .land-nav {
    position: fixed;
    top: 0; left: 0; right: 0;
    z-index: 200;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 48px;
    height: 64px;
    border-bottom: 1px solid var(--border);
    background: rgba(7,10,16,0.85);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
  }
  .land-nav-logo {
    font-family: var(--font-head);
    font-size: 17px;
    font-weight: 700;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 10px;
    letter-spacing: -0.3px;
  }
  .logo-chip {
    width: 30px; height: 30px;
    background: var(--accent);
    border-radius: 7px;
    display: flex; align-items: center; justify-content: center;
  }
  .logo-chip svg { color: #000; }
  .land-nav-links {
    display: flex;
    align-items: center;
    gap: 32px;
  }
  .land-nav-links a {
    font-size: 13px;
    color: var(--muted);
    text-decoration: none;
    transition: color 0.15s;
    cursor: pointer;
  }
  .land-nav-links a:hover { color: var(--text); }
  .nav-cta {
    display: flex;
    align-items: center;
    gap: 7px;
    background: var(--accent);
    color: #000 !important;
    font-weight: 600;
    font-size: 13px !important;
    padding: 8px 18px;
    border-radius: 8px;
    cursor: pointer;
    border: none;
    transition: filter 0.15s, transform 0.15s;
  }
  .nav-cta:hover { filter: brightness(1.1); transform: translateY(-1px); }

  /* ── HERO ── */
  .land-hero {
    position: relative;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 120px 24px 80px;
    overflow: hidden;
  }

  /* Radial glow behind hero */
  .hero-glow {
    position: absolute;
    top: 15%;
    left: 50%;
    transform: translateX(-50%);
    width: 800px;
    height: 600px;
    background: radial-gradient(ellipse at center, rgba(0,229,180,0.07) 0%, transparent 70%);
    pointer-events: none;
  }
  .hero-glow2 {
    position: absolute;
    top: 30%;
    left: 30%;
    width: 400px;
    height: 400px;
    background: radial-gradient(ellipse at center, rgba(0,102,255,0.05) 0%, transparent 70%);
    pointer-events: none;
  }

  /* Grid lines */
  .hero-grid {
    position: absolute;
    inset: 0;
    background-image:
      linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
    background-size: 60px 60px;
    mask-image: radial-gradient(ellipse 80% 60% at 50% 0%, black 20%, transparent 100%);
    pointer-events: none;
  }

  .hero-content { position: relative; z-index: 2; max-width: 800px; }

  .hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-family: var(--font-mono);
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent);
    border: 1px solid rgba(0,229,180,0.25);
    background: rgba(0,229,180,0.05);
    padding: 6px 14px;
    border-radius: 99px;
    margin-bottom: 28px;
    animation: fadeUp 0.6s 0.1s both;
  }
  .badge-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent);
    animation: badgePulse 2s ease-in-out infinite;
  }
  @keyframes badgePulse {
    0%,100% { opacity: 1; } 50% { opacity: 0.4; }
  }

  .hero-title {
    font-family: var(--font-head);
    font-size: clamp(42px, 6vw, 80px);
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -2.5px;
    color: var(--text);
    margin-bottom: 24px;
    animation: fadeUp 0.6s 0.2s both;
  }
  .hero-title-accent { color: var(--accent); }

  .hero-sub {
    font-size: clamp(15px, 2vw, 18px);
    color: var(--muted);
    line-height: 1.7;
    max-width: 560px;
    margin: 0 auto 44px;
    animation: fadeUp 0.6s 0.3s both;
    font-weight: 300;
  }

  .hero-actions {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 14px;
    animation: fadeUp 0.6s 0.4s both;
    flex-wrap: wrap;
  }
  .btn-hero-primary {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    background: var(--accent);
    color: #000;
    font-family: var(--font-body);
    font-size: 15px;
    font-weight: 600;
    padding: 14px 28px;
    border-radius: 10px;
    border: none;
    cursor: pointer;
    transition: filter 0.15s, transform 0.15s, box-shadow 0.15s;
    box-shadow: 0 0 0 0 rgba(0,229,180,0);
  }
  .btn-hero-primary:hover {
    filter: brightness(1.08);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,229,180,0.2);
  }
  .btn-hero-secondary {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    background: transparent;
    color: var(--text);
    font-family: var(--font-body);
    font-size: 15px;
    font-weight: 400;
    padding: 14px 24px;
    border-radius: 10px;
    border: 1px solid var(--border-hi);
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
  }
  .btn-hero-secondary:hover { background: rgba(255,255,255,0.04); border-color: rgba(255,255,255,0.2); }

  /* ── STATS ROW ── */
  .hero-stats {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 48px;
    margin-top: 72px;
    padding-top: 40px;
    border-top: 1px solid var(--border);
    animation: fadeUp 0.6s 0.5s both;
    flex-wrap: wrap;
  }
  .stat-item { text-align: center; }
  .stat-num {
    font-family: var(--font-head);
    font-size: 30px;
    font-weight: 700;
    letter-spacing: -1px;
    color: var(--text);
  }
  .stat-label {
    font-size: 12px;
    color: var(--muted);
    margin-top: 2px;
    font-family: var(--font-mono);
    letter-spacing: 0.05em;
  }
  .stat-divider {
    width: 1px;
    height: 36px;
    background: var(--border);
  }

  /* ── FEATURES ── */
  .land-features {
    padding: 100px 48px;
    max-width: 1200px;
    margin: 0 auto;
  }
  .section-eyebrow {
    font-family: var(--font-mono);
    font-size: 11px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 16px;
  }
  .section-title {
    font-family: var(--font-head);
    font-size: clamp(28px, 4vw, 44px);
    font-weight: 700;
    letter-spacing: -1.5px;
    line-height: 1.1;
    color: var(--text);
    margin-bottom: 16px;
  }
  .section-sub {
    font-size: 15px;
    color: var(--muted);
    max-width: 480px;
    line-height: 1.7;
    font-weight: 300;
  }
  .features-header { margin-bottom: 56px; }
  .features-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
  }
  .feature-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 28px;
    transition: border-color 0.2s, background 0.2s, transform 0.2s;
    cursor: default;
  }
  .feature-card:hover {
    border-color: var(--border-hi);
    background: var(--card2);
    transform: translateY(-3px);
  }
  .feature-icon-wrap {
    width: 44px; height: 44px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 20px;
  }
  .feature-title {
    font-family: var(--font-head);
    font-size: 16px;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 10px;
    letter-spacing: -0.3px;
  }
  .feature-desc {
    font-size: 13px;
    color: var(--muted);
    line-height: 1.7;
  }

  /* ── HOW IT WORKS ── */
  .land-how {
    padding: 80px 48px 100px;
    max-width: 1200px;
    margin: 0 auto;
  }
  .how-steps {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0;
    position: relative;
    margin-top: 56px;
  }
  .how-steps::before {
    content: '';
    position: absolute;
    top: 22px;
    left: 10%;
    right: 10%;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-hi) 20%, var(--border-hi) 80%, transparent);
  }
  .how-step { padding: 0 24px; text-align: center; }
  .step-num {
    width: 44px; height: 44px;
    border-radius: 50%;
    background: var(--card);
    border: 1px solid var(--border-hi);
    display: flex; align-items: center; justify-content: center;
    font-family: var(--font-mono);
    font-size: 13px;
    color: var(--accent);
    margin: 0 auto 20px;
    position: relative;
    z-index: 1;
  }
  .step-title {
    font-family: var(--font-head);
    font-size: 14px;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 8px;
  }
  .step-desc { font-size: 12px; color: var(--muted); line-height: 1.6; }

  /* ── CTA BANNER ── */
  .land-cta-banner {
    margin: 0 48px 100px;
    background: linear-gradient(135deg, rgba(0,229,180,0.06) 0%, rgba(0,102,255,0.06) 100%);
    border: 1px solid rgba(0,229,180,0.12);
    border-radius: 20px;
    padding: 64px 48px;
    text-align: center;
    position: relative;
    overflow: hidden;
  }
  .cta-banner-glow {
    position: absolute;
    top: -50%;
    left: 50%;
    transform: translateX(-50%);
    width: 500px; height: 300px;
    background: radial-gradient(ellipse, rgba(0,229,180,0.08) 0%, transparent 70%);
    pointer-events: none;
  }
  .cta-banner-title {
    font-family: var(--font-head);
    font-size: clamp(24px, 4vw, 40px);
    font-weight: 700;
    letter-spacing: -1.5px;
    color: var(--text);
    margin-bottom: 14px;
    position: relative;
  }
  .cta-banner-sub {
    font-size: 15px;
    color: var(--muted);
    margin-bottom: 36px;
    font-weight: 300;
    position: relative;
  }

  /* ── LAND FOOTER ── */
  .land-footer {
    border-top: 1px solid var(--border);
    padding: 28px 48px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--muted);
  }

  /* ══════════════════════════════════════════
     ANALYZER APP
  ══════════════════════════════════════════ */

  .epa-shell {
    min-height: 100vh;
    display: grid;
    grid-template-rows: auto 1fr auto;
    grid-template-columns: 240px 1fr;
    grid-template-areas:
      "topbar topbar"
      "sidebar main"
      "footer  footer";
  }

  .epa-topbar {
    grid-area: topbar;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 28px;
    height: 58px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
    position: sticky;
    top: 0;
    z-index: 100;
  }
  .epa-topbar-left {
    display: flex;
    align-items: center;
    gap: 20px;
  }
  .epa-back-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: var(--muted);
    background: none;
    border: 1px solid var(--border-hi);
    border-radius: 6px;
    padding: 5px 10px;
    cursor: pointer;
    font-family: var(--font-body);
    transition: all 0.15s;
  }
  .epa-back-btn:hover { color: var(--text); border-color: rgba(255,255,255,0.2); }
  .epa-logo {
    font-family: var(--font-head);
    font-size: 16px;
    font-weight: 700;
    letter-spacing: -0.3px;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 9px;
  }
  .epa-logo-icon {
    width: 26px; height: 26px;
    background: var(--accent);
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
  }
  .epa-logo-icon svg { color: #000; }
  .epa-version {
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--muted);
    border: 1px solid var(--border-hi);
    padding: 2px 7px;
    border-radius: 4px;
  }
  .epa-status-pill {
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 11px;
    font-family: var(--font-mono);
    color: var(--muted);
    border: 1px solid var(--border-hi);
    border-radius: 99px;
    padding: 5px 13px;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
    background: transparent;
  }
  .epa-status-pill:hover { border-color: var(--accent); background: rgba(0,229,180,0.04); }
  .epa-status-pill.online { color: var(--good); border-color: rgba(16,185,129,0.3); }
  .epa-status-pill.offline { color: var(--danger); border-color: rgba(239,68,68,0.3); }
  .dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--muted);
  }
  .dot.pulse {
    background: var(--good);
    animation: pulseGlow 1.8s ease-in-out infinite;
  }
  @keyframes pulseGlow {
    0%,100% { box-shadow: 0 0 0 0 rgba(16,185,129,0.5); }
    50%      { box-shadow: 0 0 0 5px transparent; }
  }

  .epa-sidebar {
    grid-area: sidebar;
    background: var(--surface);
    border-right: 1px solid var(--border);
    padding: 24px 0;
    display: flex;
    flex-direction: column;
    gap: 4px;
    position: sticky;
    top: 58px;
    height: calc(100vh - 58px);
    overflow-y: auto;
  }
  .epa-sidebar-section { padding: 0 14px; margin-bottom: 8px; }
  .epa-sidebar-label {
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    padding: 0 10px;
    margin-bottom: 6px;
  }
  .epa-nav-btn {
    width: 100%;
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 9px 10px;
    border: none;
    background: transparent;
    color: var(--muted);
    border-radius: 7px;
    font-family: var(--font-body);
    font-size: 13px;
    cursor: pointer;
    transition: background 0.15s, color 0.15s;
    text-align: left;
  }
  .epa-nav-btn:hover { background: rgba(255,255,255,0.04); color: var(--text); }
  .epa-nav-btn.active { background: rgba(0,229,180,0.07); color: var(--accent); }

  .epa-main {
    grid-area: main;
    padding: 32px 36px;
    overflow-y: auto;
  }
  .epa-page-title {
    font-family: var(--font-head);
    font-size: 24px;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin-bottom: 4px;
  }
  .epa-page-sub { font-size: 13px; color: var(--muted); margin-bottom: 28px; }

  .epa-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 22px;
  }
  .epa-card + .epa-card { margin-top: 14px; }
  .epa-card-title {
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 14px;
  }
  .epa-content-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 18px;
  }

  /* ── Drop zone ── */
  .drop-zone {
    border: 1.5px dashed var(--border-hi);
    border-radius: 10px;
    min-height: 200px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 10px;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
    background: rgba(255,255,255,0.01);
    padding: 24px;
    text-align: center;
  }
  .drop-zone:hover, .drop-zone.drag-over {
    border-color: var(--accent);
    background: rgba(0,229,180,0.03);
  }
  .drop-zone-icon {
    width: 44px; height: 44px;
    border-radius: 10px;
    background: rgba(255,255,255,0.04);
    display: flex; align-items: center; justify-content: center;
  }
  .drop-zone-title { font-weight: 500; font-size: 13px; }
  .drop-zone-sub { font-size: 12px; color: var(--muted); }
  .drop-zone-hint {
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--muted);
    border: 1px solid var(--border-hi);
    padding: 3px 9px;
    border-radius: 99px;
  }

  /* ── Preview ── */
  .preview-wrap {
    position: relative;
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid var(--border-hi);
  }
  .preview-wrap img { width: 100%; display: block; }
  .preview-clear {
    position: absolute;
    top: 9px; right: 9px;
    width: 30px; height: 30px;
    background: rgba(0,0,0,0.7);
    border: 1px solid var(--border-hi);
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    cursor: pointer;
    color: var(--text);
    transition: background 0.15s;
  }
  .preview-clear:hover { background: rgba(239,68,68,0.3); }

  /* ── Webcam ── */
  .webcam-preview {
    position: relative;
    border-radius: 10px;
    overflow: hidden;
    background: #000;
    border: 1px solid var(--border-hi);
    min-height: 200px;
    display: flex; align-items: center; justify-content: center;
  }
  .webcam-preview video { width: 100%; display: block; }
  .live-badge {
    position: absolute;
    top: 10px; left: 10px;
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 0.1em;
    background: rgba(239,68,68,0.9);
    color: white;
    padding: 3px 9px;
    border-radius: 4px;
    display: flex; align-items: center; gap: 5px;
  }
  .webcam-idle {
    display: flex; flex-direction: column; align-items: center; gap: 10px;
    color: var(--muted); text-align: center; padding: 40px;
    font-size: 13px;
  }

  /* ── Buttons ── */
  .btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 7px;
    padding: 9px 16px;
    border: none;
    border-radius: 7px;
    font-family: var(--font-body);
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap;
  }
  .btn-primary { background: var(--accent); color: #000; }
  .btn-primary:hover { filter: brightness(1.08); }
  .btn-primary:disabled { background: var(--border-hi); color: var(--muted); cursor: not-allowed; filter: none; }
  .btn-secondary {
    background: rgba(255,255,255,0.04);
    color: var(--text);
    border: 1px solid var(--border-hi);
  }
  .btn-secondary:hover { background: rgba(255,255,255,0.08); }
  .btn-danger {
    background: rgba(239,68,68,0.1);
    color: var(--danger);
    border: 1px solid rgba(239,68,68,0.18);
  }
  .btn-danger:hover { background: rgba(239,68,68,0.18); }
  .btn-row {
    display: flex;
    gap: 9px;
    margin-top: 12px;
    flex-wrap: wrap;
  }
  .btn-row .btn { flex: 1; min-width: 110px; }

  /* ── Analyzing state ── */
  .analyzing-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 18px;
    padding: 56px 24px;
    text-align: center;
  }
  .scan-ring {
    width: 64px; height: 64px;
    border-radius: 50%;
    border: 2px solid var(--border-hi);
    border-top-color: var(--accent);
    animation: spin 0.85s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .scan-label {
    font-family: var(--font-mono);
    font-size: 11px;
    letter-spacing: 0.1em;
    color: var(--muted);
  }

  /* ── Result Panel ── */
  .result-score-block {
    display: flex;
    align-items: flex-start;
    gap: 20px;
    padding-bottom: 18px;
    margin-bottom: 18px;
    border-bottom: 1px solid var(--border);
  }
  .score-number {
    font-family: var(--font-head);
    font-size: 58px;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -3px;
  }
  .score-label { font-family: var(--font-mono); font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); margin-top: 4px; }
  .score-badge {
    display: inline-flex;
    align-items: center;
    padding: 3px 10px;
    border-radius: 99px;
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-weight: 500;
    margin-top: 8px;
  }
  .score-bar-wrap { flex: 1; padding-top: 6px; }
  .score-bar-bg { height: 5px; background: var(--border); border-radius: 99px; overflow: hidden; margin-bottom: 7px; }
  .score-bar-fill { height: 100%; border-radius: 99px; transition: width 1s cubic-bezier(0.34, 1.56, 0.64, 1); }
  .score-markers { display: flex; justify-content: space-between; font-family: var(--font-mono); font-size: 9px; color: var(--muted); }

  .metric-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid var(--border);
    font-size: 12px;
  }
  .metric-row:last-child { border-bottom: none; }
  .metric-key { color: var(--muted); }
  .metric-val { font-family: var(--font-mono); font-size: 11px; color: var(--text); background: rgba(255,255,255,0.04); padding: 2px 7px; border-radius: 4px; }

  .issue-item {
    display: flex;
    align-items: flex-start;
    gap: 9px;
    padding: 9px 11px;
    background: rgba(239,68,68,0.04);
    border: 1px solid rgba(239,68,68,0.1);
    border-radius: 7px;
    font-size: 12px;
    color: var(--text);
    margin-bottom: 7px;
  }
  .no-issues {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 10px 12px;
    background: rgba(16,185,129,0.04);
    border: 1px solid rgba(16,185,129,0.12);
    border-radius: 7px;
    font-size: 12px;
    color: var(--good);
  }
  .feedback-block {
    background: rgba(0,102,255,0.04);
    border-left: 2px solid var(--accent2);
    border-radius: 0 7px 7px 0;
    padding: 12px 14px;
    font-size: 13px;
    color: var(--text);
    line-height: 1.7;
  }
  .prob-bar-row { display: flex; align-items: center; gap: 9px; margin-bottom: 7px; font-size: 11px; }
  .prob-cls { color: var(--muted); width: 76px; flex-shrink: 0; font-family: var(--font-mono); }
  .prob-bg { flex: 1; height: 3px; background: var(--border); border-radius: 99px; overflow: hidden; }
  .prob-fill { height: 100%; background: var(--accent); border-radius: 99px; }
  .prob-pct { font-family: var(--font-mono); font-size: 10px; color: var(--muted); width: 36px; text-align: right; }

  .epa-error {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 13px;
    background: rgba(239,68,68,0.05);
    border: 1px solid rgba(239,68,68,0.18);
    border-radius: 8px;
    color: var(--danger);
    font-size: 12px;
    line-height: 1.6;
  }
  .epa-error svg { flex-shrink: 0; margin-top: 1px; }

  /* ── Guide ── */
  .guide-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  .guide-block { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 18px; }
  .guide-block-title { font-family: var(--font-head); font-size: 13px; font-weight: 600; color: var(--text); margin-bottom: 10px; display: flex; align-items: center; gap: 7px; }
  .guide-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
  .guide-list { list-style: none; display: flex; flex-direction: column; gap: 7px; }
  .guide-list li { display: flex; align-items: flex-start; gap: 7px; font-size: 12px; color: var(--muted); line-height: 1.5; }
  .guide-list li::before { content: ''; width: 4px; height: 4px; border-radius: 50%; background: var(--border-hi); flex-shrink: 0; margin-top: 6px; }
  .score-band-table { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 4px; }
  .score-band-table tr { border-bottom: 1px solid var(--border); }
  .score-band-table tr:last-child { border-bottom: none; }
  .score-band-table td { padding: 8px 10px; color: var(--muted); }
  .score-band-table td:first-child { font-family: var(--font-mono); font-size: 11px; }

  /* ── Footer ── */
  .epa-footer {
    grid-area: footer;
    border-top: 1px solid var(--border);
    padding: 14px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--surface);
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--muted);
  }

  /* ── Responsive ── */
  @media (max-width: 960px) {
    .epa-shell { grid-template-columns: 1fr; grid-template-areas: "topbar" "main" "footer"; }
    .epa-sidebar { display: none; }
    .epa-content-grid { grid-template-columns: 1fr; }
    .guide-grid { grid-template-columns: 1fr; }
    .epa-main { padding: 20px; }
    .land-nav { padding: 0 20px; }
    .land-features, .land-how { padding: 60px 20px; }
    .features-grid { grid-template-columns: 1fr; }
    .how-steps { grid-template-columns: 1fr 1fr; }
    .land-cta-banner { margin: 0 20px 60px; padding: 40px 24px; }
    .land-footer { padding: 20px; }
    .hero-stats { gap: 28px; }
    .land-nav-links { display: none; }
  }
`;

/* ─────────────────────────────────────────────────────────────────────────
   SHARED UTILS
───────────────────────────────────────────────────────────────────────── */
function scoreAppearance(score) {
  if (score >= 85) return { color: '#10b981', label: 'Excellent' };
  if (score >= 70) return { color: '#00e5b4', label: 'Good' };
  if (score >= 50) return { color: '#f59e0b', label: 'Fair' };
  return { color: '#ef4444', label: 'Poor' };
}

/* ─────────────────────────────────────────────────────────────────────────
   LANDING PAGE
───────────────────────────────────────────────────────────────────────── */
function LandingPage({ onEnterApp }) {
  const featuresRef = useRef(null);
  const howRef      = useRef(null);
  const docsRef     = useRef(null);

  const scrollTo = (ref) => {
    if (ref.current) {
      ref.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  const features = [
    {
      icon: <Cpu size={20} />,
      color: '#00e5b4',
      bg: 'rgba(0,229,180,0.08)',
      title: 'Hybrid AI Engine',
      desc: 'Combines deep learning classification (70%) with MediaPipe geometric analysis (30%) for the most accurate posture scoring available.',
    },
    {
      icon: <Eye size={20} />,
      color: '#0066ff',
      bg: 'rgba(0,102,255,0.08)',
      title: 'Real-Time Webcam Analysis',
      desc: 'Capture your posture live from your webcam. No uploads, no delays — instant ergonomic feedback in seconds.',
    },
    {
      icon: <BarChart2 size={20} />,
      color: '#f59e0b',
      bg: 'rgba(245,158,11,0.08)',
      title: 'Detailed Biomechanics',
      desc: 'Get precise measurements — spine angle, shoulder slope, head-forward distance — with actionable correction guidance.',
    },
    {
      icon: <Shield size={20} />,
      color: '#a855f7',
      bg: 'rgba(168,85,247,0.08)',
      title: 'Privacy First',
      desc: 'Images are processed in-session only and never stored. Your posture data remains entirely private.',
    },
    {
      icon: <Activity size={20} />,
      color: '#ef4444',
      bg: 'rgba(239,68,68,0.08)',
      title: 'Clinical-Grade Scoring',
      desc: 'A 0–100 composite score with four clinical bands: Excellent, Good, Fair, and Poor for clear interpretation.',
    },
    {
      icon: <Zap size={20} />,
      color: '#00e5b4',
      bg: 'rgba(0,229,180,0.08)',
      title: 'Instant Results',
      desc: 'Analysis completes in under 3 seconds. Receive a full breakdown with class probabilities and model confidence.',
    },
  ];

  const steps = [
    { n: '01', title: 'Open Analyzer', desc: 'Launch the assessment tool from this page.' },
    { n: '02', title: 'Upload or Capture', desc: 'Use a photo or your webcam for a live frame.' },
    { n: '03', title: 'AI Processes', desc: 'Our hybrid model runs posture inference.' },
    { n: '04', title: 'Review Results', desc: 'Get your score, metrics, and recommendations.' },
  ];

  return (
    <div className="land-root page-enter">
      {/* NAV */}
      <nav className="land-nav">
        <div className="land-nav-logo">
          <div className="logo-chip"><Zap size={14} /></div>
          PostureAI
        </div>
        <div className="land-nav-links">
          <a onClick={() => scrollTo(featuresRef)}>Features</a>
          <a onClick={() => scrollTo(howRef)}>How It Works</a>
          <a onClick={() => scrollTo(docsRef)}>Documentation</a>
          <button className="nav-cta" onClick={onEnterApp}>
            Open Analyzer <ArrowRight size={13} />
          </button>
        </div>
      </nav>

      {/* HERO */}
      <section className="land-hero">
        <div className="hero-glow" />
        <div className="hero-glow2" />
        <div className="hero-grid" />
        <div className="hero-content">
          <div className="hero-badge">
            <span className="badge-dot" />
            AI-Powered Ergonomic Assessment
          </div>
          <h1 className="hero-title">
            Analyze Your Posture<br />
            with <span className="hero-title-accent">Clinical Precision</span>
          </h1>
          <p className="hero-sub">
            PostureAI combines MediaPipe skeleton tracking and deep learning classification
            to deliver professional-grade ergonomic analysis — in seconds.
          </p>
          <div className="hero-actions">
            <button className="btn-hero-primary" onClick={onEnterApp}>
              Start Analysis <ArrowRight size={15} />
            </button>
            <button className="btn-hero-secondary" onClick={() => scrollTo(docsRef)}>
              View Documentation
            </button>
          </div>

          <div className="hero-stats">
            <div className="stat-item">
              <div className="stat-num">95%</div>
              <div className="stat-label">Detection Accuracy</div>
            </div>
            <div className="stat-divider" />
            <div className="stat-item">
              <div className="stat-num">&lt; 3s</div>
              <div className="stat-label">Analysis Time</div>
            </div>
            <div className="stat-divider" />
            <div className="stat-item">
              <div className="stat-num">Hybrid</div>
              <div className="stat-label">AI Architecture</div>
            </div>
            <div className="stat-divider" />
            <div className="stat-item">
              <div className="stat-num">v2.1</div>
              <div className="stat-label">Current Version</div>
            </div>
          </div>
        </div>
      </section>

      {/* FEATURES */}
      <section className="land-features" ref={featuresRef}>
        <div className="features-header">
          <div className="section-eyebrow">Capabilities</div>
          <h2 className="section-title">Everything you need<br />for ergonomic assessment</h2>
          <p className="section-sub">
            Built on a hybrid AI pipeline that gives you clinical-grade measurements, not guesswork.
          </p>
        </div>
        <div className="features-grid">
          {features.map((f, i) => (
            <div className="feature-card" key={i}>
              <div className="feature-icon-wrap" style={{ background: f.bg }}>
                <span style={{ color: f.color }}>{f.icon}</span>
              </div>
              <div className="feature-title">{f.title}</div>
              <div className="feature-desc">{f.desc}</div>
            </div>
          ))}
        </div>
      </section>

      {/* HOW IT WORKS */}
      <section className="land-how" ref={howRef}>
        <div className="section-eyebrow">Workflow</div>
        <h2 className="section-title">How it works</h2>
        <div className="how-steps">
          {steps.map((s, i) => (
            <div className="how-step" key={i}>
              <div className="step-num">{s.n}</div>
              <div className="step-title">{s.title}</div>
              <div className="step-desc">{s.desc}</div>
            </div>
          ))}
        </div>
      </section>

      {/* DOCUMENTATION SECTION */}
      <section ref={docsRef} style={{ padding: '80px 48px', maxWidth: 1200, margin: '0 auto' }}>
        <div className="section-eyebrow">Documentation</div>
        <h2 className="section-title">Setup & scoring guide</h2>
        <p className="section-sub" style={{ marginBottom: 40 }}>
          Follow these guidelines to get the most accurate results from the analyzer.
        </p>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          {[
            {
              color: '#00e5b4',
              title: 'Optimal Camera Setup',
              items: [
                'Face the camera directly — frontal view only',
                'Keep 2–3 feet distance from the lens',
                'Ensure head, shoulders & torso are fully visible',
                'Use diffuse front-facing lighting',
                'Avoid busy or cluttered backgrounds',
              ],
            },
            {
              color: '#ef4444',
              title: 'Common Mistakes',
              items: [
                'Side or angled profile views',
                'Cropped images missing head or shoulders',
                'Strong backlight or window glare behind you',
                'Very high or low camera angles',
                'Motion blur or very low resolution',
              ],
            },
            {
              color: '#f59e0b',
              title: 'Score Bands',
              items: [
                '85–100 · Excellent — Maintain current posture',
                '70–84 · Good — Minor adjustments needed',
                '50–69 · Fair — Significant corrections needed',
                '0–49 · Poor — Immediate intervention required',
              ],
            },
          ].map((block, i) => (
            <div key={i} className="feature-card">
              <div style={{ width: 8, height: 8, borderRadius: '50%', background: block.color, marginBottom: 16 }} />
              <div className="feature-title">{block.title}</div>
              <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 8, marginTop: 8 }}>
                {block.items.map((item, j) => (
                  <li key={j} style={{ fontSize: 13, color: 'var(--muted)', lineHeight: 1.5, display: 'flex', gap: 8 }}>
                    <span style={{ color: block.color, flexShrink: 0, marginTop: 2 }}>›</span>
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </section>

      {/* CTA BANNER */}
      <div className="land-cta-banner">
        <div className="cta-banner-glow" />
        <h2 className="cta-banner-title">Ready to improve your posture?</h2>
        <p className="cta-banner-sub">Launch the analyzer and get your first assessment in under a minute.</p>
        <button className="btn-hero-primary" onClick={onEnterApp} style={{ position: 'relative' }}>
          Open PostureAI Analyzer <ArrowRight size={15} />
        </button>
      </div>

      {/* FOOTER */}
      <footer className="land-footer">
        <span>© 2025 PostureAI — Ergonomic Analysis System</span>
        <span>Powered by MediaPipe + Deep Learning · v2.1.0</span>
      </footer>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────
   ANALYZER SUB-COMPONENTS
───────────────────────────────────────────────────────────────────────── */
function StatusPill({ status, onRefresh }) {
  return (
    <button className={`epa-status-pill ${status.ready ? 'online' : 'offline'}`} onClick={onRefresh}>
      <span className={`dot ${status.ready ? 'pulse' : ''}`} />
      {status.ready ? 'Agent Online' : 'Agent Offline'}
      <RefreshCw size={11} />
    </button>
  );
}

function AnalyzingState() {
  return (
    <div className="analyzing-state">
      <div className="scan-ring" />
      <div>
        <p style={{ fontSize: 13, color: 'var(--text)', fontWeight: 500 }}>Analyzing Posture</p>
        <p className="scan-label" style={{ marginTop: 5 }}>Running AI inference…</p>
      </div>
    </div>
  );
}

function ErrorBlock({ message }) {
  return (
    <div className="epa-error">
      <AlertCircle size={15} />
      <span>{message}</span>
    </div>
  );
}

function ResultPanel({ result }) {
  const app = scoreAppearance(result.score);
  const [animated, setAnimated] = useState(false);
  useEffect(() => { const t = setTimeout(() => setAnimated(true), 80); return () => clearTimeout(t); }, []);

  return (
    <div>
      <div className="result-score-block">
        <div>
          <div className="score-number" style={{ color: app.color }}>{result.score}</div>
          <div className="score-label">Posture Score / 100</div>
          <span className="score-badge" style={{ background: `${app.color}18`, color: app.color, border: `1px solid ${app.color}40` }}>
            {app.label}
          </span>
        </div>
        <div className="score-bar-wrap">
          <div className="score-bar-bg">
            <div className="score-bar-fill" style={{ width: animated ? `${result.score}%` : '0%', background: app.color }} />
          </div>
          <div className="score-markers">
            <span>0</span><span>Poor</span><span>Fair</span><span>Good</span><span>100</span>
          </div>
        </div>
      </div>

      {result.feedback && (
        <div style={{ marginBottom: 16 }}>
          <div className="epa-card-title">Clinical Feedback</div>
          <div className="feedback-block">{result.feedback}</div>
        </div>
      )}

      {result.metrics && Object.keys(result.metrics).length > 0 && (
        <div style={{ marginBottom: 16 }}>
          <div className="epa-card-title">Biomechanical Metrics</div>
          {Object.entries(result.metrics).map(([k, v]) => v != null && (
            <div className="metric-row" key={k}>
              <span className="metric-key">{k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</span>
              <span className="metric-val">{typeof v === 'number' ? `${Number(v).toFixed(2)}°` : String(v)}</span>
            </div>
          ))}
        </div>
      )}

      <div style={{ marginBottom: 16 }}>
        <div className="epa-card-title">Detected Issues</div>
        {result.issues && result.issues.length > 0
          ? result.issues.map((issue, i) => (
              <div className="issue-item" key={i}>
                <AlertCircle size={13} style={{ color: 'var(--danger)', flexShrink: 0, marginTop: 1 }} />
                <span>{issue}</span>
              </div>
            ))
          : <div className="no-issues"><Check size={13} style={{ flexShrink: 0 }} /> No significant postural deviations detected.</div>
        }
      </div>

      {result.dlClassification && (
        <div style={{ marginBottom: 16 }}>
          <div className="epa-card-title">AI Classification</div>
          <div className="metric-row">
            <span className="metric-key">Predicted Class</span>
            <span className="metric-val" style={{ textTransform: 'capitalize' }}>{result.dlClassification.predicted_class}</span>
          </div>
          <div className="metric-row">
            <span className="metric-key">Confidence</span>
            <span className="metric-val">{(result.dlClassification.confidence * 100).toFixed(1)}%</span>
          </div>
          {result.dlClassification.all_probabilities && (
            <div style={{ marginTop: 12 }}>
              <div style={{ fontSize: 10, color: 'var(--muted)', fontFamily: 'var(--font-mono)', marginBottom: 8, letterSpacing: '0.1em' }}>CLASS PROBABILITIES</div>
              {Object.entries(result.dlClassification.all_probabilities).map(([cls, prob]) => (
                <div className="prob-bar-row" key={cls}>
                  <span className="prob-cls">{cls}</span>
                  <div className="prob-bg"><div className="prob-fill" style={{ width: `${(prob * 100).toFixed(1)}%` }} /></div>
                  <span className="prob-pct">{(prob * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {result.scores && (
        <div style={{ marginBottom: 16 }}>
          <div className="epa-card-title">Composite Scores</div>
          {[['Combined', result.scores.combined], ['Deep Learning', result.scores.deep_learning], ['MediaPipe', result.scores.mediapipe]]
            .map(([label, val]) => val != null && (
              <div className="metric-row" key={label}>
                <span className="metric-key">{label}</span>
                <span className="metric-val">{val} / 100</span>
              </div>
            ))}
        </div>
      )}

      {result.method && (
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--muted)', padding: '6px 10px', border: '1px solid var(--border-hi)', borderRadius: 5, display: 'inline-flex', alignItems: 'center', gap: 7 }}>
          <Zap size={10} /> {result.method.toUpperCase().replace(/_/g, ' ')}
        </div>
      )}
    </div>
  );
}

async function runAnalysis(imageData, agentStatus, setLoading, setError, setResult) {
  if (!imageData) return;
  if (!agentStatus.ready) { setError('Agent is offline. Please check the connection.'); return; }
  setLoading(true); setError(null); setResult(null);
  try {
    const res = await fetch(AGENT_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: [{ role: 'user', content: imageData }] }),
    });
    const data = await res.json();
    if (data.status === 'error') { setError(data.error_message || 'Analysis failed.'); return; }
    if (data.status === 'success' && data.data) {
      const a = data.data.posture_analysis;
      let metrics = a.metrics || {}, issues = a.issues || [];
      if (a.mediapipe_analysis) {
        if (a.mediapipe_analysis.metrics && !Object.keys(metrics).length) metrics = a.mediapipe_analysis.metrics;
        if (a.mediapipe_analysis.issues && !issues.length) issues = a.mediapipe_analysis.issues;
      }
      setResult({ score: a.posture_score, status: a.posture_status, feedback: a.feedback || data.data.message, metrics, issues, method: a.analysis_method, dlClassification: a.dl_classification, scores: a.scores });
    }
  } catch (e) { setError(e.message); }
  finally { setLoading(false); }
}

/* ─────────────────────────────────────────────────────────────────────────
   ANALYZER PAGES
───────────────────────────────────────────────────────────────────────── */
function UploadPage({ agentStatus }) {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [drag, setDrag] = useState(false);
  const fileRef = useRef(null);

  const readFile = (file) => {
    if (!file || !file.type.startsWith('image/')) return;
    const reader = new FileReader();
    reader.onload = () => { setImage(reader.result); setResult(null); setError(null); };
    reader.readAsDataURL(file);
  };

  return (
    <div>
      <h1 className="epa-page-title">Upload & Analyze</h1>
      <p className="epa-page-sub">Submit a frontal photograph of your seated posture for AI assessment.</p>
      <div className="epa-content-grid">
        <div>
          <div className="epa-card">
            <div className="epa-card-title">Image Input</div>
            {!image ? (
              <div className={`drop-zone ${drag ? 'drag-over' : ''}`}
                onClick={() => fileRef.current.click()}
                onDragOver={e => { e.preventDefault(); setDrag(true); }}
                onDragLeave={() => setDrag(false)}
                onDrop={e => { e.preventDefault(); setDrag(false); readFile(e.dataTransfer.files[0]); }}>
                <div className="drop-zone-icon"><Upload size={18} color="var(--muted)" /></div>
                <div className="drop-zone-title">Drop image here or click to browse</div>
                <div className="drop-zone-sub">Frontal view — upper body clearly visible</div>
                <span className="drop-zone-hint">JPG · PNG · WEBP</span>
              </div>
            ) : (
              <div className="preview-wrap">
                <img src={image} alt="Input" />
                <button className="preview-clear" onClick={() => { setImage(null); setResult(null); setError(null); }}><X size={13} /></button>
              </div>
            )}
            <input ref={fileRef} type="file" accept="image/*" style={{ display: 'none' }} onChange={e => readFile(e.target.files[0])} />
            <div className="btn-row">
              <button className="btn btn-secondary" onClick={() => fileRef.current.click()}><Upload size={13} /> Browse</button>
              <button className="btn btn-primary" disabled={!image || loading} onClick={() => runAnalysis(image, agentStatus, setLoading, setError, setResult)}>
                {loading ? 'Analyzing…' : <><span>Run Analysis</span><ChevronRight size={13} /></>}
              </button>
            </div>
            {error && <div style={{ marginTop: 12 }}><ErrorBlock message={error} /></div>}
          </div>
        </div>
        <div>
          <div className="epa-card" style={{ minHeight: 300 }}>
            <div className="epa-card-title">Analysis Results</div>
            {loading && <AnalyzingState />}
            {!loading && result && <ResultPanel result={result} />}
            {!loading && !result && <div style={{ padding: '44px 0', textAlign: 'center', color: 'var(--muted)', fontSize: 12 }}>Results will appear here after analysis.</div>}
          </div>
        </div>
      </div>
    </div>
  );
}

function WebcamPage({ agentStatus }) {
  const [stream, setStream]     = useState(null);
  const [active, setActive]     = useState(false);
  const [captured, setCaptured] = useState(null);
  const [result, setResult]     = useState(null);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => () => { if (stream) stream.getTracks().forEach(t => t.stop()); }, [stream]);

  const startCam = async () => {
    setError(null);
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' } });
      setStream(s); setActive(true);
      setTimeout(() => { if (videoRef.current) { videoRef.current.srcObject = s; videoRef.current.play(); } }, 100);
    } catch (e) {
      if (e.name === 'NotAllowedError') setError('Camera permission denied.');
      else if (e.name === 'NotFoundError') setError('No camera device found.');
      else setError(`Camera error: ${e.message}`);
    }
  };

  const stopCam = () => { if (stream) { stream.getTracks().forEach(t => t.stop()); setStream(null); } setActive(false); };

  const capture = () => {
    const v = videoRef.current, c = canvasRef.current;
    if (!v || !c || v.readyState !== v.HAVE_ENOUGH_DATA) { setError('Video not ready. Please wait.'); return; }
    c.width = v.videoWidth; c.height = v.videoHeight;
    c.getContext('2d').drawImage(v, 0, 0);
    setCaptured(c.toDataURL('image/jpeg', 0.95));
    stopCam(); setResult(null); setError(null);
  };

  return (
    <div>
      <h1 className="epa-page-title">Webcam Capture</h1>
      <p className="epa-page-sub">Capture your posture live from your webcam for real-time analysis.</p>
      <div className="epa-content-grid">
        <div>
          <div className="epa-card">
            <div className="epa-card-title">Camera Input</div>
            {!captured ? (
              <div className="webcam-preview">
                {active ? (
                  <>
                    <video ref={videoRef} autoPlay playsInline muted style={{ width: '100%', display: 'block' }} />
                    <div className="live-badge">
                      <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#ef4444', display: 'inline-block' }} /> LIVE
                    </div>
                  </>
                ) : (
                  <div className="webcam-idle"><Camera size={28} color="var(--muted)" /><p>Activate camera to begin</p></div>
                )}
              </div>
            ) : (
              <div className="preview-wrap">
                <img src={captured} alt="Captured" />
                <button className="preview-clear" onClick={() => { setCaptured(null); setResult(null); setError(null); }}><X size={13} /></button>
              </div>
            )}
            <canvas ref={canvasRef} style={{ display: 'none' }} />
            <div className="btn-row">
              {!active && !captured && <button className="btn btn-secondary" onClick={startCam}><Camera size={13} /> Open Camera</button>}
              {active && (<><button className="btn btn-primary" onClick={capture}>Capture Frame</button><button className="btn btn-danger" onClick={stopCam}><X size={13} /></button></>)}
              {captured && (
                <>
                  <button className="btn btn-secondary" onClick={() => { setCaptured(null); setResult(null); }}>Retake</button>
                  <button className="btn btn-primary" disabled={loading} onClick={() => runAnalysis(captured, agentStatus, setLoading, setError, setResult)}>
                    {loading ? 'Analyzing…' : <><span>Run Analysis</span><ChevronRight size={13} /></>}
                  </button>
                </>
              )}
            </div>
            {error && <div style={{ marginTop: 12 }}><ErrorBlock message={error} /></div>}
          </div>
        </div>
        <div>
          <div className="epa-card" style={{ minHeight: 300 }}>
            <div className="epa-card-title">Analysis Results</div>
            {loading && <AnalyzingState />}
            {!loading && result && <ResultPanel result={result} />}
            {!loading && !result && <div style={{ padding: '44px 0', textAlign: 'center', color: 'var(--muted)', fontSize: 12 }}>Results will appear here after analysis.</div>}
          </div>
        </div>
      </div>
    </div>
  );
}

function GuidePage() {
  return (
    <div>
      <h1 className="epa-page-title">Documentation</h1>
      <p className="epa-page-sub">Guidelines for accurate ergonomic posture assessment.</p>
      <div className="guide-grid">
        <div className="guide-block">
          <div className="guide-block-title"><span className="guide-dot" style={{ background: 'var(--accent)' }} />Optimal Capture Setup</div>
          <ul className="guide-list">
            <li>Position camera directly facing you at eye level</li>
            <li>Maintain 2–3 feet distance from the lens</li>
            <li>Ensure shoulders, head, and upper torso are fully visible</li>
            <li>Use diffuse front-facing lighting — avoid backlighting</li>
            <li>Avoid busy or cluttered backgrounds</li>
          </ul>
        </div>
        <div className="guide-block">
          <div className="guide-block-title"><span className="guide-dot" style={{ background: 'var(--danger)' }} />Common Mistakes</div>
          <ul className="guide-list">
            <li>Side or angled profile views — use frontal view only</li>
            <li>Cropped images excluding the head or shoulders</li>
            <li>Strong backlight or window glare behind subject</li>
            <li>Very high or low camera angles</li>
            <li>Extreme motion blur or very low resolution</li>
          </ul>
        </div>
        <div className="guide-block" style={{ gridColumn: '1 / -1' }}>
          <div className="guide-block-title"><span className="guide-dot" style={{ background: 'var(--warn)' }} />Score Interpretation</div>
          <table className="score-band-table">
            <tbody>
              {[
                ['85–100','#10b981','Excellent','Posture is well-aligned. Maintain current setup.'],
                ['70–84', '#00e5b4','Good',     'Minor adjustments recommended.'],
                ['50–69', '#f59e0b','Fair',     'Significant postural corrections needed.'],
                ['0–49',  '#ef4444','Poor',     'Immediate intervention required.'],
              ].map(([range, color, label, desc]) => (
                <tr key={range}>
                  <td>{range}</td>
                  <td style={{ color, fontWeight: 600 }}>{label}</td>
                  <td>{desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="guide-block">
          <div className="guide-block-title"><span className="guide-dot" style={{ background: 'var(--accent2)' }} />Hybrid Analysis Mode</div>
          <ul className="guide-list">
            <li>Deep Learning classification weighted at 70%</li>
            <li>MediaPipe geometric analysis weighted at 30%</li>
            <li>Provides confidence scores and class probabilities</li>
            <li>Most accurate — requires ML model loaded on server</li>
          </ul>
        </div>
        <div className="guide-block">
          <div className="guide-block-title"><span className="guide-dot" style={{ background: 'var(--muted)' }} />MediaPipe-Only Fallback</div>
          <ul className="guide-list">
            <li>Rule-based skeletal geometry analysis</li>
            <li>Measures spine angle, shoulder slope, head-forward distance</li>
            <li>Activated when DL model is unavailable</li>
            <li>Reliable for most standard posture scenarios</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────
   ANALYZER APP SHELL
───────────────────────────────────────────────────────────────────────── */
function AnalyzerApp({ onBack }) {
  const [page, setPage] = useState('upload');
  const [agentStatus, setAgentStatus] = useState({ ready: false });

  const checkStatus = async () => {
    try {
      const res = await fetch(HEALTH_URL);
      const d = await res.json();
      setAgentStatus({ ready: !!d.ready });
    } catch { setAgentStatus({ ready: false }); }
  };

  useEffect(() => { checkStatus(); }, []);

  const NAV = [
    { id: 'upload', label: 'Upload Image',   icon: <Upload size={15} /> },
    { id: 'webcam', label: 'Webcam Capture', icon: <Camera size={15} /> },
    { id: 'guide',  label: 'Documentation',  icon: <Activity size={15} /> },
  ];

  return (
    <div className="epa-shell page-enter">
      <header className="epa-topbar">
        <div className="epa-topbar-left">
          <button className="epa-back-btn" onClick={onBack}>
            ← Back
          </button>
          <div className="epa-logo">
            <div className="epa-logo-icon"><Zap size={13} /></div>
            PostureAI
            <span className="epa-version">v2.1.0</span>
          </div>
        </div>
        <StatusPill status={agentStatus} onRefresh={checkStatus} />
      </header>

      <nav className="epa-sidebar">
        <div className="epa-sidebar-section">
          <div className="epa-sidebar-label">Navigation</div>
          {NAV.map(n => (
            <button key={n.id} className={`epa-nav-btn ${page === n.id ? 'active' : ''}`} onClick={() => setPage(n.id)}>
              {n.icon}{n.label}
            </button>
          ))}
        </div>
        <div style={{ marginTop: 'auto', padding: '0 14px' }}>
          <div style={{ padding: 12, background: 'rgba(0,229,180,0.04)', border: '1px solid rgba(0,229,180,0.1)', borderRadius: 8, fontSize: 11, color: 'var(--muted)', lineHeight: 1.6 }}>
            <div style={{ color: 'var(--accent)', fontWeight: 600, marginBottom: 6, fontFamily: 'var(--font-mono)', fontSize: 9, letterSpacing: '0.1em' }}>BEST PRACTICE</div>
            Sit upright with your back against the chair. Keep your head level and shoulders relaxed for the most accurate assessment.
          </div>
        </div>
      </nav>

      <main className="epa-main">
        {page === 'upload' && <UploadPage agentStatus={agentStatus} />}
        {page === 'webcam' && <WebcamPage agentStatus={agentStatus} />}
        {page === 'guide'  && <GuidePage />}
      </main>

      <footer className="epa-footer">
        <span>Powered by MediaPipe + Deep Learning</span>
        <span>© 2025 PostureAI — Ergonomic Analysis System</span>
      </footer>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────
   ROOT
───────────────────────────────────────────────────────────────────────── */
export default function App() {
  const [view, setView] = useState('landing'); // 'landing' | 'analyzer'

  return (
    <>
      <style>{GLOBAL_CSS}</style>
      {view === 'landing'
        ? <LandingPage  onEnterApp={() => setView('analyzer')} />
        : <AnalyzerApp  onBack={() => setView('landing')} />
      }
    </>
  );
}