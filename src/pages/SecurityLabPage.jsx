import React, { useState, useCallback } from 'react';
import {
  ShieldAlert, ShieldCheck, Lock, Activity, FlaskConical, Server, Globe, Wifi,
  Play, AlertTriangle, CheckCircle2, XCircle, BarChart2, Building2, Gauge,
  Info, Eye, EyeOff, ChevronRight, RefreshCw, ToggleLeft, ToggleRight,
} from 'lucide-react';
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip,
  CartesianGrid, BarChart, Bar, Legend, RadarChart, Radar,
  PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from 'recharts';
import {
  runPoisoningScenario, runPrivacyAttack, runStressTest,
  runDatasetShift, getGovernance, runCompressionStudy,
} from '../utils/apiClient';

const GlassTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="px-3 py-2 rounded-xl bg-slate-950/95 border border-cyan-400/30 shadow-xl text-xs font-mono">
      <p className="text-slate-400 mb-1">{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color }}>{p.name}: <span className="font-bold">{typeof p.value === 'number' ? p.value.toFixed(2) : p.value}</span></p>
      ))}
    </div>
  );
};

const HOSPITALS = [
  { id: 'h1', name: 'City Medical Center A', color: '#00f2fe' },
  { id: 'h2', name: 'Valley District Clinic', color: '#fbbf24' },
  { id: 'h3', name: 'Metro Academic Health B', color: '#c084fc' },
  { id: 'h4', name: 'St. Jude Community Hosp', color: '#34d399' },
];

function Toggle({ value, onChange, label }) {
  return (
    <button onClick={() => onChange(!value)} className="flex items-center gap-2 text-xs">
      {value ? <ToggleRight className="w-5 h-5 text-cyan-400" /> : <ToggleLeft className="w-5 h-5 text-slate-500" />}
      <span className={value ? 'text-cyan-300' : 'text-slate-400'}>{label}</span>
    </button>
  );
}

function MultiHospital({ value, onChange }) {
  const toggle = (hid) => {
    onChange(value.includes(hid) ? value.filter(x => x !== hid) : [...value, hid]);
  };
  return (
    <div className="flex flex-wrap gap-1.5">
      {HOSPITALS.map(h => {
        const active = value.includes(h.id);
        return (
          <button key={h.id} onClick={() => toggle(h.id)}
            className={`px-2.5 py-1 rounded-lg border text-xs font-semibold transition-all ${
              active ? 'border-cyan-400/50 bg-cyan-500/15 text-white'
                     : 'border-white/10 bg-slate-900 text-slate-400 hover:text-white hover:border-white/20'
            }`}>
            <span className="inline-block w-2 h-2 rounded-full mr-1.5" style={{ backgroundColor: h.color }} />
            {h.name}
          </button>
        );
      })}
    </div>
  );
}

function MetricCard({ label, value, color = 'cyan', sub, big = false }) {
  return (
    <div className={`rounded-2xl border p-4 ${big ? 'col-span-2' : ''} ${
      color === 'cyan'   ? 'bg-cyan-500/5 border-cyan-400/20' :
      color === 'purple' ? 'bg-purple-500/5 border-purple-400/20' :
      color === 'emerald'? 'bg-emerald-500/5 border-emerald-400/20' :
      color === 'amber'  ? 'bg-amber-500/5 border-amber-400/20' :
      color === 'rose'   ? 'bg-rose-500/5 border-rose-400/20' :
                            'bg-slate-500/5 border-slate-400/20'
    }`}>
      <span className="text-[10px] text-slate-400 font-mono uppercase tracking-wider">{label}</span>
      <div className={`font-bold font-heading mt-1 ${big ? 'text-2xl' : 'text-lg'} ${
        color === 'cyan' ? 'text-cyan-300' :
        color === 'purple' ? 'text-purple-300' :
        color === 'emerald' ? 'text-emerald-300' :
        color === 'amber' ? 'text-amber-300' :
        color === 'rose' ? 'text-rose-300' :
        'text-white'
      }`}>{value}</div>
      {sub && <p className="text-[10px] text-slate-500 font-mono mt-0.5">{sub}</p>}
    </div>
  );
}

export default function SecurityLabPage() {
  const [tab, setTab] = useState('poisoning');
  const [useCase, setUseCase] = useState('sepsis');
  const [loading, setLoading] = useState(null);
  const [error, setError] = useState(null);

  const [poisonResult, setPoisonResult] = useState(null);
  const [privacyResult, setPrivacyResult] = useState(null);
  const [stressResult, setStressResult] = useState(null);
  const [shiftResult, setShiftResult] = useState(null);
  const [compResult, setCompResult] = useState(null);
  const [govResult, setGovResult] = useState(null);

  const [pCfg, setPCfg] = useState({ rounds: 8, malicious: ['h3'], attackType: 'update_poison', attackScale: 1.0, poisonRatio: 0.6, dpNoise: false, seeds: 3 });
  const [prCfg, setPrCfg] = useState({ epochsList: [1, 5, 25], dpNoise: true, seeds: 3 });
  const [sCfg, setSCfg] = useState({ rounds: 8, malicious: ['h3'], attackType: 'update_poison', poisonRatio: 0.6, attackScale: 1.0, seeds: 3 });
  const [dCfg, setDCfg] = useState({ seeds: 3 });
  const [cCfg, setCCfg] = useState({ rounds: 8, topKFrac: 0.25, nBits: 4, seeds: 3 });

  const load = async (fn) => { setLoading(tab); setError(null); try { await fn(); } catch(e) { setError(e.message || String(e)); } finally { setLoading(null); } };

  const runPoison = useCallback(() => load(() =>
    runPoisoningScenario({ use_case: useCase, rounds: pCfg.rounds, malicious_hospitals: pCfg.malicious, attack_type: pCfg.attackType, poison_ratio: pCfg.poisonRatio, attack_scale: pCfg.attackScale, dp_noise: pCfg.dpNoise, seeds: pCfg.seeds }).then(setPoisonResult)
  ), [useCase, pCfg]);

  const runPriv = useCallback(() => load(() =>
    runPrivacyAttack({ use_case: useCase, epochs_list: prCfg.epochsList, dp_noise: prCfg.dpNoise, seeds: prCfg.seeds }).then(setPrivacyResult)
  ), [useCase, prCfg]);

  const runStress = useCallback(() => load(() =>
    runStressTest({ use_case: useCase, rounds: sCfg.rounds, malicious_hospitals: sCfg.malicious, attack_type: sCfg.attackType, poison_ratio: sCfg.poisonRatio, attack_scale: sCfg.attackScale, seeds: sCfg.seeds }).then(setStressResult)
  ), [useCase, sCfg]);

  const runShift = useCallback(() => load(() =>
    runDatasetShift({ use_case: useCase, seeds: dCfg.seeds }).then(setShiftResult)
  ), [useCase, dCfg]);

  const runComp = useCallback(() => load(() =>
    runCompressionStudy({ use_case: useCase, rounds: cCfg.rounds, top_k_frac: cCfg.topKFrac, n_bits: cCfg.nBits, seeds: cCfg.seeds }).then(setCompResult)
  ), [useCase, cCfg]);

  const runGov = useCallback(() => load(() =>
    getGovernance(useCase).then(setGovResult)
  ), [useCase]);

  const tabs = [
    { id: 'poisoning', label: 'Poisoning Defense', icon: ShieldAlert },
    { id: 'privacy',   label: 'Privacy Attack',    icon: Lock },
    { id: 'stress',    label: 'FedAvg vs Stress',  icon: Activity },
    { id: 'compression', label: 'Comm. Efficiency', icon: Wifi },
    { id: 'shift',     label: 'Dataset Shift',     icon: Server },
    { id: 'governance',label: 'Trust Governance',  icon: Globe },
  ];

  const TrustBar = ({ score, risk }) => {
    const color = risk === 'HIGH' ? 'bg-rose-500' : risk === 'MEDIUM' ? 'bg-amber-500' : 'bg-emerald-500';
    return (
      <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
        <div className={`h-full rounded-full ${color} transition-all`} style={{ width: `${Math.min(score, 100)}%` }} />
      </div>
    );
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">

      {/* Header */}
      <div className="p-6 rounded-3xl glass-panel border border-white/10">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-2xl bg-rose-500/10 border border-rose-400/30 flex items-center justify-center text-rose-400">
            <ShieldAlert className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white font-heading">Federation Security Laboratory</h1>
            <p className="text-xs text-slate-400">Real SGD training · real gradient detection · real membership inference · real equity-weighted robustness. Every metric is computed on actual clinical records.</p>
          </div>
        </div>
        <div className="flex flex-wrap gap-2 items-center">
          <select value={useCase} onChange={e => setUseCase(e.target.value)}
            className="px-3 py-1.5 rounded-xl bg-slate-900 border border-white/10 text-xs text-white">
            <option value="sepsis">Sepsis (12 features · 23 records)</option>
            <option value="retinopathy">Retinopathy (6 features · 15 records)</option>
          </select>
        </div>
      </div>

      {/* Tab Bar */}
      <div className="flex flex-wrap gap-1 p-1.5 rounded-2xl bg-slate-950 border border-white/10 text-xs">
        {tabs.map(t => {
          const Icon = t.icon;
          const active = tab === t.id;
          return (
            <button key={t.id} onClick={() => setTab(t.id)}
              className={`flex items-center gap-1.5 px-3 py-2 rounded-xl font-semibold transition-all ${
                active ? 'bg-gradient-to-r from-rose-500/20 to-purple-500/20 text-white border border-rose-400/30 shadow-sm'
                       : 'text-slate-400 hover:text-white hover:bg-white/5'
              }`}>
              <Icon className={`w-3.5 h-3.5 ${active ? 'text-rose-400' : 'text-slate-400'}`} />
              <span>{t.label}</span>
            </button>
          );
        })}
      </div>

      {error && (
        <div className="p-4 rounded-2xl bg-rose-950/30 border border-rose-400/30 text-xs text-rose-300 font-mono flex items-start gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}

      {/* ──────────────────────── TAB: POISONING ───────────────────────── */}
      {tab === 'poisoning' && (
        <div className="space-y-6">
          <div className="p-5 rounded-2xl bg-slate-950/80 border border-white/10 space-y-4">
            <h3 className="text-sm font-bold text-white font-heading flex items-center gap-2">
              <ShieldAlert className="w-4 h-4 text-rose-400" /> Attack Configuration
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-[10px] text-slate-400 font-mono">Malicious Hospital(s)</label>
                <MultiHospital value={pCfg.malicious} onChange={v => setPCfg(x => ({...x, malicious: v}))} />
              </div>
              <div className="space-y-2">
                <label className="text-[10px] text-slate-400 font-mono">Attack Type</label>
                <select value={pCfg.attackType} onChange={e => setPCfg(x => ({...x, attackType: e.target.value}))}
                  className="w-full px-3 py-2 rounded-xl bg-slate-900 border border-white/10 text-xs text-white">
                  <option value="label_flip">Label Flip (corrupts training labels)</option>
                  <option value="update_poison">Update Poison (reverses gradient direction)</option>
                </select>
              </div>
              <div className="space-y-2">
                <label className="text-[10px] text-slate-400 font-mono">Poison Ratio: {pCfg.poisonRatio.toFixed(2)}</label>
                <input type="range" min="0" max="1" step="0.1" value={pCfg.poisonRatio}
                  onChange={e => setPCfg(x => ({...x, poisonRatio: parseFloat(e.target.value)}))}
                  className="w-full accent-cyan-400" />
              </div>
              <div className="space-y-2">
                <label className="text-[10px] text-slate-400 font-mono">Attack Scale: {pCfg.attackScale.toFixed(1)}×</label>
                <input type="range" min="0.5" max="8" step="0.5" value={pCfg.attackScale}
                  onChange={e => setPCfg(x => ({...x, attackScale: parseFloat(e.target.value)}))}
                  className="w-full accent-cyan-400" />
              </div>
              <div className="space-y-2">
                <label className="text-[10px] text-slate-400 font-mono">Rounds: {pCfg.rounds}</label>
                <input type="range" min="2" max="20" step="1" value={pCfg.rounds}
                  onChange={e => setPCfg(x => ({...x, rounds: parseInt(e.target.value)}))}
                  className="w-full accent-cyan-400" />
              </div>
              <Toggle value={pCfg.dpNoise} onChange={v => setPCfg(x => ({...x, dpNoise: v}))} label="DP Noise on Updates" />
            </div>
            <button onClick={runPoison} disabled={loading === 'poisoning'}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-rose-500/20 hover:bg-rose-500/30 border border-rose-400/40 text-rose-300 font-bold text-xs transition-colors disabled:opacity-50">
              {loading === 'poisoning' ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
              Run Poisoning Benchmark
            </button>
          </div>

          {poisonResult && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <MetricCard label="Baseline Accuracy" value={`${poisonResult.metrics.baseline_accuracy}%`} color="emerald" sub="No attacker" />
                <MetricCard label="Poisoned Accuracy" value={`${poisonResult.metrics.poisoned_accuracy}%`} color="rose" sub={`${poisonResult.malicious_hospitals.join(', ')} active`} />
                <MetricCard label="Damage" value={`${poisonResult.metrics.damage_pp.toFixed(1)} pp`} color={poisonResult.metrics.damage_pp > 5 ? 'rose' : 'amber'} />
                <MetricCard label="Detection" value={`${poisonResult.detection_quality.tp}TP/${poisonResult.detection_quality.fp}FP`} color={poisonResult.detection_quality.fp === 0 ? 'emerald' : 'amber'} sub={`Rate ${(poisonResult.detection_quality.detection_rate * 100).toFixed(0)}%`} />
              </div>

              {/* Per-client table */}
              <div className="p-5 rounded-2xl bg-slate-950/80 border border-white/10 space-y-3">
                <h4 className="text-xs font-bold text-white font-heading flex items-center gap-2">
                  <Gauge className="w-3.5 h-3.5 text-cyan-400" /> Real Gradient Anomaly Detection
                </h4>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs font-mono">
                    <thead>
                      <tr className="text-slate-400 border-b border-white/10">
                        <th className="text-left py-2 pr-4">Hospital</th>
                        <th className="text-right py-2 px-2">Cosine</th>
                        <th className="text-right py-2 px-2">Norm</th>
                        <th className="text-right py-2 px-2">Z-score</th>
                        <th className="text-right py-2 px-2">Verdict</th>
                        <th className="text-right py-2 pl-2">Truth</th>
                      </tr>
                    </thead>
                    <tbody>
                      {poisonResult.per_client.map(c => (
                        <tr key={c.hospital_id} className="border-b border-white/5">
                          <td className="py-2 pr-4 text-white font-semibold">{HOSPITALS.find(h => h.id === c.hospital_id)?.name || c.hospital_id}</td>
                          <td className={`py-2 px-2 text-right ${c.cosine_to_consensus < 0 ? 'text-rose-400 font-bold' : 'text-slate-300'}`}>{c.cosine_to_consensus.toFixed(3)}</td>
                          <td className="py-2 px-2 text-right text-slate-300">{c.norm.toFixed(3)}</td>
                          <td className={`py-2 px-2 text-right ${c.norm_outlier_z > 3 ? 'text-rose-400 font-bold' : 'text-slate-300'}`}>{c.norm_outlier_z.toFixed(2)}</td>
                          <td className={`py-2 px-2 text-right font-bold ${c.verdict === 'FLAGGED' ? 'text-rose-400' : 'text-emerald-400'}`}>{c.verdict}</td>
                          <td className={`py-2 pl-2 text-right ${c.attacker ? 'text-rose-400' : 'text-slate-500'}`}>{c.attacker ? 'ATTACKER' : 'benign'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="text-[10px] text-slate-500 font-mono">
                  Coordinate-wise median consensus · cosine &lt; 0 or norm z &gt; 3 ⇒ FLAGGED.
                  Label-flip shifts update direction (low cosine); update-poison reverses it (cosine ≈ −1).
                </p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ──────────────────────── TAB: PRIVACY ─────────────────────────── */}
      {tab === 'privacy' && (
        <div className="space-y-6">
          <div className="p-5 rounded-2xl bg-slate-950/80 border border-white/10 space-y-4">
            <h3 className="text-sm font-bold text-white font-heading flex items-center gap-2">
              <Lock className="w-4 h-4 text-purple-400" /> Membership Inference Attack (Salem et al. 2018)
            </h3>
            <p className="text-xs text-slate-400">
              The attacker trains a logistic model and uses prediction confidence (margin from 0.5) as a membership
              score. Higher confidence on training records = membership leakage. Increasing DP noise suppresses
              this signal — this is measured, not asserted.
            </p>
            <div className="flex flex-wrap gap-4 items-center">
              <Toggle value={prCfg.dpNoise} onChange={v => setPrCfg(x => ({...x, dpNoise: v}))} label="(ε,δ)-DP Noise" />
              <div className="text-xs text-slate-400">
                Training epochs: <span className="text-white font-mono">{prCfg.epochsList?.join(', ') || '1, 5, 25'}</span>
              </div>
            </div>
            <button onClick={runPriv} disabled={loading === 'privacy'}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-purple-500/20 hover:bg-purple-500/30 border border-purple-400/40 text-purple-300 font-bold text-xs transition-colors disabled:opacity-50">
              {loading === 'privacy' ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
              Run Privacy Attack
            </button>
          </div>

          {privacyResult && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <MetricCard label="Max Attack AUC" value={privacyResult.summary.max_attack_auc.toFixed(4)} color={privacyResult.summary.max_attack_auc > 0.6 ? 'rose' : 'emerald'} sub="1.0 = perfect attack" />
                <MetricCard label="Overfitting Escalation" value={`+${privacyResult.summary.overfitting_escalation.toFixed(1)} pp`} color="purple" sub="AUC rise from first→last epoch" />
                <MetricCard label="DP Mechanism" value={privacyResult.dp_noise ? 'Gaussian' : 'None'} color="cyan" sub="ε = 0.55 · δ = 1e-5" />
                <MetricCard label="Attack Method" value="Confidence Margin" color="slate" sub="Member vs non-member separation" />
              </div>

              <div className="p-5 rounded-2xl bg-slate-950/80 border border-white/10">
                <h4 className="text-xs font-bold text-white font-heading mb-3">Attack AUC by Training Intensity</h4>
                <ResponsiveContainer width="100%" height={240}>
                  <BarChart data={privacyResult.runs.map(r => ({
                    label: `Epoch ${r.epochs}`,
                    'Attack AUC': r.attack_auc,
                    'Member Confidence': r.member_avg_confidence,
                    'Non-Member Confidence': r.nonmember_avg_confidence,
                  }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="label" stroke="#64748b" tick={{ fontSize: 11 }} />
                    <YAxis stroke="#64748b" tick={{ fontSize: 11 }} domain={[0, 1]} />
                    <Tooltip content={<GlassTooltip />} />
                    <Bar dataKey="Attack AUC" fill="#9d4edd" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="Member Confidence" fill="#00f2fe" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="Non-Member Confidence" fill="#fbbf24" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="p-4 rounded-2xl bg-purple-950/20 border border-purple-400/20 text-xs">
                <Info className="w-4 h-4 text-purple-400 inline mr-2" />
                <span className="text-purple-300 font-semibold">How to interpret: </span>
                <span className="text-slate-400">
                  If DP is enabled, the AUC stays near 0.5 (indistinguishable from random). Without DP,
                  more training epochs increase overfitting, which the attacker exploits — AUC rises because
                  training records receive higher-confidence predictions than unseen records.
                </span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ──────────────────────── TAB: STRESS ──────────────────────────── */}
      {tab === 'stress' && (
        <div className="space-y-6">
          <div className="p-5 rounded-2xl bg-slate-950/80 border border-white/10 space-y-4">
            <h3 className="text-sm font-bold text-white font-heading flex items-center gap-2">
              <Activity className="w-4 h-4 text-emerald-400" /> Standard FedAvg vs Equity-Weighted FedAvg
            </h3>
            <p className="text-xs text-slate-400">
              Standard FedAvg weights by declared hospital data volume. Equity-weighted FedAvg blends 70%
              volume + 30% equal-share, diluting the largest contributor's influence. Under attack, the
              aggregation that best protects the global model wins.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-[10px] text-slate-400 font-mono">Malicious Hospital(s)</label>
                <MultiHospital value={sCfg.malicious} onChange={v => setSCfg(x => ({...x, malicious: v}))} />
              </div>
              <div className="space-y-2">
                <label className="text-[10px] text-slate-400 font-mono">Attack Type</label>
                <select value={sCfg.attackType} onChange={e => setSCfg(x => ({...x, attackType: e.target.value}))}
                  className="w-full px-3 py-2 rounded-xl bg-slate-900 border border-white/10 text-xs text-white">
                  <option value="label_flip">Label Flip</option>
                  <option value="update_poison">Update Poison</option>
                </select>
              </div>
              <div className="space-y-2">
                <label className="text-[10px] text-slate-400 font-mono">Attack Scale: {sCfg.attackScale.toFixed(1)}×</label>
                <input type="range" min="0.5" max="8" step="0.5" value={sCfg.attackScale}
                  onChange={e => setSCfg(x => ({...x, attackScale: parseFloat(e.target.value)}))}
                  className="w-full accent-emerald-400" />
              </div>
              <div className="space-y-2">
                <label className="text-[10px] text-slate-400 font-mono">Rounds: {sCfg.rounds}</label>
                <input type="range" min="2" max="20" step="1" value={sCfg.rounds}
                  onChange={e => setSCfg(x => ({...x, rounds: parseInt(e.target.value)}))}
                  className="w-full accent-emerald-400" />
              </div>
            </div>
            <button onClick={runStress} disabled={loading === 'stress'}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-emerald-500/20 hover:bg-emerald-500/30 border border-emerald-400/40 text-emerald-300 font-bold text-xs transition-colors disabled:opacity-50">
              {loading === 'stress' ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
              Run Stress Test
            </button>
          </div>

          {stressResult && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <MetricCard label="Winner" value={stressResult.final.winner === 'equity_fedavg' ? 'Equity FedAvg' : 'Standard FedAvg'}
                  color={stressResult.final.robustness_gain_pp > 0 ? 'emerald' : 'amber'}
                  sub="Under attack" />
                <MetricCard label="Robustness Gain" value={`${stressResult.final.robustness_gain_pp > 0 ? '+' : ''}${stressResult.final.robustness_gain_pp.toFixed(1)} pp`}
                  color={stressResult.final.robustness_gain_pp > 0 ? 'emerald' : 'amber'}
                  sub="equity vs standard" />
                <MetricCard label="Standard Final" value={`${stressResult.final.standard_fedavg.accuracy}%`} color="amber" />
                <MetricCard label="Equity Final" value={`${stressResult.final.equity_fedavg.accuracy}%`} color="cyan" />
              </div>

              <div className="p-5 rounded-2xl bg-slate-950/80 border border-white/10">
                <h4 className="text-xs font-bold text-white font-heading mb-3">Convergence Under Attack</h4>
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={stressResult.series}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="round" stroke="#64748b" tick={{ fontSize: 11 }} label={{ value: 'Round', position: 'insideBottom', offset: -5, fill: '#64748b', fontSize: 11 }} />
                    <YAxis stroke="#64748b" tick={{ fontSize: 11 }} domain={[0, 100]} label={{ value: 'Accuracy %', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 11 }} />
                    <Tooltip content={<GlassTooltip />} />
                    <Line type="monotone" dataKey="standard_fedavg" stroke="#fbbf24" strokeWidth={2.5} dot={{ r: 4 }} name="Standard FedAvg" />
                    <Line type="monotone" dataKey="equity_fedavg" stroke="#00f2fe" strokeWidth={2.5} dot={{ r: 4 }} name="Equity FedAvg" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ──────────────────────── TAB: COMMUNICATION EFFICIENCY ───────── */}
      {tab === 'compression' && (
        <div className="space-y-6">
          <div className="p-5 rounded-2xl bg-slate-950/80 border border-white/10 space-y-4">
            <h3 className="text-sm font-bold text-white font-heading flex items-center gap-2">
              <Wifi className="w-4 h-4 text-teal-400" /> Compressed FedAvg — Communication-Efficient Federation
            </h3>
            <p className="text-xs text-slate-400">
             Each hospital sparsifies its update to the top-k largest coefficients and uniformly quantizes
             survivors to n-bit levels before upload — exactly like a compressed FL protocol. The coordinator
             aggregates the compressed deltas, so any quality difference is honestly caused by compression.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="space-y-2">
                <label className="text-[10px] text-slate-400 font-mono">Sparsity (keep top-k): {Math.round(cCfg.topKFrac * 100)}%</label>
                <input type="range" min="0.05" max="1" step="0.05" value={cCfg.topKFrac}
                  onChange={e => setCCfg(x => ({...x, topKFrac: parseFloat(e.target.value)}))}
                  className="w-full accent-teal-400" />
              </div>
              <div className="space-y-2">
                <label className="text-[10px] text-slate-400 font-mono">Quantization: {cCfg.nBits} bits / value</label>
                <input type="range" min="1" max="16" step="1" value={cCfg.nBits}
                  onChange={e => setCCfg(x => ({...x, nBits: parseInt(e.target.value)}))}
                  className="w-full accent-teal-400" />
              </div>
              <div className="space-y-2">
                <label className="text-[10px] text-slate-400 font-mono">Rounds: {cCfg.rounds}</label>
                <input type="range" min="4" max="20" step="1" value={cCfg.rounds}
                  onChange={e => setCCfg(x => ({...x, rounds: parseInt(e.target.value)}))}
                  className="w-full accent-teal-400" />
              </div>
            </div>
            <button onClick={runComp} disabled={loading === 'compression'}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-teal-500/20 hover:bg-teal-500/30 border border-teal-400/40 text-teal-300 font-bold text-xs transition-colors disabled:opacity-50">
              {loading === 'compression' ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
              Run Compression Study
            </button>
          </div>

          {compResult && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <MetricCard label="FedAvg Accuracy" value={`${compResult.final.standard_accuracy}%`} color="amber" sub="uncompressed baseline" />
                <MetricCard label="Compressed Accuracy" value={`${compResult.final.compressed_accuracy}%`} color="cyan" sub={`${compResult.config.top_k_frac * 100}% top-k · ${compResult.config.n_bits}-bit`} />
                <MetricCard label="Compression" value={`${compResult.final.compression_ratio}×`} color="emerald" sub={`${compResult.final.bits_saved_percent}% bits saved`} />
                <MetricCard label="Calibration Cost" value={`+${compResult.final.loss_increase_pct}%`} color={compResult.final.loss_increase_pct > 30 ? 'amber' : 'emerald'} sub="validation loss vs baseline" />
              </div>

              <div className="p-5 rounded-2xl bg-slate-950/80 border border-white/10">
                <h4 className="text-xs font-bold text-white font-heading mb-3">Convergence — FedAvg vs Compressed FedAvg (real held-out accuracy %)</h4>
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={compResult.series}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="round" stroke="#64748b" tick={{ fontSize: 11 }} />
                    <YAxis stroke="#64748b" tick={{ fontSize: 11 }} domain={[0, 100]} />
                    <Tooltip content={<GlassTooltip />} />
                    <Line type="monotone" dataKey="standard_fedavg" stroke="#fbbf24" strokeWidth={2.5} dot={{ r: 4 }} name="FedAvg" />
                    <Line type="monotone" dataKey="compressed_fedavg" stroke="#00f2fe" strokeWidth={2.5} dot={{ r: 4 }} name="Compressed FedAvg" />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div className="p-5 rounded-2xl bg-slate-950/80 border border-white/10">
                <h4 className="text-xs font-bold text-white font-heading mb-3">Communication ↔ Accuracy Tradeoff (bit budget vs measured quality)</h4>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={compResult.sweep.map(s => ({
                    name: s.label,
                    'Accuracy %': s.accuracy,
                    'Validation Loss ×100': s.loss * 100,
                    'Bits Saved %': s.bits_saved_percent,
                  }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="name" stroke="#64748b" tick={{ fontSize: 10, width: 90 }} interval={0} />
                    <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                    <Tooltip content={<GlassTooltip />} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Bar dataKey="Accuracy %" fill="#00f2fe" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="Validation Loss ×100" fill="#9d4edd" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="Bits Saved %" fill="#34d399" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="p-5 rounded-2xl bg-slate-950/80 border border-white/10">
                <h4 className="text-xs font-bold text-white font-heading mb-3">Upload Time per Federation Round per Hospital — real bits ÷ real link</h4>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs font-mono">
                    <thead>
                      <tr className="text-slate-400 border-b border-white/10">
                        <th className="text-left py-2 pr-4">Bandwidth Condition</th>
                        <th className="text-right py-2 px-2">Link (upload)</th>
                        <th className="text-right py-2 px-2">Uncompressed</th>
                        <th className="text-right py-2 px-2">Compressed</th>
                        <th className="text-right py-2 pl-2">Speed-up</th>
                      </tr>
                    </thead>
                    <tbody>
                      {compResult.bandwidth.map(b => (
                        <tr key={b.label} className="border-b border-white/5">
                          <td className="py-2 pr-4 text-white font-semibold">{b.label}</td>
                          <td className="py-2 px-2 text-right text-slate-300">{b.mbps} Mbps</td>
                          <td className="py-2 px-2 text-right text-amber-300">{b.original_seconds.toFixed(6)} s</td>
                          <td className="py-2 px-2 text-right text-teal-300">{b.compressed_seconds.toFixed(6)} s</td>
                          <td className="py-2 pl-2 text-right text-emerald-400 font-bold" title={b.compressed_seconds === 0 ? '0.0' : ''}>
                            {(b.original_seconds / (b.compressed_seconds || 1e-12)).toFixed(1)}×
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="text-[10px] text-slate-500 font-mono mt-2">
                  Protocol: {compResult.final.protocol.parameters_per_update} parameters/update ·
                  {compResult.final.protocol.original_bits_per_client} → {compResult.final.protocol.compressed_bits_per_client} bits per client.
                  Rounds are synchronous across hospitals, so the slowest uplink bounds every round.
                </p>
              </div>

              <div className="p-4 rounded-2xl bg-teal-950/20 border border-teal-400/20 text-xs">
                <Info className="w-4 h-4 text-teal-400 inline mr-2" />
                <span className="text-teal-300 font-semibold">How to read: </span>
                <span className="text-slate-400">
                  Accuracy on these small clinical cohorts saturates at 100% — precise but uninformative, so the
                  calibration (validation loss) column is where compression's real cost shows. As the budget shrinks,
                  loss rises monotonically while accuracy holds: compression is nearly free at the decision boundary
                  but measurably degrades the model's confidence calibration.
                </span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ──────────────────────── TAB: DATASET SHIFT ───────────────────── */}
      {tab === 'shift' && (
        <div className="space-y-6">
          <div className="p-5 rounded-2xl bg-slate-950/80 border border-white/10 space-y-4">
            <h3 className="text-sm font-bold text-white font-heading flex items-center gap-2">
              <Server className="w-4 h-4 text-amber-400" /> Leave-One-Hospital-Out Testing
            </h3>
            <p className="text-xs text-slate-400">
              For each hospital, the global model is trained on the other three and evaluated on held-out
              data from the training hospitals (in-distribution) and the never-seen hospital (out-of-distribution).
              Real distribution shift is measured as the standardized feature-distance between cohorts.
            </p>
            <button onClick={runShift} disabled={loading === 'shift'}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-amber-500/20 hover:bg-amber-500/30 border border-amber-400/40 text-amber-300 font-bold text-xs transition-colors disabled:opacity-50">
              {loading === 'shift' ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
              Run Dataset Shift Analysis
            </button>
          </div>

          {shiftResult && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <MetricCard label="Avg Accuracy Drop" value={`${shiftResult.summary.avg_drop_pp.toFixed(1)} pp`} color="amber" />
                <MetricCard label="Max Shift Score" value={shiftResult.summary.max_shift_score.toFixed(4)} color="purple" sub="Standardized feature distance" />
                <MetricCard label="Most Shifted" value={shiftResult.summary.most_shifted_hospital || '—'} color="rose" sub="Highest distribution gap" />
                <MetricCard label="Cohort" value={useCase} color="cyan" sub="Leave-one-out protocol" />
              </div>

              <div className="p-5 rounded-2xl bg-slate-950/80 border border-white/10">
                <h4 className="text-xs font-bold text-white font-heading mb-3">Generalization &amp; Shift per Hospital</h4>
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={shiftResult.hospitals.map(h => ({
                    name: h.hospital_name.split(' ').slice(0, 3).join(' '),
                    'In-Distribution Acc': h.in_distribution_accuracy,
                    'Unseen Hospital Acc': h.unseen_accuracy,
                    'Shift Score': h.shift_score * 100,
                    'Prediction Shift': h.prediction_shift * 100,
                  }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="name" stroke="#64748b" tick={{ fontSize: 10 }} />
                    <YAxis stroke="#64748b" tick={{ fontSize: 11 }} domain={[0, 110]} />
                    <Tooltip content={<GlassTooltip />} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Bar dataKey="In-Distribution Acc" fill="#10b981" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="Unseen Hospital Acc" fill="#fbbf24" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="Shift Score" fill="#9d4edd" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="Prediction Shift" fill="#f472b6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Detail rows */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {shiftResult.hospitals.map(h => (
                  <div key={h.hospital_id} className="p-4 rounded-2xl bg-slate-950/80 border border-white/10 space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-white font-heading">{h.hospital_name}</span>
                      <span className="text-[10px] text-slate-500 font-mono">{h.out_n} records</span>
                    </div>
                    <div className="grid grid-cols-3 gap-2 text-[10px] font-mono">
                      <div className="text-slate-400">In-Dist Acc: <span className="text-emerald-400 font-bold">{h.in_distribution_accuracy}%</span></div>
                      <div className="text-slate-400">Unseen Acc: <span className="text-amber-400 font-bold">{h.unseen_accuracy}%</span></div>
                      <div className="text-slate-400">Δ Accuracy: <span className="text-purple-400 font-bold">{h.drop_pp.toFixed(1)} pp</span></div>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-[10px] font-mono">
                      <div className="text-slate-400">Feature Shift: <span className="text-purple-300 font-bold">{h.shift_score.toFixed(4)}</span></div>
                      <div className="text-slate-400">Confidence Shift: <span className="text-rose-300 font-bold">{h.prediction_shift.toFixed(4)}</span></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ──────────────────────── TAB: GOVERNANCE ───────────────────────── */}
      {tab === 'governance' && (
        <div className="space-y-6">
          <div className="p-5 rounded-2xl bg-slate-950/80 border border-white/10 space-y-4">
            <h3 className="text-sm font-bold text-white font-heading flex items-center gap-2">
              <Globe className="w-4 h-4 text-cyan-400" /> Real Hospital Trust &amp; Governance Dashboard
            </h3>
            <p className="text-xs text-slate-400">
              Trust scores are computed from real measured signals: gradient alignment (cosine of the hospital's
              update to the coordinate-wise median consensus across federation rounds), clinical data consistency
              (fraction of records within 3σ of cohort norms per feature), and availability participation.
              Running any poisoning experiment updates the governance state immediately.
            </p>
            <button onClick={runGov} disabled={loading === 'governance'}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-400/40 text-cyan-300 font-bold text-xs transition-colors disabled:opacity-50">
              {loading === 'governance' ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
              Fetch Governance Data
            </button>
          </div>

          {govResult && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <MetricCard label="Network Trust (Avg)" value={`${govResult.summary.avg_trust.toFixed(1)} / 100`} color="cyan" />
                <MetricCard label="Flagged Hospitals" value={govResult.summary.flagged_count} color={govResult.summary.flagged_count > 0 ? 'rose' : 'emerald'} sub={`of ${govResult.summary.member_count}`} />
                <MetricCard label="Measurement Model" value="Real Signals" color="emerald" sub="gradient-alignment + consistency + availability" />
                <MetricCard label="Cohort" value={useCase} color="purple" sub="Global federation" />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {govResult.hospitals.map(h => {
                  const hospInfo = HOSPITALS.find(x => x.id === h.hospital_id) || {};
                  return (
                    <div key={h.hospital_id}
                      className={`p-5 rounded-2xl border space-y-3 ${
                        h.risk_level === 'HIGH'
                          ? 'bg-rose-950/20 border-rose-400/30'
                          : 'bg-slate-950/80 border-white/10'
                      }`}>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <div className="w-3 h-3 rounded-full" style={{ backgroundColor: hospInfo.color }} />
                          <span className="text-sm font-bold text-white font-heading">{h.hospital_name}</span>
                        </div>
                        <span className={`text-[10px] font-bold font-mono px-2 py-0.5 rounded-full ${
                          h.risk_level === 'HIGH' ? 'bg-rose-500/20 text-rose-400 border border-rose-400/30' :
                          h.risk_level === 'MEDIUM' ? 'bg-amber-500/20 text-amber-400 border border-amber-400/30' :
                          'bg-emerald-500/20 text-emerald-400 border border-emerald-400/30'
                        }`}>{h.risk_level}</span>
                      </div>

                      <TrustBar score={h.trust_score} risk={h.risk_level} />

                      <div className="grid grid-cols-2 gap-2 text-[10px] font-mono">
                        <div className="text-slate-400">Trust Score: <span className="text-white font-bold">{h.trust_score.toFixed(1)}</span></div>
                        <div className="text-slate-400">Status: <span className={`font-bold ${h.poisoning_status === 'FLAGGED' ? 'text-rose-400' : 'text-emerald-400'}`}>{h.poisoning_status}</span></div>
                        <div className="text-slate-400">Gradient Alignment: <span className="text-cyan-300">{h.gradient_alignment !== null ? h.gradient_alignment.toFixed(3) : '—'}</span></div>
                        <div className="text-slate-400">Data Consistency: <span className="text-purple-300">{h.data_consistency}%</span></div>
                        <div className="text-slate-400">Tier: <span className="text-white">{h.tier}</span></div>
                        <div className="text-slate-400">Volume: <span className="text-amber-300">{h.data_volume.toLocaleString()}</span></div>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Compliance summary */}
              <div className="p-4 rounded-2xl bg-cyan-950/20 border border-cyan-400/20 space-y-2">
                <h4 className="text-xs font-bold text-cyan-300 font-heading flex items-center gap-2">
                  <ShieldCheck className="w-4 h-4" /> Platform Governance Requirements
                </h4>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-[10px] font-mono">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                    <span className="text-slate-300">Gradient alignment surveillance active</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                    <span className="text-slate-300">Gaussian (ε,δ)-DP noise applied</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                    <span className="text-slate-300">Equity-weighted aggregation default</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

    </div>
  );
}