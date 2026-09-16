import React, { useState } from 'react';
import {
  FileSearch, ChevronDown, ChevronUp, Cpu, CheckCircle2, XCircle,
  ListChecks, ScanSearch, Info
} from 'lucide-react';

const SEVERITY_STYLES = {
  low: 'bg-emerald-500/10 text-emerald-300 border-emerald-400/30',
  medium: 'bg-amber-500/10 text-amber-300 border-amber-400/30',
  high: 'bg-orange-500/10 text-orange-300 border-orange-400/30',
  critical: 'bg-rose-500/15 text-rose-300 border-rose-400/40',
};

function SeverityBadge({ severity }) {
  const style = SEVERITY_STYLES[severity] || SEVERITY_STYLES.medium;
  return (
    <span className={`px-2 py-0.5 rounded-full text-[9px] font-bold uppercase tracking-wide border ${style}`}>
      {severity || 'info'}
    </span>
  );
}

/**
 * Transparency Report — single renderer for preprocessing inconsistencies
 * and model-selection rationale. Used identically for Edge and Cloud training.
 */
export default function TransparencyReport({ preprocessing, modelSelection, source }) {
  const [open, setOpen] = useState(false);

  const issues = preprocessing?.resolvedIssues || preprocessing?.resolved_issues || [];
  const profile = preprocessing?.inputProfile || preprocessing?.input_profile || null;
  const ms = modelSelection || {};
  const selectedAlgorithm = ms.selectedAlgorithm || ms.selected_algorithm;
  const rationale = ms.rationale;
  const decisionFactors = ms.decisionFactors || ms.decision_factors;
  const candidates = ms.candidates || [];
  const rejected = ms.rejected || [];
  const title = source
    ? `Data Prep & Model Selection Transparency — ${source}`
    : 'Data Prep & Model Selection Transparency';

  const key = (obj, camel, snake) => obj?.[camel] ?? obj?.[snake];

  return (
    <div className="rounded-3xl glass-panel border border-white/10 overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full p-4 flex items-center justify-between text-xs font-bold text-white font-heading bg-slate-900/60 hover:bg-slate-900 transition-colors"
      >
        <span className="flex items-center gap-2">
          <FileSearch className="w-4 h-4 text-cyan-400" />
          {title}
          {(profile || issues.length > 0) && (
            <span className="px-2 py-0.5 rounded-full text-[9px] font-mono bg-cyan-500/10 text-cyan-300 border border-cyan-400/20">
              {issues.length} issues resolved
            </span>
          )}
        </span>
        {open ? <ChevronUp className="w-4 h-4 text-cyan-400" /> : <ChevronDown className="w-4 h-4 text-cyan-400" />}
      </button>

      {open && (
        <div className="p-4 sm:p-5 space-y-5 text-xs">
          {/* ── Preprocessing inconsistencies ── */}
          <div className="space-y-3">
            <div className="flex items-center gap-2 font-bold text-white font-heading">
              <ScanSearch className="w-4 h-4 text-amber-400" />
              Inconsistencies Found & How They Were Resolved
            </div>

            {profile && (
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                {[
                  { label: 'Records used', value: key(profile, 'recordsUsed', 'records_used') },
                  { label: 'Removed by cleaning', value: key(profile, 'recordsRemovedByCleaning', 'records_removed_by_cleaning') },
                  { label: 'Numeric features', value: key(profile, 'numericFeatures', 'numeric_features') },
                  { label: 'Categorical encoded', value: key(profile, 'categoricalFeatures', 'categorical_features') },
                  { label: 'Missingness (pre-imp.)', value: `${key(profile, 'missingRatePct', 'missing_rate_pct') ?? 0}%` },
                  { label: 'Positive prevalence', value: `${key(profile, 'classBalancePct', 'class_balance_pct') ?? 0}%` },
                  { label: 'Features after encoding', value: key(profile, 'featuresAfterEncoding', 'features_after_encoding') },
                  { label: 'Image input', value: key(profile, 'imageInput', 'image_input') ? 'Yes' : 'No' },
                ].map(item => (
                  <div key={item.label} className="p-2.5 rounded-xl bg-slate-950/70 border border-white/5">
                    <p className="text-[9px] text-slate-400 font-mono uppercase tracking-wide">{item.label}</p>
                    <p className="text-white font-mono font-semibold text-xs mt-0.5">{item.value ?? '—'}</p>
                  </div>
                ))}
              </div>
            )}

            {issues.length === 0 ? (
              <div className="flex items-center gap-2 p-3 rounded-xl bg-emerald-950/40 border border-emerald-500/20 text-emerald-300">
                <CheckCircle2 className="w-4 h-4 shrink-0" />
                No inconsistencies detected — dataset was already clean.
              </div>
            ) : (
              <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
                {issues.map((iss, idx) => (
                  <div key={idx} className="p-3 rounded-xl bg-slate-950/70 border border-white/5">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-slate-200 font-semibold leading-snug">{iss.issue}</span>
                      <SeverityBadge severity={iss.severity} />
                    </div>
                    <p className="text-[10px] text-emerald-300 mt-1.5 flex items-center gap-1">
                      <CheckCircle2 className="w-3 h-3 shrink-0" />
                      {iss.resolution}
                    </p>
                    {iss.count > 1 && (
                      <p className="text-[10px] text-slate-500 font-mono mt-1">× {iss.count} occurrences</p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* ── Model selection rationale ── */}
          <div className="pt-3 border-t border-white/5 space-y-3">
            <div className="flex items-center gap-2 font-bold text-white font-heading">
              <Cpu className="w-4 h-4 text-cyan-400" />
              Why This Model Was Selected
            </div>

            {!selectedAlgorithm ? (
              <div className="flex items-start gap-2 p-3 rounded-xl bg-slate-900 border border-white/5 text-slate-400">
                <Info className="w-4 h-4 shrink-0 mt-0.5" />
                Model selection runs when training starts — the deciding engine will report the full rationale here.
              </div>
            ) : (
              <>
                <div className="p-3 rounded-xl bg-cyan-950/40 border border-cyan-400/30">
                  <p className="text-[9px] uppercase tracking-wide text-slate-400 font-mono mb-1">Selected algorithm</p>
                  <p className="text-sm font-bold text-cyan-300 font-heading">{selectedAlgorithm}</p>
                  <p className="text-slate-300 leading-relaxed mt-1.5">{rationale}</p>
                </div>

                {/* Decision factors */}
                {decisionFactors.length > 0 && (
                  <div className="space-y-1.5">
                    <p className="text-[9px] uppercase tracking-wide text-slate-400 font-mono">Decision factors observed in your data</p>
                    {decisionFactors.map((f, idx) => (
                      <div key={idx} className="flex items-center gap-2 p-2.5 rounded-xl bg-slate-950/70 border border-white/5">
                        <span className="text-slate-300 font-semibold w-1/3 shrink-0">{f.factor}</span>
                        <span className="text-slate-400 font-mono text-[11px] flex-1">{f.observed}</span>
                        <span className="text-emerald-300 text-[10px] text-right">→ {f.tiltsToward}</span>
                      </div>
                    ))}
                  </div>
                )}

                {/* Candidates considered */}
                {candidates.length > 0 && (
                  <div className="space-y-1.5">
                    <p className="text-[9px] uppercase tracking-wide text-slate-400 font-mono">Candidate algorithms evaluated</p>
                    {candidates.map((c, idx) => (
                      <div key={idx} className="flex items-start gap-2 p-2.5 rounded-xl bg-slate-950/70 border border-white/5">
                        <ListChecks className="w-3.5 h-3.5 text-slate-500 shrink-0 mt-0.5" />
                        <div className="flex-1 min-w-0">
                          <p className="text-slate-200 font-semibold">{c.algorithm}</p>
                          <p className="text-[10px] text-slate-400 leading-snug">{c.reason}</p>
                        </div>
                        <span className={`px-1.5 py-0.5 rounded-full text-[9px] font-mono border shrink-0 ${
                          c.suitability === 'high'
                            ? 'bg-emerald-500/10 text-emerald-300 border-emerald-400/30'
                            : c.suitability === 'medium'
                              ? 'bg-amber-500/10 text-amber-300 border-amber-400/30'
                              : 'bg-slate-800 text-slate-400 border-white/10'
                        }`}>
                          {c.suitability || 'n/a'}
                        </span>
                      </div>
                    ))}
                  </div>
                )}

                {/* Rejected */}
                {rejected.length > 0 && (
                  <div className="space-y-1.5">
                    <p className="text-[9px] uppercase tracking-wide text-slate-400 font-mono">Rejected alternatives</p>
                    {rejected.map((r, idx) => (
                      <div key={idx} className="flex items-start gap-2 p-2.5 rounded-xl bg-rose-950/20 border border-rose-400/10">
                        <XCircle className="w-3.5 h-3.5 text-rose-400/70 shrink-0 mt-0.5" />
                        <div className="flex-1 min-w-0">
                          <p className="text-slate-300 font-semibold">{r.algorithm}</p>
                          <p className="text-[10px] text-slate-400 leading-snug">{r.reason}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}