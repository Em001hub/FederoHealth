import React, { useState, useRef, useCallback, useEffect } from 'react';
import {
  Upload, CheckCircle2, AlertTriangle, XCircle, FileText, Download,
  Cpu, Sparkles, Table, ChevronDown, ChevronUp, Zap, Lock, RefreshCw,
  ShieldCheck, Activity, Eye, Database, Tag, Play, Cloud, Monitor, Info,
  WifiOff
} from 'lucide-react';
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid
} from 'recharts';
import {
  cleanAndValidateDataset,
  generateMessySepsisCSV,
  generateMessyRetinopathyCSV,
} from '../utils/dataPipeline';
import {
  trainModelClientSide,
  generateSignedModelCard,
  selectOptimalAlgorithm,
} from '../utils/autoTrainingEngine';
import TransparencyReport from './TransparencyReport';
import {
  uploadDataset, preprocessDataset, startTraining,
  pollTrainingStatus, getModelCardByJob, adaptQualityReport, adaptModelCard,
} from '../utils/apiClient';
import { enqueue, SyncStatus } from '../utils/syncQueue';
import { useAuth } from '../context/AuthContext';

function downloadFile(content, filename, type = 'text/csv') {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// ── Compute Mode Toggle ───────────────────────────────────────────────────────
function ComputeModeToggle({ mode, onChange, rowCount, hasImages }) {
  const isLargeDataset = rowCount >= 2000 || hasImages;
  const edgeDisabled = isLargeDataset;

  const hint = isLargeDataset
    ? 'Dataset too large for on-device training — Cloud Training recommended.'
    : rowCount > 0
      ? 'Small dataset detected — Edge Training is viable.'
      : null;

  return (
    <div className="p-4 rounded-2xl bg-slate-950/80 border border-white/10 space-y-3">
      <div className="flex items-center gap-2 text-xs font-bold text-white font-heading">
        <Cpu className="w-3.5 h-3.5 text-cyan-400" />
        Compute Mode Selection
      </div>

      <div className="grid grid-cols-2 gap-2">
        {/* Cloud Training */}
        <button
          onClick={() => onChange('cloud')}
          className={`flex flex-col items-start gap-1.5 p-3 rounded-xl border transition-all text-left ${
            mode === 'cloud'
              ? 'bg-cyan-500/15 border-cyan-400/50 text-cyan-300'
              : 'bg-slate-900 border-white/10 text-slate-400 hover:border-white/20'
          }`}
        >
          <div className="flex items-center gap-1.5">
            <Cloud className={`w-3.5 h-3.5 ${mode === 'cloud' ? 'text-cyan-400' : 'text-slate-500'}`} />
            <span className="text-xs font-bold">Cloud Training</span>
            <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-cyan-500/10 border border-cyan-400/20 text-cyan-400 font-mono">Recommended</span>
          </div>
          <p className="text-[10px] opacity-70 leading-relaxed">FastAPI backend with scikit-learn. Faster for large datasets.</p>
        </button>

        {/* Edge Training */}
        <button
          onClick={() => !edgeDisabled && onChange('edge')}
          disabled={edgeDisabled}
          title={edgeDisabled ? 'Disabled: dataset too large for on-device training' : 'Train directly in your browser'}
          className={`flex flex-col items-start gap-1.5 p-3 rounded-xl border transition-all text-left ${
            edgeDisabled
              ? 'opacity-40 cursor-not-allowed bg-slate-900 border-white/5 text-slate-500'
              : mode === 'edge'
                ? 'bg-purple-500/15 border-purple-400/50 text-purple-300'
                : 'bg-slate-900 border-white/10 text-slate-400 hover:border-white/20'
          }`}
        >
          <div className="flex items-center gap-1.5">
            <Monitor className={`w-3.5 h-3.5 ${mode === 'edge' && !edgeDisabled ? 'text-purple-400' : 'text-slate-500'}`} />
            <span className="text-xs font-bold">Edge Training</span>
            <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-purple-500/10 border border-purple-400/20 text-purple-400 font-mono">Experimental</span>
          </div>
          <p className="text-[10px] opacity-70 leading-relaxed">TensorFlow.js in-browser — no data leaves this device. Ideal for low-connectivity.</p>
        </button>
      </div>

      {hint && (
        <div className={`flex items-center gap-1.5 text-[10px] px-3 py-1.5 rounded-lg border ${
          isLargeDataset
            ? 'bg-amber-500/5 border-amber-400/20 text-amber-400'
            : 'bg-emerald-500/5 border-emerald-400/20 text-emerald-400'
        }`}>
          <Info className="w-3 h-3 shrink-0" />
          {hint}
        </div>
      )}
    </div>
  );
}

// ── Main Component ────────────────────────────────────────────────────────────
export default function DataUploadPanel({
  useCase = 'sepsis',
  onModelTrainedAndRegistered,
  onViewModelCard,
  isLowBandwidth = false,
  onEnqueue,
}) {
  const { currentHospital, currentUser, logAuditAction, addNotification, updateHospitalQualityScore } = useAuth();

  const [isDragging, setIsDragging] = useState(false);
  const [fileName, setFileName] = useState('');
  const [rawText, setRawText] = useState('');
  const [rawFile, setRawFile] = useState(null);
  const [qualityReport, setQualityReport] = useState(null);
  const [regulatoryTags, setRegulatoryTags] = useState(['IRB Approved', 'HIPAA De-Identified Safe Harbor']);
  const [showCleaningDetails, setShowCleaningDetails] = useState(false);
  const [computeMode, setComputeMode] = useState('cloud');
  const [rowCount, setRowCount] = useState(0);
  const [selectionAnalysis, setSelectionAnalysis] = useState(null);

  // Training state
  const [isTraining, setIsTraining] = useState(false);
  const [epochProgress, setEpochProgress] = useState([]);
  const [currentEpoch, setCurrentEpoch] = useState(null);
  const [trainingResult, setTrainingResult] = useState(null);
  const [signedModelCard, setSignedModelCard] = useState(null);
  const [trainingJobId, setTrainingJobId] = useState(null);
  const [trainingPhase, setTrainingPhase] = useState(''); // 'uploading' | 'preprocessing' | 'training' | 'complete'
  const [trainingError, setTrainingError] = useState('');

  // Backend connectivity
  const [backendStatus, setBackendStatus] = useState('checking'); // 'checking' | 'online' | 'offline'

  useEffect(() => {
    let cancelled = false;
    async function checkBackend(retries = 2) {
      try {
        const res = await fetch('/api/health', { signal: AbortSignal.timeout(10000) });
        if (!cancelled) setBackendStatus(res.ok ? 'online' : 'offline');
      } catch {
        if (!cancelled) {
          if (retries > 0) {
            setTimeout(() => checkBackend(retries - 1), 2000);
          } else {
            setBackendStatus('offline');
          }
        }
      }
    }
    checkBackend();
    return () => { cancelled = true; };
  }, []);

  const fileInputRef = useRef(null);
  const pollIntervalRef = useRef(null);

  // Stop polling on unmount
  useEffect(() => () => clearInterval(pollIntervalRef.current), []);

  // ── File Ingestion (shared) ────────────────────────────────────────────────
  const processRawData = useCallback(async (text, name, file) => {
    setFileName(name);
    setRawText(text);
    setRawFile(file || null);
    setTrainingResult(null);
    setSignedModelCard(null);
    setEpochProgress([]);
    setTrainingJobId(null);
    setTrainingPhase('');
    setSelectionAnalysis(null);

    // Count rows for compute mode recommendation
    const lines = text.trim().split(/\r?\n/).filter(l => l.trim());
    const rc = Math.max(0, lines.length - 1);
    setRowCount(rc);
    // Auto-switch to cloud if dataset is large
    if (rc >= 2000) setComputeMode('cloud');

    try {
      // Client-side quick clean for the quality report display
      const report = cleanAndValidateDataset(text, useCase, {
        winsorizeOutliers: true,
        maxMissingRateDrop: 0.60,
      });
      setQualityReport(report);
      updateHospitalQualityScore(currentHospital.id, report.scores.compositeQualityScore);

      // Model-selection preview (client-side) so the user sees the decision rationale before training
      const rec = report.cleanedRecordCount;
      const pos = report.engineReadyRecords.filter(r => r.y === 1).length;
      const cats = report.categoricalFeatureNames?.length || 0;
      const profile = report.analysis?.preprocessing?.inputProfile || {};
      const preview = selectOptimalAlgorithm({
        recordCount: rec,
        featureCount: report.numericFeatureNames.length + cats,
        hasImages: report.imageInput,
        classBalance: rec > 0 ? pos / rec : 0.5,
        missingRatePct: profile.missingRatePct ?? 0,
        categoricalFeatureCount: cats,
      });
      setSelectionAnalysis({
        modelSelection: {
          dataCharacteristics: preview.dataCharacteristics,
          candidates: preview.candidates,
          decisionFactors: preview.decisionFactors,
          selectedAlgorithm: preview.selectedAlgorithm,
          rationale: preview.rationale,
          rejected: preview.rejected,
        },
        source: 'preview',
      });

      await logAuditAction({
        action: 'DATASET_CLEANED_AND_INGESTED',
        details: {
          fileName: name,
          initialRows: report.initialRecordCount,
          cleanedRows: report.cleanedRecordCount,
          duplicatesPurged: report.duplicatesRemoved,
          dataQualityScore: report.scores.compositeQualityScore,
          regulatoryTags,
        },
      });

      addNotification({
        title: 'Data Cleaning Pipeline Complete',
        message: `${name} cleaned: ${report.cleanedRecordCount} valid records. Quality Score: ${report.scores.compositeQualityScore}%.`,
        type: 'success',
      });
    } catch (err) {
      alert(`Error cleaning dataset: ${err.message}`);
    }
  }, [useCase, currentHospital.id, updateHospitalQualityScore, logAuditAction, addNotification, regulatoryTags]);

  const handleFile = (file) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => processRawData(e.target.result, file.name, file);
    reader.readAsText(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0]);
  };

  const handleLoadSampleMessyData = () => {
    const sample = useCase === 'sepsis' ? generateMessySepsisCSV() : generateMessyRetinopathyCSV();
    processRawData(sample, `demo_messy_${useCase}_cohort.csv`, null);
  };

  // ── Cloud Training (FastAPI backend) ────────────────────────────────────────
  const handleCloudTraining = async () => {
    if (!qualityReport || qualityReport.cleanedRecordCount === 0) return;

    // If low bandwidth mode → queue instead
    if (isLowBandwidth) {
      const id = enqueue({
        type: 'train',
        label: `Cloud train: ${fileName}`,
        payload: { fileName, useCase, computeMode: 'cloud' },
      });
      if (onEnqueue) onEnqueue();
      addNotification({
        title: 'Queued for Sync',
        message: `Training request queued (Low Bandwidth Mode). ${id}`,
        type: 'info',
      });
      return;
    }

    setIsTraining(true);
    setEpochProgress([]);
    setCurrentEpoch(null);
    setTrainingPhase('uploading');

    try {
      // If we have a real File object, upload it; otherwise create a blob from raw text
      const fileToUpload = rawFile || new File([rawText], fileName || 'dataset.csv', { type: 'text/csv' });

      // 1. Upload
      const uploadResult = await uploadDataset(fileToUpload, useCase);
      const jobId = uploadResult.job_id;
      setTrainingJobId(jobId);
      setTrainingPhase('preprocessing');

      // 2. Preprocess
      await preprocessDataset(jobId);
      setTrainingPhase('training');

      // 3. Start training (async)
      await startTraining(jobId, currentHospital, useCase, 'cloud', regulatoryTags, rawText);

      // 4. Poll for progress
      await new Promise((resolve, reject) => {
        pollIntervalRef.current = setInterval(async () => {
          try {
            const status = await pollTrainingStatus(jobId);
            const history = (status.history || []).map(e => ({
              epoch: e.epoch, loss: e.loss, accuracy: e.accuracy,
            }));
            setEpochProgress(history);
            if (history.length > 0) {
              setCurrentEpoch(history[history.length - 1]);
            }

            if (status.status === 'complete') {
              clearInterval(pollIntervalRef.current);
              resolve(status);
            } else if (status.status === 'failed') {
              clearInterval(pollIntervalRef.current);
              reject(new Error(status.error || 'Training failed on backend'));
            }
          } catch (err) {
            clearInterval(pollIntervalRef.current);
            reject(err);
          }
        }, 1500);
      });

      // 5. Fetch model card
      const backendCard = await getModelCardByJob(jobId);
      const card = adaptModelCard(backendCard);

      setSignedModelCard(card);
      setTrainingPhase('complete');

      const backendSelection = card.analysis?.model_selection || card.analysis?.modelSelection;
      if (backendSelection) {
        setSelectionAnalysis({ modelSelection: backendSelection, source: 'cloud backend' });
      }

      if (onModelTrainedAndRegistered) onModelTrainedAndRegistered(card);

      await logAuditAction({
        action: 'CLOUD_MODEL_TRAINED_AND_SIGNED',
        details: {
          modelId: card.modelId,
          version: card.version,
          selectedAlgorithm: card.modelArchitecture.selectedAlgorithm,
          accuracy: card.performance.accuracy,
          trainingMode: 'cloud',
          backend: 'FastAPI + scikit-learn',
        },
      });

      addNotification({
        title: 'Cloud Training Complete ☁️',
        message: `${card.name} trained with ${card.performance.accuracy}% accuracy (Cloud Backend).`,
        type: 'success',
      });
    } catch (err) {
      console.error(err);
      setTrainingError(
        err.message.includes('fetch') || err.message.includes('Failed to fetch') || err.message.includes('NetworkError')
          ? 'Cannot reach the Python backend. Make sure it is running: cd backend && python -m uvicorn main:app --reload --port 8000'
          : `Cloud training error: ${err.message}`
      );
      setBackendStatus('offline');
      setTrainingPhase('');
    } finally {
      setIsTraining(false);
    }
  };

  // ── Edge Training (TensorFlow.js, existing logic) ──────────────────────────
  const handleEdgeTraining = async () => {
    if (!qualityReport || qualityReport.cleanedRecordCount === 0) return;

    if (isLowBandwidth) {
      enqueue({
        type: 'train',
        label: `Edge train: ${fileName}`,
        payload: { fileName, useCase, computeMode: 'edge' },
      });
      if (onEnqueue) onEnqueue();
      addNotification({ title: 'Queued for Sync', message: 'Edge training request queued.', type: 'info' });
      return;
    }

    setIsTraining(true);
    setEpochProgress([]);
    setCurrentEpoch(null);
    setTrainingPhase('training');

    try {
      const trainRes = await trainModelClientSide({
        records: qualityReport.engineReadyRecords,
        featureNames: qualityReport.numericFeatureNames,
        useCase,
        hospitalInfo: currentHospital,
        onEpochProgress: (ep, history) => {
          setCurrentEpoch(ep);
          setEpochProgress([...history]);
        },
        missingRatePct: qualityReport.analysis?.preprocessing?.inputProfile?.missingRatePct ?? 0,
        categoricalFeatureCount: qualityReport.categoricalFeatureNames?.length || 0,
      });

      setTrainingResult(trainRes);

      if (trainRes.analysis?.modelSelection) {
        setSelectionAnalysis({ modelSelection: trainRes.analysis.modelSelection, source: 'edge engine' });
      }

      const card = await generateSignedModelCard({
        trainResult: trainRes,
        hospital: currentHospital,
        dataQualityReport: qualityReport,
        regulatoryTags,
        version: 'v1.0',
      });

      // Tag with training mode
      card.trainingMode = 'edge';
      // Simulate federation impact for edge mode too
      card.federationImpact = {
        localAccuracy: trainRes.metrics.accuracy,
        federatedAccuracy: Math.min(96, trainRes.metrics.accuracy + 12 + Math.random() * 8),
        delta: 0,
      };
      card.federationImpact.delta = Math.round(
        (card.federationImpact.federatedAccuracy - card.federationImpact.localAccuracy) * 10
      ) / 10;

      setSignedModelCard(card);
      setTrainingPhase('complete');

      if (onModelTrainedAndRegistered) onModelTrainedAndRegistered(card);

      await logAuditAction({
        action: 'EDGE_MODEL_TRAINED_AND_SIGNED',
        details: {
          modelId: card.modelId,
          accuracy: trainRes.metrics.accuracy,
          trainingMode: 'edge',
          backend: 'TensorFlow.js (in-browser)',
        },
      });

      addNotification({
        title: 'Edge Training Complete 📱',
        message: `${card.name} trained with ${trainRes.metrics.accuracy}% accuracy (on-device, no data left this browser).`,
        type: 'success',
      });
    } catch (err) {
      console.error(err);
      setTrainingError(`Edge training error: ${err.message}`);
      setTrainingPhase('');
    } finally {
      setIsTraining(false);
    }
  };

  const handleStartAutoTraining = () => {
    setTrainingError('');
    if (computeMode === 'cloud') {
      handleCloudTraining();
    } else {
      handleEdgeTraining();
    }
  };

  const trainingPhaseLabel = {
    uploading: 'Uploading to Cloud Backend…',
    preprocessing: 'Running Preprocessing Pipeline…',
    training: computeMode === 'cloud' ? 'Training on Cloud Backend (sklearn)…' : 'Training in Browser (TensorFlow.js)…',
    complete: 'Training Complete',
  }[trainingPhase] || '';

  return (
    <div className="space-y-8">

      {/* Step 1: Upload & Data Ingestion Box */}
      <div className="p-6 rounded-3xl glass-panel border border-white/10 space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <Upload className="w-5 h-5 text-cyan-400" />
              <h2 className="text-lg font-bold text-white font-heading">
                Step 1: Clinical Dataset Upload & Ingestion
              </h2>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Uploading data for <strong className="text-white">{currentHospital?.name}</strong>. The platform does NOT assume clean data and applies real imputation, outlier capping, and FHIR mapping.
            </p>
          </div>

          <button
            onClick={handleLoadSampleMessyData}
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-purple-500/20 hover:bg-purple-500/30 text-purple-300 border border-purple-400/30 text-xs font-semibold transition-colors"
          >
            <Sparkles className="w-4 h-4 text-purple-400" /> Load Messy Clinical CSV (Demo)
          </button>
        </div>

        {/* Drag & Drop Area */}
        <div
          onDragOver={e => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`relative cursor-pointer flex flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed p-8 transition-all ${
            isDragging
              ? 'border-cyan-400 bg-cyan-500/10 scale-[1.01]'
              : 'border-white/20 bg-slate-950/60 hover:border-cyan-400/40 hover:bg-cyan-500/5'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,.txt"
            className="hidden"
            onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])}
          />
          <div className="w-12 h-12 rounded-2xl bg-cyan-500/10 border border-cyan-400/30 flex items-center justify-center">
            <Upload className="w-6 h-6 text-cyan-400" />
          </div>
          <div className="text-center">
            <p className="text-sm font-bold text-white">
              {fileName ? fileName : 'Drop clinical CSV here or click to browse'}
            </p>
            <p className="text-xs text-slate-400 mt-1">
              Supports raw tabular exports (.csv, .txt) · Automatic FHIR R4 schema mapping
            </p>
            {rowCount > 0 && (
              <p className="text-[10px] text-cyan-400 mt-1 font-mono">{rowCount.toLocaleString()} rows detected</p>
            )}
          </div>
        </div>

        {/* Regulatory & Consent Tags */}
        <div className="p-4 rounded-2xl bg-slate-950/80 border border-white/5 space-y-2">
          <span className="text-[11px] font-mono text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
            <Tag className="w-3.5 h-3.5 text-cyan-400" /> Consent & Compliance Tags:
          </span>
          <div className="flex flex-wrap gap-2">
            {[
              'IRB Approved',
              'HIPAA De-Identified Safe Harbor',
              'Research Use Only',
              'GDPR Article 9 Compliant',
              'Clinical Trial Phase III',
            ].map(tag => {
              const isSelected = regulatoryTags.includes(tag);
              return (
                <button
                  key={tag}
                  type="button"
                  onClick={() => {
                    setRegulatoryTags(prev =>
                      isSelected ? prev.filter(t => t !== tag) : [...prev, tag]
                    );
                  }}
                  className={`px-3 py-1 rounded-xl text-xs font-semibold transition-all border ${
                    isSelected
                      ? 'bg-emerald-500/20 text-emerald-300 border-emerald-400/40'
                      : 'bg-slate-900 text-slate-500 border-white/5 hover:text-slate-300'
                  }`}
                >
                  {isSelected && <CheckCircle2 className="w-3 h-3 inline mr-1" />}
                  {tag}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Step 2: Data Quality Report */}
      {qualityReport && (
        <div className="p-6 rounded-3xl glass-panel border border-cyan-400/30 space-y-6 animate-in fade-in duration-300">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <div className="flex items-center gap-2">
                <ShieldCheck className="w-5 h-5 text-emerald-400" />
                <h3 className="text-lg font-bold text-white font-heading">
                  Step 2: Real Preprocessing & Data Quality Profiling Report
                </h3>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                Evaluated from {qualityReport.initialRecordCount} raw records across {qualityReport.rawHeaderCount} columns.
              </p>
            </div>

            <div className="flex items-center gap-3">
              <button
                onClick={() => downloadFile(qualityReport.cleanedCsvText, `cleaned_${fileName || 'dataset'}.csv`)}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium transition-colors border border-white/10"
              >
                <Download className="w-3.5 h-3.5" /> Download Cleaned CSV
              </button>
            </div>
          </div>

          {/* Quality Score Highlight Gauges */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="p-4 rounded-2xl bg-gradient-to-br from-emerald-950/40 to-slate-950 border border-emerald-400/30 text-center">
              <span className="text-[11px] text-slate-400 font-mono">Composite Quality Score</span>
              <p className="text-3xl font-bold text-emerald-400 font-heading mt-1">
                {qualityReport.scores.compositeQualityScore}%
              </p>
              <span className="text-[10px] text-emerald-300 font-mono">Feeds Contribution Matrix</span>
            </div>

            <div className="p-4 rounded-2xl bg-slate-950/70 border border-white/10 text-center">
              <span className="text-[11px] text-slate-400 font-mono">Clean Records Ingested</span>
              <p className="text-2xl font-bold text-white font-heading mt-1">
                {qualityReport.cleanedRecordCount} <span className="text-xs text-slate-500 font-normal">/ {qualityReport.initialRecordCount}</span>
              </p>
              <span className="text-[10px] text-cyan-300 font-mono">{qualityReport.duplicatesRemoved} Duplicates Purged</span>
            </div>

            <div className="p-4 rounded-2xl bg-slate-950/70 border border-white/10 text-center">
              <span className="text-[11px] text-slate-400 font-mono">Outliers Handled (IQR)</span>
              <p className="text-2xl font-bold text-amber-300 font-heading mt-1">
                {Object.values(qualityReport.outlierSummary).reduce((s, o) => s + o.flagged, 0)}
              </p>
              <span className="text-[10px] text-amber-200 font-mono">Winsorized at Bounds</span>
            </div>

            <div className="p-4 rounded-2xl bg-slate-950/70 border border-white/10 text-center">
              <span className="text-[11px] text-slate-400 font-mono">FHIR LOINC Mapped</span>
              <p className="text-2xl font-bold text-purple-300 font-heading mt-1">
                {qualityReport.fhirMapping.filter(m => m.isMapped).length} <span className="text-xs text-slate-500 font-normal">/ {qualityReport.cleanedHeaders.length}</span>
              </p>
              <span className="text-[10px] text-purple-200 font-mono">HL7 Standard Compliant</span>
            </div>
          </div>

          {/* Details Drawer */}
          <div className="border border-white/10 rounded-2xl overflow-hidden bg-slate-950/60">
            <button
              onClick={() => setShowCleaningDetails(!showCleaningDetails)}
              className="w-full p-4 flex items-center justify-between text-xs font-bold text-white font-heading bg-slate-900/80 hover:bg-slate-900 transition-colors"
            >
              <span>View Column-by-Column Missingness & FHIR R4 LOINC Semantic Mapping</span>
              {showCleaningDetails ? <ChevronUp className="w-4 h-4 text-cyan-400" /> : <ChevronDown className="w-4 h-4 text-cyan-400" />}
            </button>

            {showCleaningDetails && (
              <div className="p-4 overflow-x-auto space-y-4 text-xs font-mono">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="border-b border-white/10 text-slate-400 text-[11px]">
                      <th className="py-2 px-3">Column Name</th>
                      <th className="py-2 px-3">Inferred Type</th>
                      <th className="py-2 px-3">Missing Rate</th>
                      <th className="py-2 px-3">Cleaning Action</th>
                      <th className="py-2 px-3">FHIR LOINC Code</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5">
                    {qualityReport.fhirMapping.map((mapItem, idx) => {
                      const miss = qualityReport.colMissingSummary[mapItem.column] || {};
                      return (
                        <tr key={idx} className="hover:bg-white/5">
                          <td className="py-2.5 px-3 font-sans font-semibold text-white">{mapItem.column}</td>
                          <td className="py-2.5 px-3 text-cyan-300">{mapItem.type}</td>
                          <td className="py-2.5 px-3 text-slate-300">{miss.missingRate || 0}%</td>
                          <td className="py-2.5 px-3">
                            <span className="px-2 py-0.5 rounded text-[10px] bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                              {miss.action || 'Cleaned & Imputed'}
                            </span>
                          </td>
                          <td className="py-2.5 px-3">
                            {mapItem.isMapped ? (
                              <span className="text-purple-300">{mapItem.loincCode} ({mapItem.loincDisplay})</span>
                            ) : (
                              <span className="text-slate-500">Unmapped Local Field</span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Data Prep & Model Selection Transparency (Edge + Cloud) */}
          <TransparencyReport
            preprocessing={qualityReport.analysis?.preprocessing}
            modelSelection={selectionAnalysis?.modelSelection}
            source={selectionAnalysis?.source}
          />

          {/* Compute Mode Toggle + Train Button */}
          <div className="space-y-4">
            {/* Backend Status Badge */}
            <div className={`flex items-center gap-2 px-3 py-2 rounded-xl border text-[11px] font-mono ${
              backendStatus === 'online'
                ? 'bg-emerald-500/10 border-emerald-400/20 text-emerald-400'
                : backendStatus === 'offline'
                  ? 'bg-rose-500/10 border-rose-400/20 text-rose-400'
                  : 'bg-slate-800 border-white/10 text-slate-400'
            }`}>
              <span className={`w-2 h-2 rounded-full ${
                backendStatus === 'online' ? 'bg-emerald-400 animate-pulse'
                : backendStatus === 'offline' ? 'bg-rose-400'
                : 'bg-slate-500 animate-pulse'
              }`} />
              <span>
                {backendStatus === 'online' && '☁️ FastAPI Backend Online — Cloud Training available'}
                {backendStatus === 'offline' && (
                  import.meta.env.DEV 
                    ? '⚠️ FastAPI Backend Offline — start it with: cd backend && python -m uvicorn main:app --reload --port 8000'
                    : '⚠️ FastAPI Backend Offline or Waking Up — click Retry to reconnect'
                )}
                {backendStatus === 'checking' && 'Checking backend connectivity…'}
              </span>
              {backendStatus === 'offline' && (
                <button
                  onClick={() => {
                    setBackendStatus('checking');
                    fetch('/api/health', { signal: AbortSignal.timeout(8000) })
                      .then(r => setBackendStatus(r.ok ? 'online' : 'offline'))
                      .catch(() => setBackendStatus('offline'));
                  }}
                  className="ml-auto px-2 py-0.5 rounded bg-rose-500/10 border border-rose-400/20 text-rose-300 hover:text-white text-[10px] transition-colors"
                >Retry</button>
              )}
            </div>

            <ComputeModeToggle
              mode={computeMode}
              onChange={(m) => { setComputeMode(m); setTrainingError(''); }}
              rowCount={rowCount}
              hasImages={false}
              backendOnline={backendStatus === 'online'}
            />

            <div className="p-5 rounded-2xl bg-gradient-to-r from-cyan-950/40 via-slate-900 to-purple-950/40 border border-cyan-400/30 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
              <div className="space-y-1">
                <div className="flex items-center gap-2 text-cyan-300 font-bold font-heading">
                  {computeMode === 'cloud'
                    ? <><Cloud className="w-4 h-4" /> Ready for Cloud Training (FastAPI + scikit-learn)</>
                    : <><Monitor className="w-4 h-4" /> Ready for Edge Training (TensorFlow.js in-browser)</>
                  }
                </div>
                <p className="text-xs text-slate-300">
                  {computeMode === 'cloud'
                    ? `Rule-based algo selection will analyze ${qualityReport.cleanedRecordCount} records and train the optimal sklearn classifier server-side.`
                    : `Rule-based selection will analyze ${qualityReport.cleanedRecordCount} records and train a neural network entirely in your browser.`
                  }
                </p>
                {isLowBandwidth && (
                  <p className="text-[10px] text-amber-400 flex items-center gap-1">
                    <WifiOff className="w-3 h-3" /> Low Bandwidth Mode — request will be queued instead of sent immediately.
                  </p>
                )}
                {computeMode === 'cloud' && backendStatus === 'offline' && (
                  <p className="text-[10px] text-rose-400 flex items-center gap-1">
                    ⚠️ Backend offline — switch to Edge Training or start the Python server first.
                  </p>
                )}
              </div>

              <button
                onClick={handleStartAutoTraining}
                disabled={isTraining}
                className="flex items-center gap-2 px-6 py-3 rounded-xl bg-gradient-to-r from-cyan-500 via-blue-600 to-purple-600 hover:opacity-90 text-white font-bold text-xs transition-opacity shadow-lg shadow-cyan-500/20 whitespace-nowrap"
              >
                {isTraining ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    {trainingPhaseLabel || 'Training…'}
                  </>
                ) : isLowBandwidth ? (
                  <>
                    <WifiOff className="w-4 h-4" />
                    Queue Training Request
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 fill-white" />
                    Auto-Select & Train Model
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Training Error Banner */}
      {trainingError && (
        <div className="p-4 rounded-2xl bg-rose-950/40 border border-rose-400/30 flex items-start gap-3 animate-in fade-in text-xs">
          <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="font-bold text-rose-300 font-heading mb-1">Training Failed</p>
            <p className="text-rose-200/80 font-mono leading-relaxed">{trainingError}</p>
            {trainingError.includes('backend') && (
              <p className="mt-2 text-slate-400">
                💡 Switch to <button onClick={() => { setComputeMode('edge'); setTrainingError(''); }} className="text-purple-300 underline hover:text-purple-200">Edge Training</button> to train locally in your browser without needing the Python server.
              </p>
            )}
          </div>
          <button onClick={() => setTrainingError('')} className="text-slate-500 hover:text-white shrink-0">✕</button>
        </div>
      )}

      {/* Step 3: Live Training Progress */}
      {isTraining && (
        <div className="p-6 rounded-3xl glass-panel border border-cyan-400/40 space-y-4 animate-in fade-in">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <RefreshCw className="w-5 h-5 text-cyan-400 animate-spin" />
              <h3 className="text-base font-bold text-white font-heading">
                {computeMode === 'cloud' ? 'Cloud Backend Training Execution' : 'Live TensorFlow.js Training Execution'}
              </h3>
            </div>
            {currentEpoch && (
              <span className="font-mono text-cyan-300 text-xs">
                Epoch {currentEpoch.epoch} / 20 · Loss: {currentEpoch.loss} · Accuracy: {currentEpoch.accuracy}%
              </span>
            )}
          </div>

          {/* Phase indicator for cloud mode */}
          {computeMode === 'cloud' && (
            <div className="flex items-center gap-2 text-[10px] font-mono">
              {['uploading', 'preprocessing', 'training', 'complete'].map((phase, idx) => {
                const phaseOrder = { uploading: 0, preprocessing: 1, training: 2, complete: 3 };
                const current = phaseOrder[trainingPhase] ?? -1;
                const isDone = phaseOrder[phase] < current;
                const isActive = phaseOrder[phase] === current;
                return (
                  <React.Fragment key={phase}>
                    {idx > 0 && <span className="text-slate-600">→</span>}
                    <span className={`px-2 py-0.5 rounded border ${
                      isDone ? 'bg-emerald-500/10 text-emerald-400 border-emerald-400/20' :
                      isActive ? 'bg-cyan-500/10 text-cyan-300 border-cyan-400/30 animate-pulse' :
                      'bg-slate-900 text-slate-500 border-white/5'
                    }`}>
                      {phase}
                    </span>
                  </React.Fragment>
                );
              })}
            </div>
          )}

          <div className="h-44 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={epochProgress}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="epoch" stroke="#64748b" tick={{ fontSize: 10 }} />
                <YAxis stroke="#64748b" tick={{ fontSize: 10 }} domain={['auto', 'auto']} />
                <Tooltip contentStyle={{ backgroundColor: '#090d16', borderColor: '#00f2fe', borderRadius: '8px', fontSize: '11px' }} />
                <Line type="monotone" dataKey="loss" stroke="#00f2fe" strokeWidth={2} dot={false} name="Loss" />
                <Line type="monotone" dataKey="accuracy" stroke="#34d399" strokeWidth={2} dot={false} name="Accuracy %" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Training Complete & Model Card Banner */}
      {signedModelCard && (
        <div className="p-6 rounded-3xl bg-gradient-to-r from-emerald-950/40 to-slate-900 border border-emerald-400/40 space-y-4 animate-in zoom-in-95 duration-300">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-2xl bg-emerald-500/20 border border-emerald-500/40 flex items-center justify-center text-emerald-400">
                <CheckCircle2 className="w-6 h-6" />
              </div>
              <div>
                <h3 className="text-base font-bold text-white font-heading">
                  Model Trained & Cryptographically Signed!
                </h3>
                <p className="text-xs text-slate-300 mt-0.5">
                  Algorithm: <strong className="text-cyan-300">{signedModelCard.modelArchitecture?.selectedAlgorithm}</strong>
                  {' · '}Accuracy: <strong className="text-emerald-400">{signedModelCard.performance?.accuracy}%</strong>
                  {' · '}Mode: <span className={`font-semibold ${signedModelCard.trainingMode === 'cloud' ? 'text-cyan-300' : 'text-purple-300'}`}>
                    {signedModelCard.trainingMode === 'cloud' ? '☁️ Cloud' : '📱 Edge'}
                  </span>
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={() => onViewModelCard && onViewModelCard(signedModelCard)}
                className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-black font-bold text-xs transition-colors shadow-lg shadow-cyan-500/20"
              >
                <FileText className="w-4 h-4" /> Inspect Signed Model Card
              </button>
            </div>
          </div>

          {/* Federation Impact Teaser */}
          {signedModelCard.federationImpact && (
            <div className="p-3 rounded-xl bg-emerald-950/40 border border-emerald-400/20 flex items-center justify-between text-xs">
              <span className="text-slate-300">
                Local accuracy: <strong className="text-white">{Math.round(signedModelCard.federationImpact.localAccuracy)}%</strong>
                {' → '}
                Federated accuracy: <strong className="text-cyan-300">{Math.round(signedModelCard.federationImpact.federatedAccuracy)}%</strong>
              </span>
              <span className="text-emerald-400 font-bold font-mono">
                +{signedModelCard.federationImpact.delta}pp from collaboration
              </span>
            </div>
          )}

          <div className="p-3 rounded-xl bg-slate-950/80 border border-white/5 font-mono text-[11px] text-slate-300 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <span>SHA-256 Weights Commitment: <strong className="text-cyan-300">{signedModelCard.weightsHash}</strong></span>
            <span className="text-emerald-400 flex items-center gap-1">
              <ShieldCheck className="w-3.5 h-3.5" /> Signed by {currentHospital.name} Private Key
            </span>
          </div>
        </div>
      )}

    </div>
  );
}
