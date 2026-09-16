/**
 * Federo Health — Backend API Client
 * ====================================
 * Thin fetch wrapper for all calls to the FastAPI backend.
 * The Vite dev server proxies /api/* → http://localhost:8000 so
 * no absolute URLs or CORS issues in development.
 */

const BASE = '/api';

async function request(method, path, { body, isFormData = false } = {}) {
  const opts = {
    method,
    headers: isFormData ? undefined : { 'Content-Type': 'application/json' },
    body: body
      ? isFormData
        ? body
        : JSON.stringify(body)
      : undefined,
  };

  const res = await fetch(`${BASE}${path}`, opts);
  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`API ${method} ${path} failed (${res.status}): ${errorText}`);
  }
  return res.json();
}

// ── Upload ────────────────────────────────────────────────────────────────────
/**
 * Upload a CSV file to the backend.
 * @param {File}   file     - the File object from the file input
 * @param {string} useCase  - 'sepsis' | 'retinopathy'
 * @returns {Promise<{ job_id, file_name, row_count, use_case }>}
 */
export async function uploadDataset(file, useCase = 'sepsis') {
  const fd = new FormData();
  fd.append('file', file);
  fd.append('use_case', useCase);
  return request('POST', '/upload', { body: fd, isFormData: true });
}

// ── Preprocess ────────────────────────────────────────────────────────────────
/**
 * Run cleaning pipeline on the uploaded job.
 * @param {string} jobId
 * @returns {Promise<QualityReport>}
 */
export async function preprocessDataset(jobId) {
  return request('POST', `/preprocess/${jobId}`);
}

// ── Train ─────────────────────────────────────────────────────────────────────
/**
 * Start model training for a job.
 * @param {string} jobId
 * @param {object} hospitalInfo  - { id, name, tier, pub_key_fingerprint }
 * @param {string} useCase
 * @param {string} computeMode  - 'cloud' | 'edge'
 * @param {string[]} regulatoryTags
 * @returns {Promise<{ status, job_id, model_id }>}
 */
export async function startTraining(jobId, hospitalInfo, useCase, computeMode = 'cloud', regulatoryTags = [], rawCsv = '') {
  return request('POST', `/train/${jobId}`, {
    body: {
      hospital_id: hospitalInfo.id,
      hospital_name: hospitalInfo.name,
      hospital_tier: hospitalInfo.tier,
      pub_key_fingerprint: hospitalInfo.pubKeyFingerprint,
      use_case: useCase,
      compute_mode: computeMode,
      regulatory_tags: regulatoryTags,
      raw_csv: rawCsv || undefined,
    },
  });
}

// ── Training Status ───────────────────────────────────────────────────────────
/**
 * Poll training progress.
 * @param {string} jobId
 * @returns {Promise<{ status, current_epoch, total_epochs, history, model_id }>}
 */
export async function pollTrainingStatus(jobId) {
  return request('GET', `/train/${jobId}/status`);
}

// ── Model Card ────────────────────────────────────────────────────────────────
/**
 * Fetch the full Model Card JSON by job_id.
 * @param {string} jobId
 * @returns {Promise<ModelCard>}
 */
export async function getModelCardByJob(jobId) {
  return request('GET', `/model/job/${jobId}/card`);
}

/**
 * Fetch the full Model Card JSON by model_id.
 * @param {string} modelId
 * @returns {Promise<ModelCard>}
 */
export async function getModelCard(modelId) {
  return request('GET', `/model/${modelId}/card`);
}

// ── Federate ──────────────────────────────────────────────────────────────────
/**
 * Run one round of equity-weighted FedAvg for a use case.
 * @param {string} useCase
 * @returns {Promise<FederateResponse>}
 */
export async function federateModels(useCase) {
  return request('POST', `/federate/${useCase}`);
}

// ── Security Lab ─────────────────────────────────────────────────────────────
/**
 * Run poisoning defense benchmark.
 * @param {object} cfg  { use_case, rounds, malicious_hospitals, attack_type, poison_ratio, attack_scale, dp_noise, seeds }
 */
export async function runPoisoningScenario(cfg = {}) {
  return request('POST', '/security/poisoning', { body: cfg });
}

/**
 * Run privacy membership-inference attack.
 * @param {object} cfg  { use_case, epochs_list, dp_noise, seeds }
 */
export async function runPrivacyAttack(cfg = {}) {
  return request('POST', '/security/privacy', { body: cfg });
}

/**
 * Run FedAvg vs equity-FedAvg stress test.
 * @param {object} cfg  { use_case, rounds, malicious_hospitals, attack_type, poison_ratio, attack_scale, seeds }
 */
export async function runStressTest(cfg = {}) {
  return request('POST', '/security/stress', { body: cfg });
}

/**
 * Run FedAvg vs compressed FedAvg study (sparsify + quantize).
 * @param {object} cfg  { use_case, rounds, top_k_frac, n_bits, seeds }
 */
export async function runCompressionStudy(cfg = {}) {
  return request('POST', '/security/compression', { body: cfg });
}

/**
 * Run leave-one-hospital-out dataset shift analysis.
 * @param {object} cfg  { use_case, seeds }
 */
export async function runDatasetShift(cfg = {}) {
  return request('POST', '/security/dataset-shift', { body: cfg });
}

/**
 * Fetch real per-hospital governance trust scores.
 * @param {string} useCase
 */
export async function getGovernance(useCase = 'sepsis') {
  return request('GET', `/security/governance?use_case=${encodeURIComponent(useCase)}`);
}

/**
 * Run leave-one-hospital-out contribution valuation (Shapley-style marginal benefit).
 * @param {object} cfg  { use_case, seeds }
 */
export async function runContributionValuation(cfg = {}) {
  return request('POST', '/security/contribution', { body: cfg });
}

/**
 * Run real secure aggregation (Bonawitz pairwise additive masking).
 * @param {object} cfg  { use_case, rounds, agg, seeds }
 */
export async function runSecureAggregation(cfg = {}) {
  return request('POST', '/security/secure-agg', { body: cfg });
}

/**
 * Run local vs federated vs personalized (Ditto/FedPer).
 * @param {object} cfg  { use_case, rounds, epochs }
 */
export async function runPersonalization(cfg = {}) {
  return request('POST', '/security/personalization', { body: cfg });
}

/**
 * Run async / straggler-tolerant aggregation (FedAsync-style).
 * @param {object} cfg  { use_case, rounds, agg, seeds, straggler_penalty }
 */
export async function runAsyncStudy(cfg = {}) {
  return request('POST', '/security/async', { body: cfg });
}
// ── Helpers ───────────────────────────────────────────────────────────────────
/**
 * Convert a backend QualityReport to the shape DataUploadPanel already expects.
 * Bridges snake_case backend → camelCase frontend.
 */
export function adaptQualityReport(backendReport) {
  return {
    rawHeaderCount: backendReport.raw_header_count,
    initialRecordCount: backendReport.initial_record_count,
    cleanedRecordCount: backendReport.cleaned_record_count,
    duplicatesRemoved: backendReport.duplicates_removed,
    droppedColumns: backendReport.dropped_columns,
    keptHeaders: backendReport.kept_headers,
    cleanedHeaders: backendReport.kept_headers,
    colMissingSummary: Object.fromEntries(
      Object.entries(backendReport.col_missing_summary || {}).map(([k, v]) => [
        k,
        { missingRate: v.missing_rate, action: v.action },
      ])
    ),
    outlierSummary: Object.fromEntries(
      Object.entries(backendReport.outlier_summary || {}).map(([k, v]) => [
        k,
        { flagged: v.flagged, capped: v.capped },
      ])
    ),
    fhirMapping: (backendReport.fhir_mapping || []).map(m => ({
      column: m.column,
      rawName: m.raw_name,
      type: m.col_type,
      isMapped: m.is_mapped,
      targetConcept: m.target_concept,
      loincCode: m.loinc_code,
      loincDisplay: m.loinc_display,
    })),
    scores: {
      compositeQualityScore: backendReport.scores?.composite_quality_score ?? 90,
    },
    numericFeatureNames: backendReport.numeric_feature_names,
    demographicColumns: backendReport.demographic_columns,
    cleanedCsvText: backendReport.cleaned_csv,
    engineReadyRecords: [], // backend handles training directly
  };
}

/**
 * Convert a backend ModelCard to the shape ModelCardModal / Registry already expect.
 */
export function adaptModelCard(backendCard) {
  if (!backendCard) return null;
  return {
    modelId: backendCard.model_id,
    version: backendCard.version,
    name: backendCard.name,
    useCase: backendCard.use_case,
    createdAt: backendCard.created_at,
    status: backendCard.status,
    trainingMode: backendCard.training_mode,
    provenance: {
      hospitalId: backendCard.provenance?.hospital_id,
      hospitalName: backendCard.provenance?.hospital_name,
      hospitalTier: backendCard.provenance?.hospital_tier,
      country: backendCard.provenance?.country,
      pubKeyFingerprint: backendCard.provenance?.pub_key_fingerprint,
    },
    compliance: backendCard.compliance,
    datasetProfile: {
      totalRecords: backendCard.dataset_profile?.total_records,
      dataQualityScore: backendCard.dataset_profile?.data_quality_score,
      duplicatesRemoved: backendCard.dataset_profile?.duplicates_removed,
      outliersHandled: backendCard.dataset_profile?.outliers_handled,
      fhirMappedColumns: backendCard.dataset_profile?.fhir_mapped_columns,
    },
    modelArchitecture: {
      selectedAlgorithm: backendCard.model_architecture?.selected_algorithm,
      family: backendCard.model_architecture?.family,
      selectionRationale: backendCard.model_architecture?.selection_rationale,
      networkLayers: backendCard.model_architecture?.network_layers,
      hyperparameters: backendCard.model_architecture?.hyperparameters,
      trainingDuration: backendCard.model_architecture?.training_duration,
    },
    performance: {
      accuracy: backendCard.performance?.accuracy,
      precision: backendCard.performance?.precision,
      recall: backendCard.performance?.recall,
      f1Score: backendCard.performance?.f1_score,
      auroc: backendCard.performance?.auroc,
      confusionMatrix: backendCard.performance?.confusion_matrix
        ? {
            tp: backendCard.performance.confusion_matrix.tp,
            fp: backendCard.performance.confusion_matrix.fp,
            tn: backendCard.performance.confusion_matrix.tn,
            fn: backendCard.performance.confusion_matrix.fn,
            total: backendCard.performance.confusion_matrix.total,
          }
        : { tp: 0, fp: 0, tn: 0, fn: 0, total: 0 },
    },
    featureImportances: (backendCard.feature_importances || []).map(fi => ({
      feature: fi.feature,
      importance: fi.importance,
      normalizedScore: fi.normalized_score,
    })),
    trainingCurve: (backendCard.training_curve || []).map(e => ({
      epoch: e.epoch,
      loss: e.loss,
      accuracy: e.accuracy,
    })),
    privacy: backendCard.privacy
      ? {
          epsilon: backendCard.privacy.epsilon,
          delta: backendCard.privacy.delta,
          mechanism: backendCard.privacy.mechanism,
        }
      : {},
    weightsHash: backendCard.weights_hash,
    signatureInfo: backendCard.signature_info
      ? {
          signature: backendCard.signature_info.signature,
          signedAt: backendCard.signature_info.signed_at,
          signerPublicKeyFingerprint: backendCard.signature_info.signer_public_key_fingerprint,
          algorithm: backendCard.signature_info.algorithm,
          provenanceVerified: backendCard.signature_info.provenance_verified,
        }
      : {},
    federationImpact: backendCard.federation_impact
      ? {
          localAccuracy: backendCard.federation_impact.local_accuracy,
          federatedAccuracy: backendCard.federation_impact.federated_accuracy,
          delta: backendCard.federation_impact.delta,
        }
      : null,
    fairnessAudit: backendCard.fairness_audit
      ? {
          demographicColumns: backendCard.fairness_audit.demographic_columns,
          subgroupResults: (backendCard.fairness_audit.subgroup_results || []).map(sr => ({
            subgroup: sr.subgroup,
            column: sr.column,
            accuracy: sr.accuracy,
            f1Score: sr.f1_score,
            count: sr.count,
            flagged: sr.flagged,
          })),
          overallAccuracy: backendCard.fairness_audit.overall_accuracy,
          hasDemographicData: backendCard.fairness_audit.has_demographic_data,
        }
      : null,
    aggregationMethod: backendCard.aggregation_method,
    analysis: backendCard.analysis || null,
  };
}
