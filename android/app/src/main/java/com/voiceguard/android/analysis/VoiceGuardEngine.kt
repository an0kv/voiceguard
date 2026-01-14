package com.voiceguard.android.analysis

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

class VoiceGuardEngine(
    private val sampleRate: Int,
    private val emaAlpha: Float = 0.35f,
    private val vadThresholdDb: Float = -45.0f,
) {
    private val extractor = FeatureExtractor(sampleRate)
    private val emaSlowAlpha: Float = emaAlpha * 0.35f
    private var emaFast: Float? = null
    private var emaSlow: Float? = null
    private var noiseFloorDb: Float? = null
    private val noiseAlpha: Float = 0.05f
    private val vadMarginDb: Float = 8.0f

    fun resetState() {
        emaFast = null
        emaSlow = null
        noiseFloorDb = null
    }

    fun inferWindow(audio: FloatArray): InferenceResult {
        val indicators = extractor.extract(audio)
        val rmsDb = indicators.rmsDb
        if (noiseFloorDb == null) {
            noiseFloorDb = rmsDb
        }
        val noise = noiseFloorDb ?: vadThresholdDb
        if (rmsDb < noise + 3.0f) {
            noiseFloorDb = (1.0f - noiseAlpha) * noise + noiseAlpha * rmsDb
        }
        val dynamicThreshold = max(vadThresholdDb, (noiseFloorDb ?: vadThresholdDb) + vadMarginDb)
        val isSpeech = rmsDb > dynamicThreshold
        if (!isSpeech) {
            return InferenceResult(
                pFake = Float.NaN,
                pFakeSmooth = Float.NaN,
                confidence = 0.0f,
                isSpeech = false,
                reasons = emptyList(),
            )
        }

        val (pFakeRaw, reasons) = HeuristicDetector.pFake(indicators)
        val modelConfidence = abs(pFakeRaw - 0.5f) * 2.0f

        val quality = signalQuality(indicators, dynamicThreshold)
        val confidence = clamp(0.15f + 0.85f * (0.55f * quality + 0.45f * modelConfidence), 0.0f, 1.0f)
        val adjusted = 0.5f + (pFakeRaw - 0.5f) * confidence

        val fast = emaFast?.let { emaAlpha * adjusted + (1.0f - emaAlpha) * it } ?: adjusted
        emaFast = fast
        val slow = emaSlow?.let { emaSlowAlpha * adjusted + (1.0f - emaSlowAlpha) * it } ?: adjusted
        emaSlow = slow
        val smoothed = 0.65f * fast + 0.35f * slow

        return InferenceResult(
            pFake = clamp(adjusted, 0.0f, 1.0f),
            pFakeSmooth = clamp(smoothed, 0.0f, 1.0f),
            confidence = confidence,
            isSpeech = true,
            reasons = reasons,
        )
    }

    private fun signalQuality(indicators: Indicators, thresholdDb: Float): Float {
        val level = clamp((indicators.rmsDb - thresholdDb) / 25.0f, 0.0f, 1.0f)
        val bandwidth = clamp((indicators.spectralBandwidthHz - 900.0f) / 2500.0f, 0.0f, 1.0f)
        val hf = clamp((indicators.hfEnergyRatio - 0.01f) / 0.05f, 0.0f, 1.0f)
        val zcr = 1.0f - clamp(abs(indicators.zcr - 0.12f) / 0.12f, 0.0f, 1.0f)
        return clamp(0.45f * level + 0.25f * bandwidth + 0.20f * hf + 0.10f * zcr, 0.0f, 1.0f)
    }

    private fun clamp(x: Float, lo: Float, hi: Float): Float {
        return min(max(x, lo), hi)
    }
}
