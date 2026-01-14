package com.voiceguard.android.analysis

import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

object HeuristicDetector {
    private fun clamp(x: Float, lo: Float, hi: Float): Float {
        return min(max(x, lo), hi)
    }

    private fun sigmoid(x: Float): Float {
        return (1.0f / (1.0f + exp(-x)))
    }

    fun pFake(indicators: Indicators): Pair<Float, List<String>> {
        val hfRatio = indicators.hfEnergyRatio
        val rolloffHz = indicators.spectralRolloffHz
        val flatness = indicators.spectralFlatness
        val bandwidthHz = indicators.spectralBandwidthHz
        val centroidHz = indicators.spectralCentroidHz
        val zcr = indicators.zcr

        val cutoffScore = clamp((0.06f - hfRatio) / 0.06f, 0.0f, 1.0f)
        val rolloffScore = clamp((4200.0f - rolloffHz) / 4200.0f, 0.0f, 1.0f)
        val flatScore = clamp((0.12f - flatness) / 0.12f, 0.0f, 1.0f)
        val bandwidthScore = clamp((1800.0f - bandwidthHz) / 1800.0f, 0.0f, 1.0f)
        val centroidScore = clamp((1700.0f - centroidHz) / 1700.0f, 0.0f, 1.0f)
        val zcrScore = clamp((0.04f - zcr) / 0.04f, 0.0f, 1.0f)

        val raw = 0.36f * cutoffScore +
            0.27f * rolloffScore +
            0.13f * flatScore +
            0.12f * bandwidthScore +
            0.07f * centroidScore +
            0.05f * zcrScore
        val pFake = sigmoid((raw - 0.34f) * 7.0f)

        return clamp(pFake, 0.0f, 1.0f) to reasons(indicators)
    }

    fun reasons(indicators: Indicators): List<String> {
        val hfRatio = indicators.hfEnergyRatio
        val rolloffHz = indicators.spectralRolloffHz
        val flatness = indicators.spectralFlatness
        val bandwidthHz = indicators.spectralBandwidthHz
        val centroidHz = indicators.spectralCentroidHz
        val zcr = indicators.zcr

        val reasons = mutableListOf<String>()
        if (hfRatio < 0.02f) {
            reasons.add("Низкая доля ВЧ энергии (возможен срез/кодек/TTS)")
        }
        if (rolloffHz < 3200.0f) {
            reasons.add("Низкий spectral roll-off (возможен срез ВЧ)")
        }
        if (flatness < 0.08f) {
            reasons.add("Низкая spectral flatness (слишком стерильный спектр)")
        }
        if (bandwidthHz < 1800.0f) {
            reasons.add("Слишком узкий спектр (ограничение полосы)")
        }
        if (centroidHz < 1700.0f) {
            reasons.add("Смещение энергии в НЧ (мутный/перефильтрованный звук)")
        }
        if (zcr < 0.03f) {
            reasons.add("Низкий ZCR (слишком ровный/тональный сигнал)")
        }
        return reasons
    }
}
