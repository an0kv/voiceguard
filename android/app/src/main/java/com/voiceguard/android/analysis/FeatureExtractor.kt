package com.voiceguard.android.analysis

import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import org.jtransforms.fft.FloatFFT_1D

class FeatureExtractor(private val sampleRate: Int) {
    private var windowSize = 0
    private var fftSize = 0
    private var window = FloatArray(0)
    private var fftBuffer = FloatArray(0)
    private var fft: FloatFFT_1D? = null

    fun extract(audio: FloatArray): Indicators {
        val rmsDb = rmsDb(audio)
        val zcr = zeroCrossingRate(audio)
        val spectral = spectralIndicators(audio)
        return Indicators(
            rmsDb = rmsDb,
            zcr = zcr,
            spectralCentroidHz = spectral.spectralCentroidHz,
            spectralBandwidthHz = spectral.spectralBandwidthHz,
            spectralRolloffHz = spectral.spectralRolloffHz,
            hfEnergyRatio = spectral.hfEnergyRatio,
            spectralFlatness = spectral.spectralFlatness,
        )
    }

    private fun rmsDb(audio: FloatArray): Float {
        if (audio.isEmpty()) {
            return -120.0f
        }
        var sum = 0.0
        for (v in audio) {
            sum += (v * v).toDouble()
        }
        val rms = sqrt(sum / audio.size)
        val db = 20.0 * log10(rms + 1e-12)
        return db.toFloat()
    }

    private fun zeroCrossingRate(audio: FloatArray): Float {
        if (audio.size < 2) {
            return 0.0f
        }
        var crossings = 0
        var prevPositive = audio[0] >= 0.0f
        for (i in 1 until audio.size) {
            val positive = audio[i] >= 0.0f
            if (positive != prevPositive) {
                crossings += 1
            }
            prevPositive = positive
        }
        return crossings.toFloat() / (audio.size - 1).toFloat()
    }

    private data class Spectral(
        val spectralCentroidHz: Float,
        val spectralBandwidthHz: Float,
        val spectralRolloffHz: Float,
        val hfEnergyRatio: Float,
        val spectralFlatness: Float,
    )

    private fun spectralIndicators(audio: FloatArray): Spectral {
        if (audio.isEmpty()) {
            return Spectral(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)
        }

        ensureFft(audio.size)

        val n = audio.size
        for (i in 0 until fftSize * 2) {
            fftBuffer[i] = 0.0f
        }
        for (i in 0 until n) {
            fftBuffer[i] = audio[i] * window[i]
        }

        fft?.realForwardFull(fftBuffer)

        val half = fftSize / 2
        val power = FloatArray(half + 1)
        var total = 0.0
        for (k in 0..half) {
            val re = fftBuffer[2 * k].toDouble()
            val im = fftBuffer[2 * k + 1].toDouble()
            val p = re * re + im * im
            power[k] = p.toFloat()
            total += p
        }
        val totalSafe = total + 1e-12

        var centroid = 0.0
        for (k in 0..half) {
            val freq = k.toDouble() * sampleRate.toDouble() / fftSize.toDouble()
            centroid += freq * power[k].toDouble()
        }
        centroid /= totalSafe

        var bandwidth = 0.0
        for (k in 0..half) {
            val freq = k.toDouble() * sampleRate.toDouble() / fftSize.toDouble()
            val diff = freq - centroid
            bandwidth += diff * diff * power[k].toDouble()
        }
        bandwidth = sqrt(bandwidth / totalSafe)

        val rolloffTarget = 0.85 * total
        var cumulative = 0.0
        var rolloffHz = 0.0
        for (k in 0..half) {
            cumulative += power[k].toDouble()
            if (cumulative >= rolloffTarget) {
                rolloffHz = k.toDouble() * sampleRate.toDouble() / fftSize.toDouble()
                break
            }
        }

        var hfEnergy = 0.0
        val hfCut = 6000.0
        for (k in 0..half) {
            val freq = k.toDouble() * sampleRate.toDouble() / fftSize.toDouble()
            if (freq >= hfCut) {
                hfEnergy += power[k].toDouble()
            }
        }
        val hfRatio = hfEnergy / totalSafe

        var logSum = 0.0
        var meanSum = 0.0
        val eps = 1e-12
        for (k in 0..half) {
            val p = power[k].toDouble() + eps
            logSum += ln(p)
            meanSum += p
        }
        val geomMean = exp(logSum / (half + 1).toDouble())
        val arithMean = meanSum / (half + 1).toDouble()
        val flatness = min(max((geomMean / arithMean), 0.0), 1.0)

        return Spectral(
            spectralCentroidHz = centroid.toFloat(),
            spectralBandwidthHz = bandwidth.toFloat(),
            spectralRolloffHz = rolloffHz.toFloat(),
            hfEnergyRatio = hfRatio.toFloat(),
            spectralFlatness = flatness.toFloat(),
        )
    }

    private fun ensureFft(size: Int) {
        if (size == windowSize && fftSize != 0) {
            return
        }
        windowSize = size
        fftSize = nextPow2(size)
        window = FloatArray(size)
        if (size == 1) {
            window[0] = 1.0f
        } else {
            val denom = (size - 1).toDouble()
            for (i in 0 until size) {
                window[i] = (0.5 - 0.5 * cos(2.0 * Math.PI * i.toDouble() / denom)).toFloat()
            }
        }
        fft = FloatFFT_1D(fftSize.toLong())
        fftBuffer = FloatArray(fftSize * 2)
    }

    private fun nextPow2(n: Int): Int {
        var v = max(n, 1)
        var p = 1
        while (p < v) {
            p = p shl 1
        }
        return p
    }
}
