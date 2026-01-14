package com.voiceguard.android.analysis

class AudioWindowProcessor(
    private val sampleRate: Int,
    windowSec: Float,
    hopSec: Float,
) {
    val windowSamples: Int = (windowSec * sampleRate).toInt().coerceAtLeast(1)
    private val hopSamples: Int = (hopSec * sampleRate).toInt().coerceAtLeast(1)
    private var buffer = FloatArray(0)
    private var processedSamples: Long = 0

    data class Window(val samples: FloatArray, val startSample: Long)

    fun push(samples: FloatArray): List<Window> {
        if (samples.isEmpty()) {
            return emptyList()
        }
        val combined = FloatArray(buffer.size + samples.size)
        if (buffer.isNotEmpty()) {
            System.arraycopy(buffer, 0, combined, 0, buffer.size)
        }
        System.arraycopy(samples, 0, combined, buffer.size, samples.size)
        buffer = combined

        val windows = mutableListOf<Window>()
        while (buffer.size >= windowSamples) {
            val window = buffer.copyOfRange(0, windowSamples)
            windows.add(Window(window, processedSamples))
            buffer = if (buffer.size > hopSamples) {
                buffer.copyOfRange(hopSamples, buffer.size)
            } else {
                FloatArray(0)
            }
            processedSamples += hopSamples.toLong()
        }
        return windows
    }
}
