package com.voiceguard.android.analysis

import kotlin.math.max

class AlertTracker(
    private val threshold: Float,
    private val holdSec: Float,
    private val stepSec: Float,
) {
    private var holdRemaining = 0.0f

    fun update(p: Float, isSpeech: Boolean): Boolean {
        if (isSpeech && p >= threshold) {
            holdRemaining = holdSec
        } else {
            holdRemaining = max(0.0f, holdRemaining - stepSec)
        }
        return holdRemaining > 0.0f
    }
}
