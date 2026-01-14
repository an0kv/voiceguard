package com.voiceguard.android.analysis

data class InferenceResult(
    val pFake: Float,
    val pFakeSmooth: Float,
    val confidence: Float,
    val isSpeech: Boolean,
    val reasons: List<String>,
)
