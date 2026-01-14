package com.voiceguard.android.analysis

data class Indicators(
    val rmsDb: Float,
    val zcr: Float,
    val spectralCentroidHz: Float,
    val spectralBandwidthHz: Float,
    val spectralRolloffHz: Float,
    val hfEnergyRatio: Float,
    val spectralFlatness: Float,
)
