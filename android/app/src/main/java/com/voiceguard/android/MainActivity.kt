package com.voiceguard.android

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.audiofx.AcousticEchoCanceler
import android.media.audiofx.AutomaticGainControl
import android.media.audiofx.NoiseSuppressor
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.OpenableColumns
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.app.AppCompatDelegate
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.voiceguard.android.analysis.AlertTracker
import com.voiceguard.android.analysis.AudioWindowProcessor
import com.voiceguard.android.analysis.InferenceResult
import com.voiceguard.android.analysis.VoiceGuardEngine
import com.voiceguard.android.databinding.ActivityMainBinding
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding

    private var analysisJob: Job? = null
    private var audioRecord: AudioRecord? = null
    private var mediaProjection: MediaProjection? = null
    private var pendingSource: Source? = null
    private var noiseSuppressor: NoiseSuppressor? = null
    private var echoCanceler: AcousticEchoCanceler? = null
    private var autoGain: AutomaticGainControl? = null
    private var alertThreshold: Float = 0.80f
    private var sessionStartMs: Long = 0L
    private var sumProb: Float = 0.0f
    private var countProb: Int = 0
    private var peakProb: Float = 0.0f
    private var selectedFileUri: Uri? = null
    private var selectedFileName: String = ""
    private var startAfterFilePick: Boolean = false

    private val sourceOptions = listOf(Source.MIC, Source.SYSTEM, Source.FILE)
    private val prefs by lazy { getSharedPreferences("voiceguard_prefs", Context.MODE_PRIVATE) }
    private val themePrefKey = "pref_dark_theme"

    private val recordPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        val source = pendingSource
        pendingSource = null
        if (granted && source != null) {
            startForSource(source)
        } else if (!granted) {
            setStatus(getString(R.string.status_permission_denied))
        }
    }

    private val projectionLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult(),
    ) { result ->
        if (result.resultCode == RESULT_OK && result.data != null) {
            val mgr = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
            val projection = mgr.getMediaProjection(result.resultCode, result.data!!)
            mediaProjection = projection
            startPlaybackCapture(projection)
        } else {
            setStatus(getString(R.string.status_playback_denied))
        }
    }

    private val filePickerLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocument(),
    ) { uri ->
        if (uri != null) {
            try {
                contentResolver.takePersistableUriPermission(
                    uri,
                    Intent.FLAG_GRANT_READ_URI_PERMISSION,
                )
            } catch (_: SecurityException) {
            }
            selectedFileUri = uri
            selectedFileName = resolveFileName(uri)
            binding.fileNameText.text = selectedFileName
        }
        if (startAfterFilePick && uri != null && analysisJob == null) {
            startAfterFilePick = false
            startFileAnalysis(uri)
        } else {
            startAfterFilePick = false
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        applySavedTheme()
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val sources = listOf(
            getString(R.string.source_mic),
            getString(R.string.source_system),
            getString(R.string.source_file),
        )
        val adapter = ArrayAdapter(this, R.layout.spinner_item, sources)
        adapter.setDropDownViewResource(R.layout.spinner_dropdown_item)
        binding.sourceSpinner.adapter = adapter
        binding.sourceSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(
                parent: AdapterView<*>?,
                view: android.view.View?,
                position: Int,
                id: Long,
            ) {
                updateSourceUi(controlsEnabled = analysisJob == null)
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {
                updateSourceUi(controlsEnabled = analysisJob == null)
            }
        }

        binding.noiseSuppressSwitch.isChecked = true
        binding.echoCancelSwitch.isChecked = true
        binding.autoGainSwitch.isChecked = false
        binding.fileNameText.text = getString(R.string.file_not_selected)
        binding.selectFileButton.setOnClickListener {
            filePickerLauncher.launch(arrayOf("audio/*"))
        }
        val isDarkTheme = prefs.getBoolean(themePrefKey, false)
        binding.themeSwitch.isChecked = isDarkTheme
        binding.themeSwitch.setOnCheckedChangeListener { _, isChecked ->
            onThemeToggle(isChecked)
        }

        binding.modeChipGroup.check(binding.chipMeeting.id)
        binding.modeChipGroup.setOnCheckedStateChangeListener { _, checkedIds ->
            val checked = checkedIds.firstOrNull() ?: return@setOnCheckedStateChangeListener
            val preset = when (checked) {
                binding.chipCall.id -> Preset.CALL
                binding.chipStudio.id -> Preset.STUDIO
                else -> Preset.MEETING
            }
            applyPreset(preset)
        }
        applyPreset(Preset.MEETING)

        binding.thresholdSlider.addOnChangeListener { _, value, _ ->
            alertThreshold = value
            binding.thresholdValue.text = String.format(Locale.getDefault(), "%.2f", value)
        }
        alertThreshold = binding.thresholdSlider.value
        binding.thresholdValue.text = String.format(Locale.getDefault(), "%.2f", binding.thresholdSlider.value)

        binding.startStopButton.setOnClickListener {
            if (analysisJob != null) {
                stopAnalysis()
            } else {
                val source = currentSource()
                if (source != Source.FILE && !hasRecordPermission()) {
                    pendingSource = source
                    recordPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                } else {
                    startForSource(source)
                }
            }
        }

        updateSourceUi(controlsEnabled = true)
        setStatus(getString(R.string.status_idle))
        updateResult(null, false)
    }

    override fun onStop() {
        super.onStop()
        stopAnalysis()
    }

    private fun currentSource(): Source {
        return sourceOptions.getOrNull(binding.sourceSpinner.selectedItemPosition) ?: Source.MIC
    }

    private fun hasRecordPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.RECORD_AUDIO,
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun applySavedTheme() {
        val isDark = prefs.getBoolean(themePrefKey, false)
        applyThemeMode(isDark, recreate = false)
    }

    private fun onThemeToggle(isDark: Boolean) {
        prefs.edit().putBoolean(themePrefKey, isDark).apply()
        if (analysisJob != null) {
            stopAnalysis()
        }
        applyThemeMode(isDark, recreate = true)
    }

    private fun applyThemeMode(isDark: Boolean, recreate: Boolean) {
        val mode = if (isDark) AppCompatDelegate.MODE_NIGHT_YES else AppCompatDelegate.MODE_NIGHT_NO
        if (AppCompatDelegate.getDefaultNightMode() != mode) {
            AppCompatDelegate.setDefaultNightMode(mode)
            if (recreate) {
                recreate()
            }
        }
    }

    private fun startForSource(source: Source) {
        when (source) {
            Source.MIC -> startMicCapture()
            Source.SYSTEM -> startSystemCapture()
            Source.FILE -> startFileAnalysisOrPick()
        }
    }

    private fun startMicCapture() {
        val (sampleRate, bufferSize) = pickAudioParams()
        val audioFormat = AudioFormat.Builder()
            .setSampleRate(sampleRate)
            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
            .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
            .build()

        val record = AudioRecord.Builder()
            .setAudioSource(MediaRecorder.AudioSource.VOICE_COMMUNICATION)
            .setAudioFormat(audioFormat)
            .setBufferSizeInBytes(bufferSize)
            .build()

        if (record.state != AudioRecord.STATE_INITIALIZED) {
            setStatus(getString(R.string.status_failed_mic))
            record.release()
            return
        }

        startAnalysis(record, sampleRate, getString(R.string.status_mic_fmt, sampleRate))
    }

    private fun startSystemCapture() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            setStatus(getString(R.string.status_playback_requires_q))
            return
        }
        val mgr = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        projectionLauncher.launch(mgr.createScreenCaptureIntent())
    }

    private fun startFileAnalysisOrPick() {
        val uri = selectedFileUri
        if (uri == null) {
            startAfterFilePick = true
            filePickerLauncher.launch(arrayOf("audio/*"))
            return
        }
        startFileAnalysis(uri)
    }

    private fun startPlaybackCapture(projection: MediaProjection) {
        val (sampleRate, bufferSize) = pickAudioParams()
        val audioFormat = AudioFormat.Builder()
            .setSampleRate(sampleRate)
            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
            .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
            .build()

        val configBuilder = android.media.AudioPlaybackCaptureConfiguration.Builder(projection)
        if (binding.voiceOnlySwitch.isChecked) {
            configBuilder.addMatchingUsage(AudioAttributes.USAGE_VOICE_COMMUNICATION)
            configBuilder.addMatchingUsage(AudioAttributes.USAGE_VOICE_COMMUNICATION_SIGNALLING)
        } else {
            configBuilder.addMatchingUsage(AudioAttributes.USAGE_MEDIA)
            configBuilder.addMatchingUsage(AudioAttributes.USAGE_VOICE_COMMUNICATION)
            configBuilder.addMatchingUsage(AudioAttributes.USAGE_VOICE_COMMUNICATION_SIGNALLING)
        }
        val config = configBuilder.build()

        val record = AudioRecord.Builder()
            .setAudioFormat(audioFormat)
            .setBufferSizeInBytes(bufferSize)
            .setAudioPlaybackCaptureConfig(config)
            .build()

        if (record.state != AudioRecord.STATE_INITIALIZED) {
            setStatus(getString(R.string.status_failed_system))
            record.release()
            return
        }

        startAnalysis(record, sampleRate, getString(R.string.status_system_fmt, sampleRate))
    }

    private fun startFileAnalysis(uri: Uri) {
        val name = selectedFileName.ifBlank { resolveFileName(uri) }
        selectedFileName = name
        binding.fileNameText.text = name

        val windowSec = 2.0f
        val hopSec = 0.5f
        val alertHoldSec = 3.0f

        setStatus(getString(R.string.status_file_fmt, name))
        binding.startStopButton.setText(R.string.stop)
        setControlsEnabled(false)
        binding.fileProgress.visibility = View.VISIBLE
        binding.fileProgress.progress = 0
        resetSessionStats()

        analysisJob = lifecycleScope.launch(Dispatchers.Default) {
            var extractor: MediaExtractor? = null
            var decoder: MediaCodec? = null
            var pfd: android.os.ParcelFileDescriptor? = null

            try {
                val localPfd = contentResolver.openFileDescriptor(uri, "r")
                    ?: throw IllegalStateException(getString(R.string.status_failed_file))
                pfd = localPfd
                val localExtractor = MediaExtractor()
                extractor = localExtractor
                localExtractor.setDataSource(localPfd.fileDescriptor)

                val trackIndex = (0 until localExtractor.trackCount).firstOrNull { idx ->
                    val format = localExtractor.getTrackFormat(idx)
                    val mime = format.getString(MediaFormat.KEY_MIME) ?: return@firstOrNull false
                    mime.startsWith("audio/")
                } ?: throw IllegalStateException(getString(R.string.status_failed_file))

                localExtractor.selectTrack(trackIndex)
                val format = localExtractor.getTrackFormat(trackIndex)
                val mime = format.getString(MediaFormat.KEY_MIME) ?: throw IllegalStateException(getString(R.string.status_failed_file))
                val sampleRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                var channelCount = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                val durationUs = if (format.containsKey(MediaFormat.KEY_DURATION)) {
                    format.getLong(MediaFormat.KEY_DURATION)
                } else {
                    -1L
                }
                val durationMs = if (durationUs > 0) durationUs / 1000L else -1L

                val engine = VoiceGuardEngine(sampleRate)
                val processor = AudioWindowProcessor(sampleRate, windowSec, hopSec)
                val alert = AlertTracker(alertThreshold, alertHoldSec, hopSec)

                val codec = MediaCodec.createDecoderByType(mime)
                decoder = codec
                codec.configure(format, null, null, 0)
                codec.start()

                val bufferInfo = MediaCodec.BufferInfo()
                var sawInputEOS = false
                var sawOutputEOS = false
                var outputEncoding = if (format.containsKey(MediaFormat.KEY_PCM_ENCODING)) {
                    format.getInteger(MediaFormat.KEY_PCM_ENCODING)
                } else {
                    AudioFormat.ENCODING_PCM_16BIT
                }
                var lastProgress = -1

                withContext(Dispatchers.Main) {
                    binding.fileProgress.isIndeterminate = durationMs <= 0
                }

                while (isActive && !sawOutputEOS) {
                    if (!sawInputEOS) {
                        val inputIndex = codec.dequeueInputBuffer(10_000)
                        if (inputIndex >= 0) {
                            val inputBuffer = codec.getInputBuffer(inputIndex)
                            if (inputBuffer == null) {
                                continue
                            }
                            val sampleSize = localExtractor.readSampleData(inputBuffer, 0)
                            if (sampleSize < 0) {
                                codec.queueInputBuffer(
                                    inputIndex,
                                    0,
                                    0,
                                    0,
                                    MediaCodec.BUFFER_FLAG_END_OF_STREAM,
                                )
                                sawInputEOS = true
                            } else {
                                val presentationTimeUs = localExtractor.sampleTime
                                codec.queueInputBuffer(inputIndex, 0, sampleSize, presentationTimeUs, 0)
                                localExtractor.advance()
                            }
                        }
                    }

                    val outputIndex = codec.dequeueOutputBuffer(bufferInfo, 10_000)
                    when {
                        outputIndex >= 0 -> {
                            val outputBuffer = codec.getOutputBuffer(outputIndex)
                            if (outputBuffer != null && bufferInfo.size > 0) {
                                outputBuffer.position(bufferInfo.offset)
                                outputBuffer.limit(bufferInfo.offset + bufferInfo.size)
                                val samples = decodePcmToFloat(outputBuffer, bufferInfo.size, channelCount, outputEncoding)
                                for (window in processor.push(samples)) {
                                    val result = engine.inferWindow(window.samples)
                                    val isAlert = if (result.isSpeech) {
                                        alert.update(result.pFakeSmooth, true)
                                    } else {
                                        alert.update(0.0f, false)
                                    }
                                    val endMs = ((window.startSample + processor.windowSamples).toDouble() * 1000.0 / sampleRate).toLong()
                                    val progress = if (durationMs > 0) {
                                        (endMs * 100 / durationMs).toInt().coerceIn(0, 100)
                                    } else {
                                        -1
                                    }
                                    val progressToSet = if (progress >= 0 && progress != lastProgress) {
                                        lastProgress = progress
                                        progress
                                    } else {
                                        null
                                    }
                                    withContext(Dispatchers.Main) {
                                        updateResult(result, isAlert, endMs)
                                        if (progressToSet != null) {
                                            binding.fileProgress.progress = progressToSet
                                        }
                                    }
                                }
                            }
                            codec.releaseOutputBuffer(outputIndex, false)
                            if ((bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                                sawOutputEOS = true
                            }
                        }
                        outputIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                            val outFormat = codec.outputFormat
                            if (outFormat.containsKey(MediaFormat.KEY_CHANNEL_COUNT)) {
                                channelCount = outFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                            }
                            outputEncoding = if (outFormat.containsKey(MediaFormat.KEY_PCM_ENCODING)) {
                                outFormat.getInteger(MediaFormat.KEY_PCM_ENCODING)
                            } else {
                                AudioFormat.ENCODING_PCM_16BIT
                            }
                        }
                    }
                }
            } catch (exc: Exception) {
                withContext(Dispatchers.Main) {
                    setStatus(getString(R.string.status_error_fmt, exc.message ?: "—"))
                }
            } finally {
                try {
                    decoder?.stop()
                } catch (_: Exception) {
                }
                try {
                    decoder?.release()
                } catch (_: Exception) {
                }
                try {
                    extractor?.release()
                } catch (_: Exception) {
                }
                try {
                    pfd?.close()
                } catch (_: Exception) {
                }
                withContext(Dispatchers.Main) {
                    analysisJob = null
                    resetUiAfterStop()
                    setStatus(getString(R.string.status_idle))
                }
            }
        }
    }

    private fun resolveFileName(uri: Uri): String {
        val projection = arrayOf(OpenableColumns.DISPLAY_NAME)
        contentResolver.query(uri, projection, null, null, null)?.use { cursor ->
            val idx = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (idx >= 0 && cursor.moveToFirst()) {
                val name = cursor.getString(idx)
                if (!name.isNullOrBlank()) {
                    return name
                }
            }
        }
        return uri.lastPathSegment ?: getString(R.string.file_not_selected)
    }

    private fun decodePcmToFloat(
        buffer: ByteBuffer,
        size: Int,
        channelCount: Int,
        encoding: Int,
    ): FloatArray {
        if (size <= 0) {
            return FloatArray(0)
        }
        buffer.order(ByteOrder.LITTLE_ENDIAN)
        val channels = channelCount.coerceAtLeast(1)
        return if (encoding == AudioFormat.ENCODING_PCM_FLOAT) {
            val frames = size / (4 * channels)
            val out = FloatArray(frames)
            for (i in 0 until frames) {
                var sum = 0.0f
                for (ch in 0 until channels) {
                    sum += buffer.float
                }
                out[i] = (sum / channels).coerceIn(-1.0f, 1.0f)
            }
            out
        } else {
            val frames = size / (2 * channels)
            val out = FloatArray(frames)
            for (i in 0 until frames) {
                var sum = 0.0f
                for (ch in 0 until channels) {
                    sum += buffer.short / 32768.0f
                }
                out[i] = (sum / channels).coerceIn(-1.0f, 1.0f)
            }
            out
        }
    }

    private fun pickAudioParams(): Pair<Int, Int> {
        val candidates = intArrayOf(16000, 48000, 44100)
        var sampleRate = 16000
        var minBuffer = 0
        for (rate in candidates) {
            val buffer = AudioRecord.getMinBufferSize(
                rate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
            )
            if (buffer > 0) {
                sampleRate = rate
                minBuffer = buffer
                break
            }
        }
        val preferredSamples = sampleRate / 2
        val bufferSize = maxOf(minBuffer, preferredSamples * 2)
        return sampleRate to bufferSize
    }

    private fun startAnalysis(record: AudioRecord, sampleRate: Int, status: String) {
        audioRecord = record
        val windowSec = 2.0f
        val hopSec = 0.5f
        val alertHoldSec = 3.0f

        setStatus(status)
        binding.startStopButton.setText(R.string.stop)
        setControlsEnabled(false)
        binding.fileProgress.visibility = View.GONE

        applyAudioEffects(record)
        resetSessionStats()

        analysisJob = lifecycleScope.launch(Dispatchers.Default) {
            val engine = VoiceGuardEngine(sampleRate)
            val processor = AudioWindowProcessor(sampleRate, windowSec, hopSec)
            val alert = AlertTracker(alertThreshold, alertHoldSec, hopSec)
            val frameBuffer = if (record.bufferSizeInFrames > 0) record.bufferSizeInFrames else sampleRate / 2
            val buffer = ShortArray(frameBuffer.coerceAtLeast(sampleRate / 2))

            try {
                record.startRecording()
                while (isActive) {
                    val read = record.read(buffer, 0, buffer.size)
                    if (read <= 0) {
                        continue
                    }
                    val floats = FloatArray(read)
                    for (i in 0 until read) {
                        floats[i] = buffer[i] / 32768.0f
                    }
                    for (window in processor.push(floats)) {
                        val result = engine.inferWindow(window.samples)
                        val isAlert = if (result.isSpeech) {
                            alert.update(result.pFakeSmooth, true)
                        } else {
                            alert.update(0.0f, false)
                        }
                        withContext(Dispatchers.Main) {
                            updateResult(result, isAlert)
                        }
                    }
                }
            } catch (exc: Exception) {
                withContext(Dispatchers.Main) {
                    setStatus(getString(R.string.status_error_fmt, exc.message ?: "—"))
                }
            } finally {
                try {
                    record.stop()
                } catch (_: Exception) {
                }
                record.release()
                releaseAudioEffects()
                withContext(Dispatchers.Main) {
                    analysisJob = null
                    resetUiAfterStop()
                }
            }
        }
    }

    private fun stopAnalysis() {
        analysisJob?.cancel()
        analysisJob = null
        audioRecord?.let { record ->
            try {
                record.stop()
            } catch (_: Exception) {
            }
            try {
                record.release()
            } catch (_: Exception) {
            }
        }
        audioRecord = null
        mediaProjection?.stop()
        mediaProjection = null
        resetUiAfterStop()
        releaseAudioEffects()
        setStatus(getString(R.string.status_idle))
    }

    private fun updateResult(result: InferenceResult?, isAlert: Boolean, elapsedMs: Long? = null) {
        val elapsed = elapsedMs ?: (System.currentTimeMillis() - sessionStartMs).coerceAtLeast(0L)
        binding.sessionText.text = formatElapsed(elapsed)
        if (result == null || !result.isSpeech) {
            binding.probText.text = getString(R.string.probability_empty)
            binding.alertText.text = getString(R.string.alert_empty)
            binding.reasonsText.text = "—"
            binding.avgText.text = "—"
            binding.peakText.text = "—"
            binding.probText.setTextColor(ContextCompat.getColor(this, R.color.vg_on_surface))
            binding.alertText.setTextColor(ContextCompat.getColor(this, R.color.vg_on_surface))
            return
        }
        val pct = (result.pFakeSmooth * 100.0f).coerceIn(0.0f, 100.0f)
        binding.probText.text = String.format(Locale.getDefault(), getString(R.string.probability_fmt), pct)
        binding.alertText.text = if (isAlert) getString(R.string.alert_active) else getString(R.string.alert_empty)
        binding.reasonsText.text = if (result.reasons.isEmpty()) "—" else result.reasons.joinToString("\n")

        val p = result.pFakeSmooth
        val color = when {
            p >= alertThreshold -> R.color.vg_danger
            p >= alertThreshold * 0.60f -> R.color.vg_warning
            else -> R.color.vg_on_surface
        }
        binding.probText.setTextColor(ContextCompat.getColor(this, color))
        binding.alertText.setTextColor(
            ContextCompat.getColor(this, if (isAlert) R.color.vg_danger else R.color.vg_on_surface)
        )

        sumProb += p
        countProb += 1
        peakProb = maxOf(peakProb, p)
        val avg = if (countProb > 0) sumProb / countProb.toFloat() else 0.0f
        binding.avgText.text = String.format(Locale.getDefault(), "%.0f%%", avg * 100.0f)
        binding.peakText.text = String.format(Locale.getDefault(), "%.0f%%", peakProb * 100.0f)

    }

    private fun setStatus(text: String) {
        binding.statusText.text = text
    }

    private fun resetUiAfterStop() {
        binding.startStopButton.setText(R.string.start)
        setControlsEnabled(true)
        binding.fileProgress.visibility = View.GONE
        binding.fileProgress.isIndeterminate = false
        binding.fileProgress.progress = 0
        resetSessionStats()
    }

    private fun resetSessionStats() {
        sessionStartMs = System.currentTimeMillis()
        sumProb = 0.0f
        countProb = 0
        peakProb = 0.0f
        binding.avgText.text = "—"
        binding.peakText.text = "—"
        binding.sessionText.text = "00:00"
    }

    private fun setControlsEnabled(enabled: Boolean) {
        binding.sourceSpinner.isEnabled = enabled
        binding.thresholdSlider.isEnabled = enabled
        setChipGroupEnabled(enabled)
        updateSourceUi(controlsEnabled = enabled)
    }

    private fun updateSourceUi(controlsEnabled: Boolean) {
        val source = currentSource()
        val isSystem = source == Source.SYSTEM
        val isFile = source == Source.FILE

        binding.fileSection.visibility = if (isFile) View.VISIBLE else View.GONE
        binding.selectFileButton.isEnabled = controlsEnabled
        binding.fileNameText.isEnabled = controlsEnabled
        binding.fileHintText.isEnabled = controlsEnabled
        binding.voiceOnlySwitch.isEnabled = controlsEnabled && isSystem
        binding.noiseSuppressSwitch.isEnabled = controlsEnabled && !isFile
        binding.echoCancelSwitch.isEnabled = controlsEnabled && !isFile
        binding.autoGainSwitch.isEnabled = controlsEnabled && !isFile

        binding.noteText.text = when (source) {
            Source.SYSTEM -> getString(R.string.playback_note)
            Source.FILE -> getString(R.string.note_file)
            Source.MIC -> getString(R.string.note_general)
        }
    }

    private fun setChipGroupEnabled(enabled: Boolean) {
        binding.modeChipGroup.isEnabled = enabled
        binding.chipCall.isEnabled = enabled
        binding.chipMeeting.isEnabled = enabled
        binding.chipStudio.isEnabled = enabled
    }

    private fun applyPreset(preset: Preset) {
        when (preset) {
            Preset.CALL -> {
                binding.thresholdSlider.value = 0.78f
                binding.voiceOnlySwitch.isChecked = true
                binding.noiseSuppressSwitch.isChecked = true
                binding.echoCancelSwitch.isChecked = true
                binding.autoGainSwitch.isChecked = false
            }
            Preset.MEETING -> {
                binding.thresholdSlider.value = 0.80f
                binding.voiceOnlySwitch.isChecked = true
                binding.noiseSuppressSwitch.isChecked = true
                binding.echoCancelSwitch.isChecked = true
                binding.autoGainSwitch.isChecked = false
            }
            Preset.STUDIO -> {
                binding.thresholdSlider.value = 0.85f
                binding.voiceOnlySwitch.isChecked = false
                binding.noiseSuppressSwitch.isChecked = false
                binding.echoCancelSwitch.isChecked = false
                binding.autoGainSwitch.isChecked = false
            }
        }
    }

    private fun formatElapsed(ms: Long): String {
        val totalSec = (ms / 1000L).toInt()
        val minutes = totalSec / 60
        val seconds = totalSec % 60
        return String.format(Locale.getDefault(), "%02d:%02d", minutes, seconds)
    }

    private fun applyAudioEffects(record: AudioRecord) {
        releaseAudioEffects()
        val sessionId = record.audioSessionId
        if (NoiseSuppressor.isAvailable()) {
            noiseSuppressor = NoiseSuppressor.create(sessionId)
            noiseSuppressor?.enabled = binding.noiseSuppressSwitch.isChecked
        }
        if (AcousticEchoCanceler.isAvailable()) {
            echoCanceler = AcousticEchoCanceler.create(sessionId)
            echoCanceler?.enabled = binding.echoCancelSwitch.isChecked
        }
        if (AutomaticGainControl.isAvailable()) {
            autoGain = AutomaticGainControl.create(sessionId)
            autoGain?.enabled = binding.autoGainSwitch.isChecked
        }
    }

    private fun releaseAudioEffects() {
        try {
            noiseSuppressor?.release()
        } catch (_: Exception) {
        }
        try {
            echoCanceler?.release()
        } catch (_: Exception) {
        }
        try {
            autoGain?.release()
        } catch (_: Exception) {
        }
        noiseSuppressor = null
        echoCanceler = null
        autoGain = null
    }

    private enum class Source {
        MIC,
        SYSTEM,
        FILE,
    }

    private enum class Preset {
        CALL,
        MEETING,
        STUDIO,
    }
}
