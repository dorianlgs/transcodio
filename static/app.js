// DOM Elements
const uploadSection = document.getElementById('uploadSection');
const processingSection = document.getElementById('processingSection');
const resultsSection = document.getElementById('resultsSection');
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const enableDiarizationCheckbox = document.getElementById('enableDiarization');
const enableMinutesCheckbox = document.getElementById('enableMinutes');
const processingStatus = document.getElementById('processingStatus');
const progressFill = document.getElementById('progressFill');
const segments = document.getElementById('segments');
const duration = document.getElementById('duration');
const language = document.getElementById('language');
const audioPlayer = document.getElementById('audioPlayer');
const downloadBtn = document.getElementById('downloadBtn');
const downloadSrtBtn = document.getElementById('downloadSrtBtn');
const downloadVttBtn = document.getElementById('downloadVttBtn');
const newTranscriptionBtn = document.getElementById('newTranscriptionBtn');
const toast = document.getElementById('toast');

// Minutes elements
const minutesSection = document.getElementById('minutesSection');
const minutesLoading = document.getElementById('minutesLoading');
const minutesContent = document.getElementById('minutesContent');
const minutesSummary = document.getElementById('minutesSummary');
const minutesKeyPoints = document.getElementById('minutesKeyPoints');
const minutesDecisions = document.getElementById('minutesDecisions');
const minutesActions = document.getElementById('minutesActions');
const minutesParticipants = document.getElementById('minutesParticipants');
const downloadMinutesBtn = document.getElementById('downloadMinutesBtn');

// Tab elements
const tabBtns = document.querySelectorAll('.tab-btn');
const transcriptionTab = document.getElementById('transcriptionTab');
const minutesTab = document.getElementById('minutesTab');
const minutesTabBtn = document.getElementById('minutesTabBtn');

// Mode selector elements
const transcriptionModeBtn = document.getElementById('transcriptionModeBtn');
const voiceCloneModeBtn = document.getElementById('voiceCloneModeBtn');
const imageGenModeBtn = document.getElementById('imageGenModeBtn');

// Voice clone elements
const voiceCloneSection = document.getElementById('voiceCloneSection');
const voiceCloneProcessing = document.getElementById('voiceCloneProcessing');
const voiceCloneResults = document.getElementById('voiceCloneResults');
const refUploadTab = document.getElementById('refUploadTab');
const refRecordTab = document.getElementById('refRecordTab');
const refUploadContent = document.getElementById('refUploadContent');
const refRecordContent = document.getElementById('refRecordContent');
const refUploadArea = document.getElementById('refUploadArea');
const refFileInput = document.getElementById('refFileInput');
const refAudioPreview = document.getElementById('refAudioPreview');
const refAudioPlayer = document.getElementById('refAudioPlayer');
const removeRefAudio = document.getElementById('removeRefAudio');
const recordBtn = document.getElementById('recordBtn');
const recordTimer = document.getElementById('recordTimer');
const recordProgressBar = document.getElementById('recordProgressBar');
const recordedPreview = document.getElementById('recordedPreview');
const recordedAudioPlayer = document.getElementById('recordedAudioPlayer');
const removeRecordedAudio = document.getElementById('removeRecordedAudio');
const refTextInput = document.getElementById('refTextInput');
const targetTextInput = document.getElementById('targetTextInput');
const charCount = document.getElementById('charCount');
const languageSelect = document.getElementById('languageSelect');
const ttsModelSelect = document.getElementById('ttsModelSelect');
const modelHint = document.getElementById('modelHint');
const generateVoiceBtn = document.getElementById('generateVoiceBtn');
const generatedAudioPlayer = document.getElementById('generatedAudioPlayer');
const generatedDuration = document.getElementById('generatedDuration');
const downloadVoiceBtn = document.getElementById('downloadVoiceBtn');
const newVoiceCloneBtn = document.getElementById('newVoiceCloneBtn');

// Saved voices elements
const savedVoiceModeTab = document.getElementById('savedVoiceModeTab');
const newVoiceModeTab = document.getElementById('newVoiceModeTab');
const savedVoiceModeContent = document.getElementById('savedVoiceModeContent');
const newVoiceModeContent = document.getElementById('newVoiceModeContent');
const savedVoicesList = document.getElementById('savedVoicesList');
const noVoicesMessage = document.getElementById('noVoicesMessage');
const goToNewVoiceBtn = document.getElementById('goToNewVoiceBtn');
const savedVoiceTargetText = document.getElementById('savedVoiceTargetText');
const savedVoiceCharCount = document.getElementById('savedVoiceCharCount');
const synthesizeBtn = document.getElementById('synthesizeBtn');
const saveVoiceBtn = document.getElementById('saveVoiceBtn');
const voiceNameInput = document.getElementById('voiceNameInput');

// Image generation elements
const imageGenSection = document.getElementById('imageGenSection');
const imageGenProcessing = document.getElementById('imageGenProcessing');
const imageGenResults = document.getElementById('imageGenResults');
const imagePromptInput = document.getElementById('imagePromptInput');
const imageCharCount = document.getElementById('imageCharCount');
const imageWidthSelect = document.getElementById('imageWidthSelect');
const imageHeightSelect = document.getElementById('imageHeightSelect');
const generateImageBtn = document.getElementById('generateImageBtn');
const generatedImagePreview = document.getElementById('generatedImagePreview');
const generatedImageDimensions = document.getElementById('generatedImageDimensions');
const downloadImageBtn = document.getElementById('downloadImageBtn');
const newImageGenBtn = document.getElementById('newImageGenBtn');

// State
let currentTranscription = '';
let currentFile = null;
let currentSegments = []; // Store segments for subtitle export
let currentMinutes = null; // Store minutes data

// Voice clone state
let currentMode = 'transcription';
let refAudioFile = null;
let refAudioBlob = null;
let mediaRecorder = null;
let recordedChunks = [];
let recordingStartTime = null;
let recordingInterval = null;
let generatedAudioSessionId = null;
let generatedImageSessionId = null;

// Saved voices state
let savedVoices = [];
let selectedVoiceId = null;

// Initialize
function init() {
    setupEventListeners();
    // Don't load voices on init - will load when user switches to voice clone mode
}

// Event Listeners
function setupEventListeners() {
    // Upload area click
    uploadArea.addEventListener('click', () => fileInput.click());

    // File input change
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);

    // Audio player events
    audioPlayer.addEventListener('loadedmetadata', () => {
    });
    audioPlayer.addEventListener('error', (e) => {
        console.error('Audio player error:', e, audioPlayer.error);
        showToast('No se pudo cargar el archivo de audio', 'error');
    });

    // Button clicks
    downloadBtn.addEventListener('click', downloadTranscription);
    downloadSrtBtn.addEventListener('click', downloadSRT);
    downloadVttBtn.addEventListener('click', downloadVTT);
    downloadMinutesBtn.addEventListener('click', downloadMinutes);
    newTranscriptionBtn.addEventListener('click', resetApp);

    // Tab switching
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            if (btn.classList.contains('disabled')) return;
            switchTab(btn.dataset.tab);
        });
    });

    // Mode switching
    transcriptionModeBtn.addEventListener('click', () => switchMode('transcription'));
    voiceCloneModeBtn.addEventListener('click', () => switchMode('voice-clone'));
    imageGenModeBtn.addEventListener('click', () => switchMode('image-gen'));

    // Voice clone - Reference audio tabs
    refUploadTab.addEventListener('click', () => switchRefAudioTab('upload'));
    refRecordTab.addEventListener('click', () => switchRefAudioTab('record'));

    // Voice clone - Upload area
    refUploadArea.addEventListener('click', () => refFileInput.click());
    refFileInput.addEventListener('change', handleRefFileSelect);
    refUploadArea.addEventListener('dragover', (e) => { e.preventDefault(); refUploadArea.classList.add('dragover'); });
    refUploadArea.addEventListener('dragleave', (e) => { e.preventDefault(); refUploadArea.classList.remove('dragover'); });
    refUploadArea.addEventListener('drop', handleRefFileDrop);
    removeRefAudio.addEventListener('click', removeRefAudioFile);

    // Voice clone - Recording
    recordBtn.addEventListener('click', toggleRecording);
    removeRecordedAudio.addEventListener('click', removeRecordedAudioFile);

    // Voice clone - Text input
    targetTextInput.addEventListener('input', updateCharCount);

    // Voice clone - Model selection
    ttsModelSelect.addEventListener('change', updateModelHint);

    // Voice clone - Generate
    generateVoiceBtn.addEventListener('click', generateVoiceClone);

    // Voice clone - Results
    downloadVoiceBtn.addEventListener('click', downloadGeneratedVoice);
    newVoiceCloneBtn.addEventListener('click', resetVoiceClone);

    // Image generation
    imagePromptInput.addEventListener('input', updateImageCharCount);
    generateImageBtn.addEventListener('click', generateImage);
    downloadImageBtn.addEventListener('click', downloadGeneratedImage);
    newImageGenBtn.addEventListener('click', resetImageGen);

    // Saved voices
    savedVoiceModeTab.addEventListener('click', () => switchVoiceMode('saved'));
    newVoiceModeTab.addEventListener('click', () => switchVoiceMode('new'));
    goToNewVoiceBtn.addEventListener('click', () => switchVoiceMode('new'));
    savedVoiceTargetText.addEventListener('input', updateSavedVoiceCharCount);
    synthesizeBtn.addEventListener('click', synthesizeWithSavedVoice);
    saveVoiceBtn.addEventListener('click', saveVoice);
}

// Tab switching function
function switchTab(tabName) {
    // Update button states
    tabBtns.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabName);
    });

    // Update tab content visibility
    transcriptionTab.classList.toggle('active', tabName === 'transcription');
    minutesTab.classList.toggle('active', tabName === 'minutes');
}

// File Handling
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processFile(file);
    }
}

function handleDragOver(event) {
    event.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');

    const file = event.dataTransfer.files[0];
    if (file) {
        processFile(file);
    }
}

// File Processing
async function processFile(file) {
    // Store the file reference BEFORE creating FormData
    currentFile = file;

    // Validate file
    if (!validateFile(file)) {
        return;
    }

    // Show processing section
    showSection('processing');
    processingStatus.textContent = 'Uploading file...';
    progressFill.style.width = '10%';

    try {
        // Create form data - this does NOT consume the file
        const formData = new FormData();
        formData.append('file', file);

        // Add options from checkboxes
        const enableDiarization = enableDiarizationCheckbox.checked;
        const enableMinutes = enableMinutesCheckbox.checked;
        formData.append('enable_diarization', enableDiarization);
        formData.append('enable_minutes', enableMinutes);

        // Upload and transcribe with streaming
        await transcribeWithStreaming(formData, enableDiarization, enableMinutes);

    } catch (error) {
        console.error('Error:', error);
        showToast(`Error: ${error.message}`, 'error');
        showSection('upload');
    }
}

function validateFile(file) {
    // Check file type
    const validTypes = ['audio/mpeg', 'audio/wav', 'audio/x-wav', 'audio/wave', 'audio/mp4', 'audio/x-m4a', 'audio/flac', 'audio/ogg', 'audio/webm', 'video/mp4'];
    if (!validTypes.includes(file.type) && !file.name.match(/\.(mp3|wav|m4a|flac|ogg|webm|mp4)$/i)) {
        showToast('Por favor sube un archivo de audio válido (MP3, WAV, M4A, FLAC, OGG, MP4)', 'error');
        return false;
    }

    // Check file size (100MB max)
    const maxSize = 100 * 1024 * 1024;
    if (file.size > maxSize) {
        showToast('El archivo excede el límite de 100MB', 'error');
        return false;
    }

    return true;
}

// Transcription with Streaming
async function transcribeWithStreaming(formData, enableDiarization = false, enableMinutes = false) {
    processingStatus.textContent = 'Iniciando transcripción...';
    progressFill.style.width = '20%';

    // Reset results
    segments.innerHTML = '';
    currentTranscription = '';
    currentSegments = [];
    currentMinutes = null;

    // Reset minutes section
    minutesLoading.classList.add('hidden');
    minutesContent.classList.remove('hidden');
    minutesSummary.textContent = '';
    minutesKeyPoints.innerHTML = '';
    minutesDecisions.innerHTML = '';
    minutesActions.innerHTML = '';
    minutesParticipants.innerHTML = '';

    // Configure tabs based on whether minutes are enabled
    if (enableMinutes) {
        minutesTabBtn.classList.remove('disabled');
        minutesLoading.classList.remove('hidden');
        minutesContent.classList.add('hidden');
    } else {
        minutesTabBtn.classList.add('disabled');
    }

    // Always start on transcription tab
    switchTab('transcription');

    try {
        const response = await fetch('/api/transcribe/stream', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Transcription failed');
        }

        // Keep processing section visible during streaming
        showSection('processing');
        processingStatus.textContent = 'Transcribing...';

        if (!response.body) {
            throw new Error('Response body is null');
        }

        // Process SSE stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let eventCount = 0;

        while (true) {
            const { done, value } = await reader.read();

            if (done) {
                break;
            }

            // Decode chunk
            const chunk = decoder.decode(value, { stream: true });

            buffer += chunk;

            // Process complete events (SSE format: event ends with double newline)
            // Normalize line endings first
            const normalizedBuffer = buffer.replace(/\r\n/g, '\n');
            const eventStrings = normalizedBuffer.split('\n\n');

            // Last element might be incomplete, keep in buffer
            buffer = eventStrings[eventStrings.length - 1];

            // Process all complete events (all except last)
            for (let i = 0; i < eventStrings.length - 1; i++) {
                const eventText = eventStrings[i];
                if (!eventText.trim()) {
                    continue;
                }

                const lines = eventText.split('\n');
                let eventType = 'message';
                let eventData = '';

                for (const line of lines) {
                    if (line.startsWith('event:')) {
                        eventType = line.substring(6).trim();
                    } else if (line.startsWith('data:')) {
                        eventData = line.substring(5).trim();
                    }
                }

                if (eventData) {
                    eventCount++;
                    handleStreamEvent(eventType, JSON.parse(eventData));
                }
            }
        }

    } catch (error) {
        console.error('Streaming error:', error);
        showToast(`Error: ${error.message}`, 'error');
        showSection('upload');
    }
}

// Handle Stream Events
function handleStreamEvent(eventType, data) {
    switch (eventType) {
        case 'metadata':
            // Update metadata
            if (data.duration) {
                duration.textContent = formatDuration(data.duration);
            }
            if (data.language) {
                language.textContent = data.language.toUpperCase();
            }
            processingStatus.textContent = 'Transcribiendo audio...';
            progressFill.style.width = '40%';
            break;

        case 'progress':
            // Add segment
            addSegment(data);
            // Update progress (estimate based on time)
            const progress = Math.min(90, 40 + (data.end / 600) * 50);
            progressFill.style.width = `${progress}%`;
            break;

        case 'speakers_ready':
            // Update segments with speaker labels
            updateSegmentsWithSpeakers(data.segments);
            showToast('Identificación de hablantes completada', 'success');
            break;

        case 'complete':
            // Set final text
            currentTranscription = data.text;
            progressFill.style.width = '100%';

            // Set audio player source using the preprocessed audio from backend
            if (data.audio_session_id) {
                // Use the preprocessed audio from the backend (16kHz mono WAV)
                const audioURL = `/api/audio/${data.audio_session_id}`;
                audioPlayer.src = audioURL;
                audioPlayer.load(); // Force the browser to load the audio metadata
            } else {
                console.warn('No se proporcionó ID de sesión de audio, no se puede cargar el audio');
            }

            // Show results section now
            showSection('results');
            showToast('Transcripción completada!', 'success');
            break;

        case 'minutes_ready':
            // Display meeting minutes
            displayMeetingMinutes(data.minutes);
            showToast('Minuta generada!', 'success');
            break;

        case 'minutes_error':
            // Handle minutes generation error
            minutesLoading.classList.add('hidden');
            minutesContent.classList.remove('hidden');
            minutesSummary.textContent = 'No se pudo generar la minuta. ' + (data.error || '');
            showToast('Error al generar la minuta', 'error');
            break;

        case 'error':
            console.error('Stream error:', data.error);
            showToast(`Error: ${data.error}`, 'error');
            showSection('upload');
            break;
    }
}

// Add Segment to UI
function addSegment(segmentData) {
    // Store segment data for subtitle export
    currentSegments.push({
        id: segmentData.id,
        start: segmentData.start,
        end: segmentData.end,
        text: segmentData.text,
        speaker: segmentData.speaker || null
    });

    const segmentEl = document.createElement('div');
    segmentEl.className = 'segment';
    segmentEl.setAttribute('data-segment-id', segmentData.id);  // For later updates

    const timeEl = document.createElement('div');
    timeEl.className = 'segment-time';
    timeEl.textContent = `${formatTime(segmentData.start)} - ${formatTime(segmentData.end)}`;

    // Add speaker badge if present
    if (segmentData.speaker) {
        const speakerBadge = document.createElement('span');
        speakerBadge.className = 'speaker-badge';
        speakerBadge.textContent = segmentData.speaker;
        speakerBadge.setAttribute('data-speaker', segmentData.speaker);
        timeEl.appendChild(speakerBadge);
    }

    const textEl = document.createElement('div');
    textEl.className = 'segment-text';
    textEl.textContent = segmentData.text;

    segmentEl.appendChild(timeEl);
    segmentEl.appendChild(textEl);
    segments.appendChild(segmentEl);

    // Scroll to bottom
    segmentEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Update Segments with Speaker Labels
function updateSegmentsWithSpeakers(annotatedSegments) {
    annotatedSegments.forEach(seg => {
        const segmentEl = segments.querySelector(`[data-segment-id="${seg.id}"]`);
        if (segmentEl && seg.speaker) {
            const timeEl = segmentEl.querySelector('.segment-time');

            // Create speaker badge
            const speakerBadge = document.createElement('span');
            speakerBadge.className = 'speaker-badge';
            speakerBadge.textContent = seg.speaker;
            speakerBadge.setAttribute('data-speaker', seg.speaker);
            timeEl.appendChild(speakerBadge);

            // Update currentSegments array
            const idx = currentSegments.findIndex(s => s.id === seg.id);
            if (idx >= 0) currentSegments[idx].speaker = seg.speaker;
        }
    });
}

// Utility Functions
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

function formatDuration(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}m ${secs}s`;
}

function showSection(section) {
    uploadSection.classList.add('hidden');
    processingSection.classList.add('hidden');
    resultsSection.classList.add('hidden');

    switch (section) {
        case 'upload':
            uploadSection.classList.remove('hidden');
            break;
        case 'processing':
            processingSection.classList.remove('hidden');
            break;
        case 'results':
            resultsSection.classList.remove('hidden');
            break;
    }
}

function showToast(message, type = 'info') {
    toast.textContent = message;
    toast.className = `toast ${type}`;
    toast.classList.remove('hidden');

    setTimeout(() => {
        toast.classList.add('hidden');
    }, 4000);
}

// Actions
function downloadTranscription() {
    const blob = new Blob([currentTranscription], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `transcription-${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast('Descargado!', 'success');
}

// Format time for SRT (HH:MM:SS,mmm)
function formatTimeSRT(seconds) {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.round((seconds % 1) * 1000);
    return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')},${ms.toString().padStart(3, '0')}`;
}

// Format time for VTT (HH:MM:SS.mmm)
function formatTimeVTT(seconds) {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.round((seconds % 1) * 1000);
    return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(3, '0')}`;
}

// Generate SRT subtitle content
function generateSRT() {
    return currentSegments.map((segment, index) => {
        const speaker = segment.speaker ? `[${segment.speaker}] ` : '';
        return `${index + 1}\n${formatTimeSRT(segment.start)} --> ${formatTimeSRT(segment.end)}\n${speaker}${segment.text.trim()}\n`;
    }).join('\n');
}

// Generate VTT subtitle content
function generateVTT() {
    const header = 'WEBVTT\n\n';
    const cues = currentSegments.map((segment, index) => {
        const speaker = segment.speaker ? `[${segment.speaker}] ` : '';
        return `${index + 1}\n${formatTimeVTT(segment.start)} --> ${formatTimeVTT(segment.end)}\n${speaker}${segment.text.trim()}\n`;
    }).join('\n');
    return header + cues;
}

// Download SRT file
function downloadSRT() {
    if (currentSegments.length === 0) {
        showToast('No hay datos de transcripción disponibles', 'error');
        return;
    }
    const srtContent = generateSRT();
    const blob = new Blob([srtContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `transcription-${Date.now()}.srt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast('SRT descargado!', 'success');
}

// Download VTT file
function downloadVTT() {
    if (currentSegments.length === 0) {
        showToast('No hay datos de transcripción disponibles', 'error');
        return;
    }
    const vttContent = generateVTT();
    const blob = new Blob([vttContent], { type: 'text/vtt' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `transcription-${Date.now()}.vtt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast('VTT descargado!', 'success');
}

// Display Meeting Minutes
function displayMeetingMinutes(minutes) {
    currentMinutes = minutes;

    // Hide loading, show content
    minutesLoading.classList.add('hidden');
    minutesContent.classList.remove('hidden');

    // Executive Summary
    minutesSummary.textContent = minutes.executive_summary || 'No hay resumen disponible.';

    // Key Discussion Points
    minutesKeyPoints.innerHTML = '';
    if (minutes.key_discussion_points && minutes.key_discussion_points.length > 0) {
        minutes.key_discussion_points.forEach(point => {
            const li = document.createElement('li');
            li.textContent = point;
            minutesKeyPoints.appendChild(li);
        });
    } else {
        const li = document.createElement('li');
        li.textContent = 'No se identificaron puntos clave.';
        li.className = 'empty-item';
        minutesKeyPoints.appendChild(li);
    }

    // Decisions Made
    minutesDecisions.innerHTML = '';
    if (minutes.decisions_made && minutes.decisions_made.length > 0) {
        minutes.decisions_made.forEach(decision => {
            const li = document.createElement('li');
            li.textContent = decision;
            minutesDecisions.appendChild(li);
        });
    } else {
        const li = document.createElement('li');
        li.textContent = 'No se registraron decisiones.';
        li.className = 'empty-item';
        minutesDecisions.appendChild(li);
    }

    // Action Items
    minutesActions.innerHTML = '';
    if (minutes.action_items && minutes.action_items.length > 0) {
        minutes.action_items.forEach(item => {
            const li = document.createElement('li');
            li.className = 'action-item';
            li.innerHTML = `
                <div class="action-task">${item.task || 'Sin tarea especificada'}</div>
                <div class="action-meta">
                    <span class="action-label">Responsable:</span> <span class="action-assignee">${item.assignee || 'Sin asignar'}</span>
                    <span class="action-separator">|</span>
                    <span class="action-label">Fecha:</span> <span class="action-deadline">${item.deadline || 'Por definir'}</span>
                </div>
            `;
            minutesActions.appendChild(li);
        });
    } else {
        const li = document.createElement('li');
        li.textContent = 'No se identificaron acciones pendientes.';
        li.className = 'empty-item';
        minutesActions.appendChild(li);
    }

    // Participants Mentioned
    minutesParticipants.innerHTML = '';
    if (minutes.participants_mentioned && minutes.participants_mentioned.length > 0) {
        minutes.participants_mentioned.forEach(participant => {
            const tag = document.createElement('span');
            tag.className = 'participant-tag';
            tag.textContent = participant;
            minutesParticipants.appendChild(tag);
        });
    } else {
        const span = document.createElement('span');
        span.textContent = 'No se mencionaron participantes por nombre.';
        span.className = 'empty-item';
        minutesParticipants.appendChild(span);
    }
}

// Download Meeting Minutes as TXT
function downloadMinutes() {
    if (!currentMinutes) {
        showToast('No hay minuta disponible', 'error');
        return;
    }

    let content = 'MINUTA DE REUNIÓN\n';
    content += '='.repeat(50) + '\n\n';

    content += 'RESUMEN EJECUTIVO\n';
    content += '-'.repeat(30) + '\n';
    content += (currentMinutes.executive_summary || 'No hay resumen disponible.') + '\n\n';

    content += 'PUNTOS CLAVE DISCUTIDOS\n';
    content += '-'.repeat(30) + '\n';
    if (currentMinutes.key_discussion_points && currentMinutes.key_discussion_points.length > 0) {
        currentMinutes.key_discussion_points.forEach((point, i) => {
            content += `${i + 1}. ${point}\n`;
        });
    } else {
        content += 'No se identificaron puntos clave.\n';
    }
    content += '\n';

    content += 'DECISIONES TOMADAS\n';
    content += '-'.repeat(30) + '\n';
    if (currentMinutes.decisions_made && currentMinutes.decisions_made.length > 0) {
        currentMinutes.decisions_made.forEach((decision, i) => {
            content += `${i + 1}. ${decision}\n`;
        });
    } else {
        content += 'No se registraron decisiones.\n';
    }
    content += '\n';

    content += 'ACCIONES PENDIENTES\n';
    content += '-'.repeat(30) + '\n';
    if (currentMinutes.action_items && currentMinutes.action_items.length > 0) {
        currentMinutes.action_items.forEach((item, i) => {
            content += `${i + 1}. ${item.task || 'Sin tarea especificada'}\n`;
            content += `   - Responsable: ${item.assignee || 'Sin asignar'}\n`;
            content += `   - Fecha: ${item.deadline || 'Por definir'}\n\n`;
        });
    } else {
        content += 'No se identificaron acciones pendientes.\n';
    }
    content += '\n';

    content += 'PARTICIPANTES MENCIONADOS\n';
    content += '-'.repeat(30) + '\n';
    if (currentMinutes.participants_mentioned && currentMinutes.participants_mentioned.length > 0) {
        content += currentMinutes.participants_mentioned.join(', ') + '\n';
    } else {
        content += 'No se mencionaron participantes por nombre.\n';
    }

    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `minuta-reunion-${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast('Minuta descargada!', 'success');
}

function resetApp() {
    showSection('upload');
    fileInput.value = '';
    currentFile = null;
    currentTranscription = '';
    currentSegments = [];
    currentMinutes = null;
    segments.innerHTML = '';
    progressFill.style.width = '0%';

    // Reset audio player
    audioPlayer.src = '';
    audioPlayer.load();

    // Reset minutes section
    minutesLoading.classList.add('hidden');
    minutesContent.classList.remove('hidden');
    minutesSummary.textContent = '';
    minutesKeyPoints.innerHTML = '';
    minutesDecisions.innerHTML = '';
    minutesActions.innerHTML = '';
    minutesParticipants.innerHTML = '';

    // Reset tabs
    minutesTabBtn.classList.add('disabled');
    switchTab('transcription');

    // Reset checkboxes
    enableDiarizationCheckbox.checked = false;
    enableMinutesCheckbox.checked = false;
}

// ============================================
// Voice Clone Functions
// ============================================

// Mode Switching
function switchMode(mode) {
    currentMode = mode;

    // Update mode buttons
    transcriptionModeBtn.classList.toggle('active', mode === 'transcription');
    voiceCloneModeBtn.classList.toggle('active', mode === 'voice-clone');
    imageGenModeBtn.classList.toggle('active', mode === 'image-gen');

    // Hide all sections first
    uploadSection.classList.add('hidden');
    processingSection.classList.add('hidden');
    resultsSection.classList.add('hidden');
    voiceCloneSection.classList.add('hidden');
    voiceCloneProcessing.classList.add('hidden');
    voiceCloneResults.classList.add('hidden');
    imageGenSection.classList.add('hidden');
    imageGenProcessing.classList.add('hidden');
    imageGenResults.classList.add('hidden');

    // Show/hide sections based on mode
    if (mode === 'transcription') {
        uploadSection.classList.remove('hidden');
    } else if (mode === 'voice-clone') {
        voiceCloneSection.classList.remove('hidden');
        // Load saved voices when switching to voice clone mode
        loadSavedVoices();
    } else if (mode === 'image-gen') {
        imageGenSection.classList.remove('hidden');
    }
}

// Reference Audio Tab Switching
function switchRefAudioTab(tab) {
    refUploadTab.classList.toggle('active', tab === 'upload');
    refRecordTab.classList.toggle('active', tab === 'record');
    refUploadContent.classList.toggle('active', tab === 'upload');
    refRecordContent.classList.toggle('active', tab === 'record');
}

// Handle Reference File Selection
function handleRefFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        loadRefAudioFile(file);
    }
}

function handleRefFileDrop(event) {
    event.preventDefault();
    refUploadArea.classList.remove('dragover');
    const file = event.dataTransfer.files[0];
    if (file) {
        loadRefAudioFile(file);
    }
}

function loadRefAudioFile(file) {
    // Validate file type
    if (!file.type.startsWith('audio/')) {
        showToast('Por favor sube un archivo de audio válido', 'error');
        return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        showToast('El archivo es demasiado grande. Máximo 10MB.', 'error');
        return;
    }

    refAudioFile = file;
    refAudioBlob = null; // Clear any recorded audio

    // Show preview
    const url = URL.createObjectURL(file);
    refAudioPlayer.src = url;
    refUploadArea.classList.add('hidden');
    refAudioPreview.classList.remove('hidden');
}

function removeRefAudioFile() {
    refAudioFile = null;
    refAudioPlayer.src = '';
    refFileInput.value = '';
    refUploadArea.classList.remove('hidden');
    refAudioPreview.classList.add('hidden');
}

// Recording Functions
async function toggleRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
        recordedChunks = [];

        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) {
                recordedChunks.push(e.data);
            }
        };

        mediaRecorder.onstop = () => {
            stream.getTracks().forEach(track => track.stop());
            refAudioBlob = new Blob(recordedChunks, { type: 'audio/webm' });
            refAudioFile = null; // Clear any uploaded file

            // Show recorded preview
            const url = URL.createObjectURL(refAudioBlob);
            recordedAudioPlayer.src = url;
            recordedPreview.classList.remove('hidden');
        };

        mediaRecorder.start(100);
        recordingStartTime = Date.now();
        recordBtn.classList.add('recording');

        // Start timer
        recordingInterval = setInterval(() => {
            const elapsed = (Date.now() - recordingStartTime) / 1000;
            const minutes = Math.floor(elapsed / 60);
            const seconds = Math.floor(elapsed % 60);
            recordTimer.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')} / 01:00`;
            recordProgressBar.style.width = `${Math.min(100, (elapsed / 60) * 100)}%`;

            // Auto-stop at 60 seconds
            if (elapsed >= 60) {
                stopRecording();
            }
        }, 100);

    } catch (error) {
        console.error('Recording error:', error);
        showToast('No se pudo acceder al micrófono', 'error');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        recordBtn.classList.remove('recording');
        clearInterval(recordingInterval);
    }
}

function removeRecordedAudioFile() {
    refAudioBlob = null;
    recordedAudioPlayer.src = '';
    recordedPreview.classList.add('hidden');
    recordTimer.textContent = '00:00 / 01:00';
    recordProgressBar.style.width = '0%';
}

// Character Counter
function updateCharCount() {
    const count = targetTextInput.value.length;
    charCount.textContent = count;
}

// Update Model Hint
function updateModelHint() {
    modelHint.textContent = 'Rápido y buena calidad para la mayoría de casos.';
}

// Generate Voice Clone
async function generateVoiceClone() {
    // Get reference audio
    let audioToSend = null;
    if (refAudioFile) {
        audioToSend = refAudioFile;
    } else if (refAudioBlob) {
        audioToSend = new File([refAudioBlob], 'recording.webm', { type: 'audio/webm' });
    }

    if (!audioToSend) {
        showToast('Por favor sube o graba un audio de referencia', 'error');
        return;
    }

    const refText = refTextInput.value.trim();
    if (!refText) {
        showToast('Por favor escribe la transcripción del audio de referencia', 'error');
        return;
    }

    const targetText = targetTextInput.value.trim();
    if (!targetText) {
        showToast('Por favor escribe el texto a sintetizar', 'error');
        return;
    }

    if (targetText.length > 500) {
        showToast('El texto a sintetizar no puede exceder 500 caracteres', 'error');
        return;
    }

    const language = languageSelect.value;
    const ttsModel = ttsModelSelect.value;

    // Show processing
    voiceCloneSection.classList.add('hidden');
    voiceCloneProcessing.classList.remove('hidden');

    try {
        const formData = new FormData();
        formData.append('ref_audio', audioToSend);
        formData.append('ref_text', refText);
        formData.append('target_text', targetText);
        formData.append('language', language);
        formData.append('tts_model', ttsModel);

        const response = await fetch('/api/voice-clone', {
            method: 'POST',
            body: formData,
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || 'Error al generar la voz');
        }

        if (!result.success) {
            throw new Error(result.error || 'Error al generar la voz');
        }

        // Show results
        generatedAudioSessionId = result.audio_session_id;
        generatedAudioPlayer.src = `/api/audio/${result.audio_session_id}`;
        generatedDuration.textContent = formatDuration(result.duration || 0);

        voiceCloneProcessing.classList.add('hidden');
        voiceCloneResults.classList.remove('hidden');

        showToast('Voz generada exitosamente!', 'success');

    } catch (error) {
        console.error('Voice clone error:', error);
        showToast(error.message, 'error');
        voiceCloneProcessing.classList.add('hidden');
        voiceCloneSection.classList.remove('hidden');
    }
}

// Download Generated Voice
function downloadGeneratedVoice() {
    if (!generatedAudioSessionId) {
        showToast('No hay audio para descargar', 'error');
        return;
    }

    const a = document.createElement('a');
    a.href = `/api/audio/${generatedAudioSessionId}`;
    a.download = `voice-clone-${Date.now()}.wav`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    showToast('Descargado!', 'success');
}

// Reset Voice Clone
function resetVoiceClone() {
    // Clear reference audio
    refAudioFile = null;
    refAudioBlob = null;
    refAudioPlayer.src = '';
    refFileInput.value = '';
    refUploadArea.classList.remove('hidden');
    refAudioPreview.classList.add('hidden');
    recordedAudioPlayer.src = '';
    recordedPreview.classList.add('hidden');
    recordTimer.textContent = '00:00 / 01:00';
    recordProgressBar.style.width = '0%';

    // Clear text inputs
    refTextInput.value = '';
    targetTextInput.value = '';
    charCount.textContent = '0';
    languageSelect.value = 'Spanish';
    ttsModelSelect.value = 'qwen';
    voiceNameInput.value = '';
    updateModelHint();

    // Clear saved voice inputs
    savedVoiceTargetText.value = '';
    savedVoiceCharCount.textContent = '0';
    selectedVoiceId = null;
    updateSynthesizeButton();

    // Clear generated audio
    generatedAudioSessionId = null;
    generatedAudioPlayer.src = '';

    // Show voice clone section
    voiceCloneResults.classList.add('hidden');
    voiceCloneProcessing.classList.add('hidden');
    voiceCloneSection.classList.remove('hidden');

    // Switch to saved voices mode and reload
    switchVoiceMode('saved');
    switchRefAudioTab('upload');
}

// ============================================
// Image Generation Functions
// ============================================

// Character Counter for Image Prompt
function updateImageCharCount() {
    const count = imagePromptInput.value.length;
    imageCharCount.textContent = count;
}

// Generate Image
async function generateImage() {
    const prompt = imagePromptInput.value.trim();

    if (!prompt) {
        showToast('Por favor escribe una descripción de la imagen', 'error');
        return;
    }

    if (prompt.length > 500) {
        showToast('La descripción no puede exceder 500 caracteres', 'error');
        return;
    }

    const width = parseInt(imageWidthSelect.value);
    const height = parseInt(imageHeightSelect.value);

    // Show processing
    imageGenSection.classList.add('hidden');
    imageGenProcessing.classList.remove('hidden');

    try {
        const formData = new FormData();
        formData.append('prompt', prompt);
        formData.append('width', width);
        formData.append('height', height);

        const response = await fetch('/api/generate-image', {
            method: 'POST',
            body: formData,
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || 'Error al generar la imagen');
        }

        if (!result.success) {
            throw new Error(result.error || 'Error al generar la imagen');
        }

        // Show results
        generatedImageSessionId = result.image_session_id;
        generatedImagePreview.src = `/api/image/${result.image_session_id}`;
        generatedImageDimensions.textContent = `${result.width} x ${result.height} px`;

        imageGenProcessing.classList.add('hidden');
        imageGenResults.classList.remove('hidden');

        showToast('Imagen generada exitosamente!', 'success');

    } catch (error) {
        console.error('Image generation error:', error);
        showToast(error.message, 'error');
        imageGenProcessing.classList.add('hidden');
        imageGenSection.classList.remove('hidden');
    }
}

// Download Generated Image
function downloadGeneratedImage() {
    if (!generatedImageSessionId) {
        showToast('No hay imagen para descargar', 'error');
        return;
    }

    const a = document.createElement('a');
    a.href = `/api/image/${generatedImageSessionId}`;
    a.download = `generated-image-${Date.now()}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    showToast('Descargado!', 'success');
}

// Reset Image Generation
function resetImageGen() {
    // Clear prompt
    imagePromptInput.value = '';
    imageCharCount.textContent = '0';

    // Reset dimensions to defaults
    imageWidthSelect.value = '768';
    imageHeightSelect.value = '768';

    // Clear generated image
    generatedImageSessionId = null;
    generatedImagePreview.src = '';

    // Show image gen section
    imageGenResults.classList.add('hidden');
    imageGenProcessing.classList.add('hidden');
    imageGenSection.classList.remove('hidden');
}

// ============================================
// Saved Voices Functions
// ============================================

// Switch between saved and new voice modes
function switchVoiceMode(mode) {
    savedVoiceModeTab.classList.toggle('active', mode === 'saved');
    newVoiceModeTab.classList.toggle('active', mode === 'new');
    savedVoiceModeContent.classList.toggle('active', mode === 'saved');
    newVoiceModeContent.classList.toggle('active', mode === 'new');

    if (mode === 'saved') {
        loadSavedVoices();
    }
}

// Load saved voices from API
async function loadSavedVoices() {
    savedVoicesList.innerHTML = '<div class="loading-voices">Cargando voces...</div>';
    noVoicesMessage.classList.add('hidden');

    try {
        // Add timeout to prevent infinite loading
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 15000); // 15s timeout

        const response = await fetch('/api/voices', {
            signal: controller.signal
        });
        clearTimeout(timeoutId);

        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            throw new Error(data.detail || `Error ${response.status}`);
        }

        const data = await response.json();
        savedVoices = data.voices || [];
        renderSavedVoices();

    } catch (error) {
        console.error('Error loading voices:', error);
        let errorMsg = 'Error al cargar voces';
        if (error.name === 'AbortError') {
            errorMsg = 'Timeout al cargar voces. El servicio puede estar iniciando.';
        } else if (error.message) {
            errorMsg = `Error: ${error.message}`;
        }

        // Show error in the list area with retry button
        savedVoicesList.innerHTML = `
            <div class="error-message">
                ${errorMsg}
                <button class="btn btn-secondary retry-btn" onclick="loadSavedVoices()" style="margin-top: 0.5rem;">
                    Reintentar
                </button>
            </div>`;

        // Show the "create new voice" option even on error
        const msgElement = noVoicesMessage.querySelector('p');
        if (msgElement) {
            msgElement.textContent = 'No se pudieron cargar las voces guardadas.';
        }
        noVoicesMessage.classList.remove('hidden');

        // Disable synthesize button since we can't select a voice
        synthesizeBtn.disabled = true;
    }
}

// Render saved voices list
function renderSavedVoices() {
    if (savedVoices.length === 0) {
        savedVoicesList.innerHTML = '';
        noVoicesMessage.classList.remove('hidden');
        synthesizeBtn.disabled = true;
        return;
    }

    noVoicesMessage.classList.add('hidden');
    savedVoicesList.innerHTML = savedVoices.map(voice => `
        <div class="saved-voice-item ${selectedVoiceId === voice.id ? 'selected' : ''}" data-voice-id="${voice.id}">
            <div class="saved-voice-info">
                <div class="saved-voice-name">${escapeHtml(voice.name)}</div>
                <div class="saved-voice-meta">
                    <span class="saved-voice-language">${voice.language}</span>
                    <span class="saved-voice-date">${formatDate(voice.created_at)}</span>
                </div>
                <div class="saved-voice-ref-text">"${escapeHtml(truncateText(voice.ref_text, 60))}"</div>
            </div>
            <button class="btn-icon delete-voice-btn" title="Eliminar voz" data-voice-id="${voice.id}">
                <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
                    <path d="M3 5H15M6 5V4C6 3.45 6.45 3 7 3H11C11.55 3 12 3.45 12 4V5M7 8V13M11 8V13M4 5L5 15C5 15.55 5.45 16 6 16H12C12.55 16 13 15.55 13 15L14 5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </button>
        </div>
    `).join('');

    // Add click handlers
    savedVoicesList.querySelectorAll('.saved-voice-item').forEach(item => {
        item.addEventListener('click', (e) => {
            if (!e.target.closest('.delete-voice-btn')) {
                selectVoice(item.dataset.voiceId);
            }
        });
    });

    // Add delete handlers
    savedVoicesList.querySelectorAll('.delete-voice-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            deleteVoice(btn.dataset.voiceId);
        });
    });

    updateSynthesizeButton();
}

// Select a voice
function selectVoice(voiceId) {
    selectedVoiceId = voiceId;

    // Update UI
    savedVoicesList.querySelectorAll('.saved-voice-item').forEach(item => {
        item.classList.toggle('selected', item.dataset.voiceId === voiceId);
    });

    updateSynthesizeButton();
}

// Update synthesize button state
function updateSynthesizeButton() {
    const hasVoice = selectedVoiceId !== null;
    const hasText = savedVoiceTargetText.value.trim().length > 0;
    synthesizeBtn.disabled = !hasVoice || !hasText;
}

// Update character count for saved voice target text
function updateSavedVoiceCharCount() {
    const count = savedVoiceTargetText.value.length;
    savedVoiceCharCount.textContent = count;
    updateSynthesizeButton();
}

// Synthesize with saved voice
async function synthesizeWithSavedVoice() {
    if (!selectedVoiceId) {
        showToast('Por favor selecciona una voz', 'error');
        return;
    }

    const targetText = savedVoiceTargetText.value.trim();
    if (!targetText) {
        showToast('Por favor escribe el texto a sintetizar', 'error');
        return;
    }

    if (targetText.length > 500) {
        showToast('El texto no puede exceder 500 caracteres', 'error');
        return;
    }

    // Show processing
    voiceCloneSection.classList.add('hidden');
    voiceCloneProcessing.classList.remove('hidden');

    try {
        const formData = new FormData();
        formData.append('voice_id', selectedVoiceId);
        formData.append('target_text', targetText);

        const response = await fetch('/api/synthesize', {
            method: 'POST',
            body: formData,
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || 'Error al sintetizar');
        }

        if (!result.success) {
            throw new Error(result.error || 'Error al sintetizar');
        }

        // Show results
        generatedAudioSessionId = result.audio_session_id;
        generatedAudioPlayer.src = `/api/audio/${result.audio_session_id}`;
        generatedDuration.textContent = formatDuration(result.duration || 0);

        voiceCloneProcessing.classList.add('hidden');
        voiceCloneResults.classList.remove('hidden');

        showToast('Audio generado exitosamente!', 'success');

    } catch (error) {
        console.error('Synthesize error:', error);
        showToast(error.message, 'error');
        voiceCloneProcessing.classList.add('hidden');
        voiceCloneSection.classList.remove('hidden');
    }
}

// Save a new voice
async function saveVoice() {
    // Get voice name
    const voiceName = voiceNameInput.value.trim();
    if (!voiceName) {
        showToast('Por favor ingresa un nombre para la voz', 'error');
        return;
    }

    if (voiceName.length > 50) {
        showToast('El nombre no puede exceder 50 caracteres', 'error');
        return;
    }

    // Get reference audio
    let audioToSend = null;
    if (refAudioFile) {
        audioToSend = refAudioFile;
    } else if (refAudioBlob) {
        audioToSend = new File([refAudioBlob], 'recording.webm', { type: 'audio/webm' });
    }

    if (!audioToSend) {
        showToast('Por favor sube o graba un audio de referencia', 'error');
        return;
    }

    // Get reference text
    const refText = refTextInput.value.trim();
    if (!refText) {
        showToast('Por favor escribe la transcripción del audio de referencia', 'error');
        return;
    }

    const language = languageSelect.value;

    // Show processing
    voiceCloneSection.classList.add('hidden');
    voiceCloneProcessing.classList.remove('hidden');

    try {
        const formData = new FormData();
        formData.append('name', voiceName);
        formData.append('ref_audio', audioToSend);
        formData.append('ref_text', refText);
        formData.append('language', language);

        const response = await fetch('/api/voices', {
            method: 'POST',
            body: formData,
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || 'Error al guardar la voz');
        }

        if (!result.success) {
            throw new Error(result.error || 'Error al guardar la voz');
        }

        showToast('Voz guardada exitosamente!', 'success');

        // Clear the form
        voiceNameInput.value = '';

        // Switch to saved voices tab and reload
        voiceCloneProcessing.classList.add('hidden');
        voiceCloneSection.classList.remove('hidden');
        switchVoiceMode('saved');

    } catch (error) {
        console.error('Save voice error:', error);
        showToast(error.message, 'error');
        voiceCloneProcessing.classList.add('hidden');
        voiceCloneSection.classList.remove('hidden');
    }
}

// Delete a voice
async function deleteVoice(voiceId) {
    if (!confirm('¿Estás seguro de eliminar esta voz?')) {
        return;
    }

    try {
        const response = await fetch(`/api/voices/${voiceId}`, {
            method: 'DELETE',
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || 'Error al eliminar la voz');
        }

        // Clear selection if deleted voice was selected
        if (selectedVoiceId === voiceId) {
            selectedVoiceId = null;
        }

        showToast('Voz eliminada', 'success');
        loadSavedVoices();

    } catch (error) {
        console.error('Delete voice error:', error);
        showToast(error.message, 'error');
    }
}

// Helper: Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Helper: Truncate text
function truncateText(text, maxLength) {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

// Helper: Format date
function formatDate(isoString) {
    const date = new Date(isoString);
    return date.toLocaleDateString('es-ES', {
        day: '2-digit',
        month: '2-digit',
        year: 'numeric',
    });
}

// Initialize app
init();
