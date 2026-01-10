// DOM Elements
const uploadSection = document.getElementById('uploadSection');
const processingSection = document.getElementById('processingSection');
const resultsSection = document.getElementById('resultsSection');
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const processingStatus = document.getElementById('processingStatus');
const progressFill = document.getElementById('progressFill');
const segments = document.getElementById('segments');
const fullText = document.getElementById('fullText');
const duration = document.getElementById('duration');
const language = document.getElementById('language');
const audioPlayer = document.getElementById('audioPlayer');
const copyBtn = document.getElementById('copyBtn');
const downloadBtn = document.getElementById('downloadBtn');
const downloadSrtBtn = document.getElementById('downloadSrtBtn');
const downloadVttBtn = document.getElementById('downloadVttBtn');
const newTranscriptionBtn = document.getElementById('newTranscriptionBtn');
const toast = document.getElementById('toast');

// State
let currentTranscription = '';
let currentFile = null;
let currentSegments = []; // Store segments for subtitle export

// Initialize
function init() {
    setupEventListeners();
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
        console.log('Audio loaded successfully, duration:', audioPlayer.duration);
    });
    audioPlayer.addEventListener('error', (e) => {
        console.error('Audio player error:', e, audioPlayer.error);
        showToast('Could not load audio file for playback', 'error');
    });

    // Button clicks
    copyBtn.addEventListener('click', copyToClipboard);
    downloadBtn.addEventListener('click', downloadTranscription);
    downloadSrtBtn.addEventListener('click', downloadSRT);
    downloadVttBtn.addEventListener('click', downloadVTT);
    newTranscriptionBtn.addEventListener('click', resetApp);
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
    console.log('Stored file reference:', file.name, file.type, file.size);

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

        // Upload and transcribe with streaming
        await transcribeWithStreaming(formData);

    } catch (error) {
        console.error('Error:', error);
        showToast(`Error: ${error.message}`, 'error');
        showSection('upload');
    }
}

function validateFile(file) {
    // Check file type
    const validTypes = ['audio/mpeg', 'audio/wav', 'audio/x-wav', 'audio/wave', 'audio/mp4', 'audio/x-m4a', 'audio/flac', 'audio/ogg', 'audio/webm'];
    if (!validTypes.includes(file.type) && !file.name.match(/\.(mp3|wav|m4a|flac|ogg|webm)$/i)) {
        showToast('Please upload a valid audio file (MP3, WAV, M4A, FLAC, OGG)', 'error');
        return false;
    }

    // Check file size (100MB max)
    const maxSize = 100 * 1024 * 1024;
    if (file.size > maxSize) {
        showToast('File size exceeds 100MB limit', 'error');
        return false;
    }

    return true;
}

// Transcription with Streaming
async function transcribeWithStreaming(formData) {
    processingStatus.textContent = 'Starting transcription...';
    progressFill.style.width = '20%';

    // Reset results
    segments.innerHTML = '';
    fullText.textContent = '';
    currentTranscription = '';
    currentSegments = [];

    try {
        console.log('Sending transcription request...');
        const response = await fetch('/api/transcribe/stream', {
            method: 'POST',
            body: formData,
        });

        console.log('Response received:', response.status);
        console.log('Response body:', response.body);
        console.log('Response headers:', response.headers);

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Transcription failed');
        }

        // Keep processing section visible during streaming
        showSection('processing');
        processingStatus.textContent = 'Transcribing...';

        console.log('About to get reader...');

        if (!response.body) {
            throw new Error('Response body is null');
        }

        // Process SSE stream
        const reader = response.body.getReader();
        console.log('Got reader:', reader);
        const decoder = new TextDecoder();
        let buffer = '';
        let eventCount = 0;

        console.log('Starting to read stream...');

        while (true) {
            console.log('Reading next chunk...');
            const { done, value } = await reader.read();
            console.log('Read result - done:', done, 'value length:', value?.length);

            if (done) {
                console.log('Stream done. Total events:', eventCount);
                break;
            }

            // Decode chunk
            const chunk = decoder.decode(value, { stream: true });
            console.log('Received chunk (first 200 chars):', chunk.substring(0, 200));

            // Debug: Show actual bytes
            console.log('Chunk char codes at positions 40-50:',
                [...chunk.substring(40, 50)].map(c => c.charCodeAt(0)));

            buffer += chunk;

            // Process complete events (SSE format: event ends with double newline)
            // Normalize line endings first
            const normalizedBuffer = buffer.replace(/\r\n/g, '\n');
            const eventStrings = normalizedBuffer.split('\n\n');
            console.log('Split into', eventStrings.length, 'parts');

            // Last element might be incomplete, keep in buffer
            buffer = eventStrings[eventStrings.length - 1];
            console.log('Keeping last part in buffer, will process', eventStrings.length - 1, 'events');

            // Process all complete events (all except last)
            for (let i = 0; i < eventStrings.length - 1; i++) {
                const eventText = eventStrings[i];
                console.log('Event', i, ':', eventText.substring(0, 50), '...');
                if (!eventText.trim()) {
                    console.log('Skipping empty event');
                    continue;
                }

                console.log('Processing event text:', eventText.substring(0, 100));

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
                    console.log('Event #' + eventCount + ':', eventType, eventData.substring(0, 50));
                    handleStreamEvent(eventType, JSON.parse(eventData));
                }
            }
        }

        console.log('Stream processing complete');

    } catch (error) {
        console.error('Streaming error:', error);
        showToast(`Error: ${error.message}`, 'error');
        showSection('upload');
    }
}

// Handle Stream Events
function handleStreamEvent(eventType, data) {
    console.log('Handling event:', eventType, data);
    switch (eventType) {
        case 'metadata':
            // Update metadata
            if (data.duration) {
                duration.textContent = formatDuration(data.duration);
            }
            if (data.language) {
                language.textContent = data.language.toUpperCase();
            }
            processingStatus.textContent = 'Transcribing audio...';
            progressFill.style.width = '40%';
            console.log('Metadata updated');
            break;

        case 'progress':
            // Add segment
            console.log('Adding segment:', data.text);
            addSegment(data);
            // Update progress (estimate based on time)
            const progress = Math.min(90, 40 + (data.end / 600) * 50);
            progressFill.style.width = `${progress}%`;
            break;

        case 'complete':
            // Set final text
            console.log('Transcription complete, showing results');
            currentTranscription = data.text;
            fullText.textContent = data.text;
            progressFill.style.width = '100%';

            // Set audio player source using the preprocessed audio from backend
            console.log('Setting up audio player...');
            console.log('Audio session ID:', data.audio_session_id);

            if (data.audio_session_id) {
                // Use the preprocessed audio from the backend (16kHz mono WAV)
                const audioURL = `/api/audio/${data.audio_session_id}`;
                console.log('Audio URL:', audioURL);
                audioPlayer.src = audioURL;
                audioPlayer.load(); // Force the browser to load the audio metadata
                console.log('Audio player configured with preprocessed audio');
            } else {
                console.warn('No audio session ID provided, cannot load audio');
            }

            // Show results section now
            showSection('results');
            showToast('Transcription completed!', 'success');
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
        text: segmentData.text
    });

    const segmentEl = document.createElement('div');
    segmentEl.className = 'segment';

    const timeEl = document.createElement('div');
    timeEl.className = 'segment-time';
    timeEl.textContent = `${formatTime(segmentData.start)} - ${formatTime(segmentData.end)}`;

    const textEl = document.createElement('div');
    textEl.className = 'segment-text';
    textEl.textContent = segmentData.text;

    segmentEl.appendChild(timeEl);
    segmentEl.appendChild(textEl);
    segments.appendChild(segmentEl);

    // Scroll to bottom
    segmentEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
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
async function copyToClipboard() {
    try {
        await navigator.clipboard.writeText(currentTranscription);
        showToast('Copied to clipboard!', 'success');
    } catch (error) {
        showToast('Failed to copy', 'error');
    }
}

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
    showToast('Downloaded!', 'success');
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
        return `${index + 1}\n${formatTimeSRT(segment.start)} --> ${formatTimeSRT(segment.end)}\n${segment.text.trim()}\n`;
    }).join('\n');
}

// Generate VTT subtitle content
function generateVTT() {
    const header = 'WEBVTT\n\n';
    const cues = currentSegments.map((segment, index) => {
        return `${index + 1}\n${formatTimeVTT(segment.start)} --> ${formatTimeVTT(segment.end)}\n${segment.text.trim()}\n`;
    }).join('\n');
    return header + cues;
}

// Download SRT file
function downloadSRT() {
    if (currentSegments.length === 0) {
        showToast('No transcription data available', 'error');
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
    showToast('SRT downloaded!', 'success');
}

// Download VTT file
function downloadVTT() {
    if (currentSegments.length === 0) {
        showToast('No transcription data available', 'error');
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
    showToast('VTT downloaded!', 'success');
}

function resetApp() {
    showSection('upload');
    fileInput.value = '';
    currentFile = null;
    currentTranscription = '';
    currentSegments = [];
    segments.innerHTML = '';
    fullText.textContent = '';
    progressFill.style.width = '0%';

    // Reset audio player
    audioPlayer.src = '';
    audioPlayer.load();
}

// Initialize app
init();
