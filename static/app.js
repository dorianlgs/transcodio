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
const copyBtn = document.getElementById('copyBtn');
const downloadBtn = document.getElementById('downloadBtn');
const newTranscriptionBtn = document.getElementById('newTranscriptionBtn');
const toast = document.getElementById('toast');

// State
let currentTranscription = '';
let currentFile = null;

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

    // Button clicks
    copyBtn.addEventListener('click', copyToClipboard);
    downloadBtn.addEventListener('click', downloadTranscription);
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
        // Create form data
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

    try {
        const response = await fetch('/api/transcribe/stream', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Transcription failed');
        }

        // Show results section
        showSection('results');

        // Process SSE stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();

            if (done) break;

            // Decode chunk
            buffer += decoder.decode(value, { stream: true });

            // Process complete events
            const events = buffer.split('\n\n');
            buffer = events.pop(); // Keep incomplete event in buffer

            for (const eventText of events) {
                if (!eventText.trim()) continue;

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
            processingStatus.textContent = 'Transcribing audio...';
            progressFill.style.width = '40%';
            break;

        case 'progress':
            // Add segment
            addSegment(data);
            // Update progress (estimate based on time)
            const progress = Math.min(90, 40 + (data.end / 600) * 50);
            progressFill.style.width = `${progress}%`;
            break;

        case 'complete':
            // Set final text
            currentTranscription = data.text;
            fullText.textContent = data.text;
            progressFill.style.width = '100%';
            showToast('Transcription completed!', 'success');
            break;

        case 'error':
            showToast(`Error: ${data.error}`, 'error');
            showSection('upload');
            break;
    }
}

// Add Segment to UI
function addSegment(segmentData) {
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

function resetApp() {
    showSection('upload');
    fileInput.value = '';
    currentFile = null;
    currentTranscription = '';
    segments.innerHTML = '';
    fullText.textContent = '';
    progressFill.style.width = '0%';
}

// Initialize app
init();
