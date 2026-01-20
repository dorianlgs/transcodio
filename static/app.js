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

// State
let currentTranscription = '';
let currentFile = null;
let currentSegments = []; // Store segments for subtitle export
let currentMinutes = null; // Store minutes data

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
    downloadMinutesBtn.addEventListener('click', downloadMinutes);
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
        showToast('Please upload a valid audio file (MP3, WAV, M4A, FLAC, OGG, MP4)', 'error');
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
async function transcribeWithStreaming(formData, enableDiarization = false, enableMinutes = false) {
    processingStatus.textContent = 'Starting transcription...';
    progressFill.style.width = '20%';

    // Reset results
    segments.innerHTML = '';
    fullText.textContent = '';
    currentTranscription = '';
    currentSegments = [];
    currentMinutes = null;

    // Reset minutes section
    minutesSection.classList.add('hidden');
    minutesLoading.classList.add('hidden');
    minutesContent.classList.remove('hidden');
    minutesSummary.textContent = '';
    minutesKeyPoints.innerHTML = '';
    minutesDecisions.innerHTML = '';
    minutesActions.innerHTML = '';
    minutesParticipants.innerHTML = '';

    // Show minutes section with loading if enabled
    if (enableMinutes) {
        minutesSection.classList.remove('hidden');
        minutesLoading.classList.remove('hidden');
        minutesContent.classList.add('hidden');
    }

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

        case 'speakers_ready':
            // Update segments with speaker labels
            updateSegmentsWithSpeakers(data.segments);
            showToast('Speaker identification completed', 'success');
            break;

        case 'complete':
            // Set final text
            currentTranscription = data.text;
            fullText.textContent = data.text;
            progressFill.style.width = '100%';

            // Set audio player source using the preprocessed audio from backend
            if (data.audio_session_id) {
                // Use the preprocessed audio from the backend (16kHz mono WAV)
                const audioURL = `/api/audio/${data.audio_session_id}`;
                audioPlayer.src = audioURL;
                audioPlayer.load(); // Force the browser to load the audio metadata
            } else {
                console.warn('No audio session ID provided, cannot load audio');
            }

            // Show results section now
            showSection('results');
            showToast('Transcription completed!', 'success');
            break;

        case 'minutes_ready':
            // Display meeting minutes
            displayMeetingMinutes(data.minutes);
            showToast('Meeting minutes generated!', 'success');
            break;

        case 'minutes_error':
            // Handle minutes generation error
            minutesLoading.classList.add('hidden');
            minutesContent.classList.remove('hidden');
            minutesSummary.textContent = 'Unable to generate meeting minutes. ' + (data.error || '');
            showToast('Failed to generate minutes', 'error');
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

// Display Meeting Minutes
function displayMeetingMinutes(minutes) {
    currentMinutes = minutes;

    // Hide loading, show content
    minutesLoading.classList.add('hidden');
    minutesContent.classList.remove('hidden');

    // Executive Summary
    minutesSummary.textContent = minutes.executive_summary || 'No summary available.';

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
        li.textContent = 'No key discussion points identified.';
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
        li.textContent = 'No decisions recorded.';
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
                <span class="action-task">${item.task || 'No task specified'}</span>
                <span class="action-meta">
                    <span class="action-assignee">${item.assignee || 'Unassigned'}</span>
                    <span class="action-deadline">${item.deadline || 'No deadline'}</span>
                </span>
            `;
            minutesActions.appendChild(li);
        });
    } else {
        const li = document.createElement('li');
        li.textContent = 'No action items identified.';
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
        span.textContent = 'No participants mentioned by name.';
        span.className = 'empty-item';
        minutesParticipants.appendChild(span);
    }
}

// Download Meeting Minutes as TXT
function downloadMinutes() {
    if (!currentMinutes) {
        showToast('No meeting minutes available', 'error');
        return;
    }

    let content = 'MEETING MINUTES\n';
    content += '=' .repeat(50) + '\n\n';

    content += 'EXECUTIVE SUMMARY\n';
    content += '-'.repeat(30) + '\n';
    content += (currentMinutes.executive_summary || 'No summary available.') + '\n\n';

    content += 'KEY DISCUSSION POINTS\n';
    content += '-'.repeat(30) + '\n';
    if (currentMinutes.key_discussion_points && currentMinutes.key_discussion_points.length > 0) {
        currentMinutes.key_discussion_points.forEach((point, i) => {
            content += `${i + 1}. ${point}\n`;
        });
    } else {
        content += 'No key discussion points identified.\n';
    }
    content += '\n';

    content += 'DECISIONS MADE\n';
    content += '-'.repeat(30) + '\n';
    if (currentMinutes.decisions_made && currentMinutes.decisions_made.length > 0) {
        currentMinutes.decisions_made.forEach((decision, i) => {
            content += `${i + 1}. ${decision}\n`;
        });
    } else {
        content += 'No decisions recorded.\n';
    }
    content += '\n';

    content += 'ACTION ITEMS\n';
    content += '-'.repeat(30) + '\n';
    if (currentMinutes.action_items && currentMinutes.action_items.length > 0) {
        currentMinutes.action_items.forEach((item, i) => {
            content += `${i + 1}. ${item.task || 'No task specified'}\n`;
            content += `   Assignee: ${item.assignee || 'Unassigned'}\n`;
            content += `   Deadline: ${item.deadline || 'No deadline'}\n`;
        });
    } else {
        content += 'No action items identified.\n';
    }
    content += '\n';

    content += 'PARTICIPANTS MENTIONED\n';
    content += '-'.repeat(30) + '\n';
    if (currentMinutes.participants_mentioned && currentMinutes.participants_mentioned.length > 0) {
        content += currentMinutes.participants_mentioned.join(', ') + '\n';
    } else {
        content += 'No participants mentioned by name.\n';
    }

    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `meeting-minutes-${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast('Minutes downloaded!', 'success');
}

function resetApp() {
    showSection('upload');
    fileInput.value = '';
    currentFile = null;
    currentTranscription = '';
    currentSegments = [];
    currentMinutes = null;
    segments.innerHTML = '';
    fullText.textContent = '';
    progressFill.style.width = '0%';

    // Reset audio player
    audioPlayer.src = '';
    audioPlayer.load();

    // Reset minutes section
    minutesSection.classList.add('hidden');
    minutesLoading.classList.add('hidden');
    minutesContent.classList.remove('hidden');
    minutesSummary.textContent = '';
    minutesKeyPoints.innerHTML = '';
    minutesDecisions.innerHTML = '';
    minutesActions.innerHTML = '';
    minutesParticipants.innerHTML = '';

    // Reset checkboxes
    enableDiarizationCheckbox.checked = false;
    enableMinutesCheckbox.checked = false;
}

// Initialize app
init();
