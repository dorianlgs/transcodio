// ============================================
// i18n System
// ============================================

const translations = {
    en: {
        // Mode buttons
        'mode.transcribe': 'Transcribe',
        'mode.voice_clone': 'Clone Voice',
        'mode.image_gen': 'Generate Image',

        // Upload section
        'upload.title': 'Drag your audio file here',
        'upload.subtitle': 'or click to select',
        'upload.file_info': 'Supports MP3, WAV, M4A, FLAC, OGG \u2022 Max 100MB \u2022 Up to 60 minutes',
        'upload.identify_speakers': 'Identify speakers',
        'upload.generate_minutes': 'Generate meeting minutes',

        // Processing section
        'processing.title': 'Transcribing your audio...',
        'processing.uploading': 'Uploading file...',
        'processing.starting': 'Starting transcription...',
        'processing.transcribing': 'Transcribing audio...',

        // Results section
        'results.title': 'Transcription Results',
        'results.new_transcription': 'New Transcription',
        'results.tab_transcription': 'Transcription',
        'results.tab_minutes': 'Minutes',

        // Minutes section
        'minutes.title': 'Meeting Minutes',
        'minutes.download': 'Download Minutes',
        'minutes.loading': 'Generating meeting minutes...',
        'minutes.executive_summary': 'Executive Summary',
        'minutes.key_points': 'Key Discussion Points',
        'minutes.decisions': 'Decisions Made',
        'minutes.action_items': 'Action Items',
        'minutes.participants': 'Participants Mentioned',
        'minutes.no_summary': 'No summary available.',
        'minutes.no_key_points': 'No key points identified.',
        'minutes.no_decisions': 'No decisions recorded.',
        'minutes.no_task': 'No task specified',
        'minutes.assignee_label': 'Assignee:',
        'minutes.unassigned': 'Unassigned',
        'minutes.date_label': 'Date:',
        'minutes.date_tbd': 'TBD',
        'minutes.no_actions': 'No pending actions identified.',
        'minutes.no_participants': 'No participants mentioned by name.',
        'minutes.could_not_generate': 'Could not generate minutes. ',
        'minutes.download_header': 'MEETING MINUTES',
        'minutes.download_executive_summary': 'EXECUTIVE SUMMARY',
        'minutes.download_key_points': 'KEY DISCUSSION POINTS',
        'minutes.download_decisions': 'DECISIONS MADE',
        'minutes.download_actions': 'ACTION ITEMS',
        'minutes.download_participants': 'PARTICIPANTS MENTIONED',

        // Voice clone section
        'voice.title': 'Clone Voice',
        'voice.subtitle': 'Use a saved voice or create a new one',
        'voice.saved_tab': 'Saved Voices',
        'voice.new_tab': 'New Voice',
        'voice.select_voice': 'Select a voice',
        'voice.loading': 'Loading voices...',
        'voice.no_voices': 'No saved voices.',
        'voice.create_new': 'Create New Voice',
        'voice.target_label': 'Text to Synthesize (max. 50000 characters)',
        'voice.target_placeholder': 'Write the text you want to generate with the selected voice...',
        'voice.synthesize': 'Synthesize',
        'voice.ref_audio_label': 'Reference Audio (3-30 seconds)',
        'voice.upload_tab': 'Upload File',
        'voice.record_tab': 'Record',
        'voice.upload_drag': 'Drag or click to upload',
        'voice.ref_text_label': 'Reference Audio Transcription',
        'voice.ref_text_placeholder': 'Write exactly what the reference audio says...',
        'voice.new_target_label': 'Text to Synthesize (max. 50000 characters)',
        'voice.new_target_placeholder': 'Write the text you want to generate with the cloned voice...',
        'voice.model_label': 'TTS Model',
        'voice.model_qwen': 'Qwen3-TTS (Fast, 1.7B)',
        'voice.model_hint': 'Fast and good quality for most cases',
        'voice.language_label': 'Language',
        'voice.name_label': 'Voice Name (for saving)',
        'voice.name_placeholder': 'E.g.: My voice, John\'s voice...',
        'voice.generate': 'Generate Voice',
        'voice.save': 'Save Voice',
        'voice.processing_title': 'Generating cloned voice...',
        'voice.processing_subtitle': 'This may take a few seconds',
        'voice.result_title': 'Generated Voice',
        'voice.download_wav': 'Download WAV',
        'voice.new_clone': 'New Clone',

        // Language options
        'lang.spanish': 'Spanish',
        'lang.english': 'English',
        'lang.chinese': 'Chinese',
        'lang.japanese': 'Japanese',
        'lang.korean': 'Korean',
        'lang.german': 'German',
        'lang.french': 'French',
        'lang.russian': 'Russian',
        'lang.portuguese': 'Portuguese',
        'lang.italian': 'Italian',

        // Image generation section
        'image.title': 'Generate Image',
        'image.subtitle': 'Describe the image you want to create with text',
        'image.prompt_label': 'Image description (max. 500 characters)',
        'image.prompt_placeholder': 'Describe the image you want to generate. For example: A futuristic landscape with crystal skyscrapers at sunset...',
        'image.dimensions': 'Dimensions',
        'image.width': 'Width',
        'image.height': 'Height',
        'image.generate': 'Generate Image',
        'image.processing_title': 'Generating image...',
        'image.processing_subtitle': 'This may take a few seconds',
        'image.result_title': 'Generated Image',
        'image.download_png': 'Download PNG',
        'image.new_image': 'New Image',
        'image.alt': 'Generated image',

        // Footer
        'footer.developed_by': 'Developed by',

        // Toast messages
        'toast.audio_load_error': 'Could not load audio file',
        'toast.invalid_audio': 'Please upload a valid audio file (MP3, WAV, M4A, FLAC, OGG, MP4)',
        'toast.file_too_large': 'File exceeds 100MB limit',
        'toast.speakers_done': 'Speaker identification completed',
        'toast.transcription_done': 'Transcription completed!',
        'toast.minutes_done': 'Minutes generated!',
        'toast.minutes_error': 'Error generating minutes',
        'toast.downloaded': 'Downloaded!',
        'toast.no_transcription_data': 'No transcription data available',
        'toast.srt_downloaded': 'SRT downloaded!',
        'toast.vtt_downloaded': 'VTT downloaded!',
        'toast.no_minutes': 'No minutes available',
        'toast.minutes_downloaded': 'Minutes downloaded!',
        'toast.ref_audio_invalid': 'Please upload a valid audio file',
        'toast.ref_audio_too_large': 'File is too large. Max 15MB.',
        'toast.no_ref_audio': 'Please upload or record a reference audio',
        'toast.no_ref_text': 'Please write the reference audio transcription',
        'toast.no_target_text': 'Please write the text to synthesize',
        'toast.target_too_long': 'Text to synthesize cannot exceed 50000 characters',
        'toast.voice_generated': 'Voice generated successfully!',
        'toast.voice_gen_error': 'Error generating voice',
        'toast.no_audio_download': 'No audio to download',
        'toast.no_image_prompt': 'Please write an image description',
        'toast.prompt_too_long': 'Description cannot exceed 500 characters',
        'toast.image_generated': 'Image generated successfully!',
        'toast.image_gen_error': 'Error generating image',
        'toast.no_image_download': 'No image to download',
        'toast.mic_error': 'Could not access microphone',
        'toast.select_voice': 'Please select a voice',
        'toast.synth_error': 'Error synthesizing',
        'toast.audio_generated': 'Audio generated successfully!',
        'toast.voice_name_required': 'Please enter a name for the voice',
        'toast.voice_name_too_long': 'Name cannot exceed 50 characters',
        'toast.voice_saved': 'Voice saved successfully!',
        'toast.voice_save_error': 'Error saving voice',
        'toast.voice_deleted': 'Voice deleted',
        'toast.voice_delete_error': 'Error deleting voice',
        'toast.voices_load_error': 'Error loading voices',
        'toast.voices_timeout': 'Timeout loading voices. Service may be starting.',
        'toast.voices_load_failed': 'Could not load saved voices.',

        // Confirm dialogs
        'confirm.delete_voice': 'Are you sure you want to delete this voice?',

        // Misc
        'misc.retry': 'Retry',
        'misc.remove': 'Remove',
        'misc.delete_voice': 'Delete voice',
        'misc.no_audio_session': 'No audio session ID provided, cannot load audio',
    },
    es: {
        // Mode buttons
        'mode.transcribe': 'Transcribir',
        'mode.voice_clone': 'Clonar Voz',
        'mode.image_gen': 'Generar Imagen',

        // Upload section
        'upload.title': 'Arrastra tu archivo de audio aqu\u00ed',
        'upload.subtitle': 'o haz clic para seleccionar',
        'upload.file_info': 'Soporta MP3, WAV, M4A, FLAC, OGG \u2022 M\u00e1x 100MB \u2022 Hasta 60 minutos',
        'upload.identify_speakers': 'Identificar hablantes',
        'upload.generate_minutes': 'Generar minuta de reuni\u00f3n',

        // Processing section
        'processing.title': 'Transcribiendo tu audio...',
        'processing.uploading': 'Subiendo archivo...',
        'processing.starting': 'Iniciando transcripci\u00f3n...',
        'processing.transcribing': 'Transcribiendo audio...',

        // Results section
        'results.title': 'Resultados de Transcripci\u00f3n',
        'results.new_transcription': 'Nueva Transcripci\u00f3n',
        'results.tab_transcription': 'Transcripci\u00f3n',
        'results.tab_minutes': 'Minuta',

        // Minutes section
        'minutes.title': 'Minuta de Reuni\u00f3n',
        'minutes.download': 'Descargar Minuta',
        'minutes.loading': 'Generando minuta de reuni\u00f3n...',
        'minutes.executive_summary': 'Resumen Ejecutivo',
        'minutes.key_points': 'Puntos Clave Discutidos',
        'minutes.decisions': 'Decisiones Tomadas',
        'minutes.action_items': 'Acciones Pendientes',
        'minutes.participants': 'Participantes Mencionados',
        'minutes.no_summary': 'No hay resumen disponible.',
        'minutes.no_key_points': 'No se identificaron puntos clave.',
        'minutes.no_decisions': 'No se registraron decisiones.',
        'minutes.no_task': 'Sin tarea especificada',
        'minutes.assignee_label': 'Responsable:',
        'minutes.unassigned': 'Sin asignar',
        'minutes.date_label': 'Fecha:',
        'minutes.date_tbd': 'Por definir',
        'minutes.no_actions': 'No se identificaron acciones pendientes.',
        'minutes.no_participants': 'No se mencionaron participantes por nombre.',
        'minutes.could_not_generate': 'No se pudo generar la minuta. ',
        'minutes.download_header': 'MINUTA DE REUNI\u00d3N',
        'minutes.download_executive_summary': 'RESUMEN EJECUTIVO',
        'minutes.download_key_points': 'PUNTOS CLAVE DISCUTIDOS',
        'minutes.download_decisions': 'DECISIONES TOMADAS',
        'minutes.download_actions': 'ACCIONES PENDIENTES',
        'minutes.download_participants': 'PARTICIPANTES MENCIONADOS',

        // Voice clone section
        'voice.title': 'Clonar Voz',
        'voice.subtitle': 'Usa una voz guardada o crea una nueva',
        'voice.saved_tab': 'Voces Guardadas',
        'voice.new_tab': 'Nueva Voz',
        'voice.select_voice': 'Selecciona una voz',
        'voice.loading': 'Cargando voces...',
        'voice.no_voices': 'No hay voces guardadas.',
        'voice.create_new': 'Crear Nueva Voz',
        'voice.target_label': 'Texto a Sintetizar (max. 50000 caracteres)',
        'voice.target_placeholder': 'Escribe el texto que quieres generar con la voz seleccionada...',
        'voice.synthesize': 'Sintetizar',
        'voice.ref_audio_label': 'Audio de Referencia (3-30 segundos)',
        'voice.upload_tab': 'Subir Archivo',
        'voice.record_tab': 'Grabar',
        'voice.upload_drag': 'Arrastra o haz clic para subir',
        'voice.ref_text_label': 'Transcripci\u00f3n del Audio de Referencia',
        'voice.ref_text_placeholder': 'Escribe exactamente lo que dice el audio de referencia...',
        'voice.new_target_label': 'Texto a Sintetizar (max. 50000 caracteres)',
        'voice.new_target_placeholder': 'Escribe el texto que quieres generar con la voz clonada...',
        'voice.model_label': 'Modelo TTS',
        'voice.model_qwen': 'Qwen3-TTS (R\u00e1pido, 1.7B)',
        'voice.model_hint': 'R\u00e1pido y buena calidad para la mayor\u00eda de casos',
        'voice.language_label': 'Idioma',
        'voice.name_label': 'Nombre de la Voz (para guardar)',
        'voice.name_placeholder': 'Ej: Mi voz, Voz de Juan...',
        'voice.generate': 'Generar Voz',
        'voice.save': 'Guardar Voz',
        'voice.processing_title': 'Generando voz clonada...',
        'voice.processing_subtitle': 'Esto puede tardar unos segundos',
        'voice.result_title': 'Voz Generada',
        'voice.download_wav': 'Descargar WAV',
        'voice.new_clone': 'Nueva Clonaci\u00f3n',

        // Language options
        'lang.spanish': 'Espa\u00f1ol',
        'lang.english': 'Ingl\u00e9s',
        'lang.chinese': 'Chino',
        'lang.japanese': 'Japon\u00e9s',
        'lang.korean': 'Coreano',
        'lang.german': 'Alem\u00e1n',
        'lang.french': 'Franc\u00e9s',
        'lang.russian': 'Ruso',
        'lang.portuguese': 'Portugu\u00e9s',
        'lang.italian': 'Italiano',

        // Image generation section
        'image.title': 'Generar Imagen',
        'image.subtitle': 'Describe la imagen que deseas crear con texto',
        'image.prompt_label': 'Descripci\u00f3n de la imagen (max. 500 caracteres)',
        'image.prompt_placeholder': 'Describe la imagen que quieres generar. Por ejemplo: Un paisaje futurista con rascacielos de cristal al atardecer...',
        'image.dimensions': 'Dimensiones',
        'image.width': 'Ancho',
        'image.height': 'Alto',
        'image.generate': 'Generar Imagen',
        'image.processing_title': 'Generando imagen...',
        'image.processing_subtitle': 'Esto puede tardar unos segundos',
        'image.result_title': 'Imagen Generada',
        'image.download_png': 'Descargar PNG',
        'image.new_image': 'Nueva Imagen',
        'image.alt': 'Imagen generada',

        // Footer
        'footer.developed_by': 'Desarrollado por',

        // Toast messages
        'toast.audio_load_error': 'No se pudo cargar el archivo de audio',
        'toast.invalid_audio': 'Por favor sube un archivo de audio v\u00e1lido (MP3, WAV, M4A, FLAC, OGG, MP4)',
        'toast.file_too_large': 'El archivo excede el l\u00edmite de 100MB',
        'toast.speakers_done': 'Identificaci\u00f3n de hablantes completada',
        'toast.transcription_done': 'Transcripci\u00f3n completada!',
        'toast.minutes_done': 'Minuta generada!',
        'toast.minutes_error': 'Error al generar la minuta',
        'toast.downloaded': 'Descargado!',
        'toast.no_transcription_data': 'No hay datos de transcripci\u00f3n disponibles',
        'toast.srt_downloaded': 'SRT descargado!',
        'toast.vtt_downloaded': 'VTT descargado!',
        'toast.no_minutes': 'No hay minuta disponible',
        'toast.minutes_downloaded': 'Minuta descargada!',
        'toast.ref_audio_invalid': 'Por favor sube un archivo de audio v\u00e1lido',
        'toast.ref_audio_too_large': 'El archivo es demasiado grande. M\u00e1ximo 15MB.',
        'toast.no_ref_audio': 'Por favor sube o graba un audio de referencia',
        'toast.no_ref_text': 'Por favor escribe la transcripci\u00f3n del audio de referencia',
        'toast.no_target_text': 'Por favor escribe el texto a sintetizar',
        'toast.target_too_long': 'El texto a sintetizar no puede exceder 50000 caracteres',
        'toast.voice_generated': 'Voz generada exitosamente!',
        'toast.voice_gen_error': 'Error al generar la voz',
        'toast.no_audio_download': 'No hay audio para descargar',
        'toast.no_image_prompt': 'Por favor escribe una descripci\u00f3n de la imagen',
        'toast.prompt_too_long': 'La descripci\u00f3n no puede exceder 500 caracteres',
        'toast.image_generated': 'Imagen generada exitosamente!',
        'toast.image_gen_error': 'Error al generar la imagen',
        'toast.no_image_download': 'No hay imagen para descargar',
        'toast.mic_error': 'No se pudo acceder al micr\u00f3fono',
        'toast.select_voice': 'Por favor selecciona una voz',
        'toast.synth_error': 'Error al sintetizar',
        'toast.audio_generated': 'Audio generado exitosamente!',
        'toast.voice_name_required': 'Por favor ingresa un nombre para la voz',
        'toast.voice_name_too_long': 'El nombre no puede exceder 50 caracteres',
        'toast.voice_saved': 'Voz guardada exitosamente!',
        'toast.voice_save_error': 'Error al guardar la voz',
        'toast.voice_deleted': 'Voz eliminada',
        'toast.voice_delete_error': 'Error al eliminar la voz',
        'toast.voices_load_error': 'Error al cargar voces',
        'toast.voices_timeout': 'Timeout al cargar voces. El servicio puede estar iniciando.',
        'toast.voices_load_failed': 'No se pudieron cargar las voces guardadas.',

        // Confirm dialogs
        'confirm.delete_voice': '\u00bfEst\u00e1s seguro de eliminar esta voz?',

        // Misc
        'misc.retry': 'Reintentar',
        'misc.remove': 'Eliminar',
        'misc.delete_voice': 'Eliminar voz',
        'misc.no_audio_session': 'No se proporcion\u00f3 ID de sesi\u00f3n de audio, no se puede cargar el audio',
    }
};

let currentLanguage = localStorage.getItem('transcodio-lang') || 'en';

function t(key) {
    const lang = translations[currentLanguage] || translations.en;
    return lang[key] || translations.en[key] || key;
}

function applyTranslations() {
    // Text content
    document.querySelectorAll('[data-i18n]').forEach(el => {
        el.textContent = t(el.dataset.i18n);
    });
    // Placeholders
    document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
        el.placeholder = t(el.dataset.i18nPlaceholder);
    });
    // Alt text
    document.querySelectorAll('[data-i18n-alt]').forEach(el => {
        el.alt = t(el.dataset.i18nAlt);
    });
    // Title attributes
    document.querySelectorAll('[data-i18n-title]').forEach(el => {
        el.title = t(el.dataset.i18nTitle);
    });
    // Update html lang attribute
    document.documentElement.lang = currentLanguage;
    // Update language toggle button text
    const langToggle = document.getElementById('langToggle');
    if (langToggle) {
        langToggle.textContent = currentLanguage === 'en' ? 'ES' : 'EN';
    }
}

function setLanguage(lang) {
    currentLanguage = lang;
    localStorage.setItem('transcodio-lang', lang);
    applyTranslations();
    // Update date formatting locale
    if (savedVoices.length > 0) {
        renderSavedVoices();
    }
}

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
    applyTranslations();
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
        showToast(t('toast.audio_load_error'), 'error');
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

    // Language toggle
    const langToggle = document.getElementById('langToggle');
    if (langToggle) {
        langToggle.addEventListener('click', () => {
            setLanguage(currentLanguage === 'en' ? 'es' : 'en');
        });
    }
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
    processingStatus.textContent = t('processing.uploading');
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
        showToast(t('toast.invalid_audio'), 'error');
        return false;
    }

    // Check file size (100MB max)
    const maxSize = 100 * 1024 * 1024;
    if (file.size > maxSize) {
        showToast(t('toast.file_too_large'), 'error');
        return false;
    }

    return true;
}

// Transcription with Streaming
async function transcribeWithStreaming(formData, enableDiarization = false, enableMinutes = false) {
    processingStatus.textContent = t('processing.starting');
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
        processingStatus.textContent = t('processing.transcribing');

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
            processingStatus.textContent = t('processing.transcribing');
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
            showToast(t('toast.speakers_done'), 'success');
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
                console.warn(t('misc.no_audio_session'));
            }

            // Show results section now
            showSection('results');
            showToast(t('toast.transcription_done'), 'success');
            break;

        case 'minutes_ready':
            // Display meeting minutes
            displayMeetingMinutes(data.minutes);
            showToast(t('toast.minutes_done'), 'success');
            break;

        case 'minutes_error':
            // Handle minutes generation error
            minutesLoading.classList.add('hidden');
            minutesContent.classList.remove('hidden');
            minutesSummary.textContent = t('minutes.could_not_generate') + (data.error || '');
            showToast(t('toast.minutes_error'), 'error');
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
    showToast(t('toast.downloaded'), 'success');
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
        showToast(t('toast.no_transcription_data'), 'error');
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
    showToast(t('toast.srt_downloaded'), 'success');
}

// Download VTT file
function downloadVTT() {
    if (currentSegments.length === 0) {
        showToast(t('toast.no_transcription_data'), 'error');
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
    showToast(t('toast.vtt_downloaded'), 'success');
}

// Display Meeting Minutes
function displayMeetingMinutes(minutes) {
    currentMinutes = minutes;

    // Hide loading, show content
    minutesLoading.classList.add('hidden');
    minutesContent.classList.remove('hidden');

    // Executive Summary
    minutesSummary.textContent = minutes.executive_summary || t('minutes.no_summary');

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
        li.textContent = t('minutes.no_key_points');
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
        li.textContent = t('minutes.no_decisions');
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
                <div class="action-task">${item.task || t('minutes.no_task')}</div>
                <div class="action-meta">
                    <span class="action-label">${t('minutes.assignee_label')}</span> <span class="action-assignee">${item.assignee || t('minutes.unassigned')}</span>
                    <span class="action-separator">|</span>
                    <span class="action-label">${t('minutes.date_label')}</span> <span class="action-deadline">${item.deadline || t('minutes.date_tbd')}</span>
                </div>
            `;
            minutesActions.appendChild(li);
        });
    } else {
        const li = document.createElement('li');
        li.textContent = t('minutes.no_actions');
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
        span.textContent = t('minutes.no_participants');
        span.className = 'empty-item';
        minutesParticipants.appendChild(span);
    }
}

// Download Meeting Minutes as TXT
function downloadMinutes() {
    if (!currentMinutes) {
        showToast(t('toast.no_minutes'), 'error');
        return;
    }

    let content = t('minutes.download_header') + '\n';
    content += '='.repeat(50) + '\n\n';

    content += t('minutes.download_executive_summary') + '\n';
    content += '-'.repeat(30) + '\n';
    content += (currentMinutes.executive_summary || t('minutes.no_summary')) + '\n\n';

    content += t('minutes.download_key_points') + '\n';
    content += '-'.repeat(30) + '\n';
    if (currentMinutes.key_discussion_points && currentMinutes.key_discussion_points.length > 0) {
        currentMinutes.key_discussion_points.forEach((point, i) => {
            content += `${i + 1}. ${point}\n`;
        });
    } else {
        content += t('minutes.no_key_points') + '\n';
    }
    content += '\n';

    content += t('minutes.download_decisions') + '\n';
    content += '-'.repeat(30) + '\n';
    if (currentMinutes.decisions_made && currentMinutes.decisions_made.length > 0) {
        currentMinutes.decisions_made.forEach((decision, i) => {
            content += `${i + 1}. ${decision}\n`;
        });
    } else {
        content += t('minutes.no_decisions') + '\n';
    }
    content += '\n';

    content += t('minutes.download_actions') + '\n';
    content += '-'.repeat(30) + '\n';
    if (currentMinutes.action_items && currentMinutes.action_items.length > 0) {
        currentMinutes.action_items.forEach((item, i) => {
            content += `${i + 1}. ${item.task || t('minutes.no_task')}\n`;
            content += `   - ${t('minutes.assignee_label')} ${item.assignee || t('minutes.unassigned')}\n`;
            content += `   - ${t('minutes.date_label')} ${item.deadline || t('minutes.date_tbd')}\n\n`;
        });
    } else {
        content += t('minutes.no_actions') + '\n';
    }
    content += '\n';

    content += t('minutes.download_participants') + '\n';
    content += '-'.repeat(30) + '\n';
    if (currentMinutes.participants_mentioned && currentMinutes.participants_mentioned.length > 0) {
        content += currentMinutes.participants_mentioned.join(', ') + '\n';
    } else {
        content += t('minutes.no_participants') + '\n';
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
    showToast(t('toast.minutes_downloaded'), 'success');
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
        showToast(t('toast.ref_audio_invalid'), 'error');
        return;
    }

    // Validate file size (max 15MB)
    if (file.size > 15 * 1024 * 1024) {
        showToast(t('toast.ref_audio_too_large'), 'error');
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
        showToast(t('toast.mic_error'), 'error');
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
    modelHint.textContent = t('voice.model_hint');
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
        showToast(t('toast.no_ref_audio'), 'error');
        return;
    }

    const refText = refTextInput.value.trim();
    if (!refText) {
        showToast(t('toast.no_ref_text'), 'error');
        return;
    }

    const targetText = targetTextInput.value.trim();
    if (!targetText) {
        showToast(t('toast.no_target_text'), 'error');
        return;
    }

    if (targetText.length > 50000) {
        showToast(t('toast.target_too_long'), 'error');
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
            throw new Error(result.detail || t('toast.voice_gen_error'));
        }

        if (!result.success) {
            throw new Error(result.error || t('toast.voice_gen_error'));
        }

        // Show results
        generatedAudioSessionId = result.audio_session_id;
        generatedAudioPlayer.src = `/api/audio/${result.audio_session_id}`;
        generatedDuration.textContent = formatDuration(result.duration || 0);

        voiceCloneProcessing.classList.add('hidden');
        voiceCloneResults.classList.remove('hidden');

        showToast(t('toast.voice_generated'), 'success');

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
        showToast(t('toast.no_audio_download'), 'error');
        return;
    }

    const a = document.createElement('a');
    a.href = `/api/audio/${generatedAudioSessionId}`;
    a.download = `voice-clone-${Date.now()}.wav`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    showToast(t('toast.downloaded'), 'success');
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
        showToast(t('toast.no_image_prompt'), 'error');
        return;
    }

    if (prompt.length > 500) {
        showToast(t('toast.prompt_too_long'), 'error');
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
            throw new Error(result.detail || t('toast.image_gen_error'));
        }

        if (!result.success) {
            throw new Error(result.error || t('toast.image_gen_error'));
        }

        // Show results
        generatedImageSessionId = result.image_session_id;
        generatedImagePreview.src = `/api/image/${result.image_session_id}`;
        generatedImageDimensions.textContent = `${result.width} x ${result.height} px`;

        imageGenProcessing.classList.add('hidden');
        imageGenResults.classList.remove('hidden');

        showToast(t('toast.image_generated'), 'success');

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
        showToast(t('toast.no_image_download'), 'error');
        return;
    }

    const a = document.createElement('a');
    a.href = `/api/image/${generatedImageSessionId}`;
    a.download = `generated-image-${Date.now()}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    showToast(t('toast.downloaded'), 'success');
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
    savedVoicesList.innerHTML = `<div class="loading-voices">${t('voice.loading')}</div>`;
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
        let errorMsg = t('toast.voices_load_error');
        if (error.name === 'AbortError') {
            errorMsg = t('toast.voices_timeout');
        } else if (error.message) {
            errorMsg = `Error: ${error.message}`;
        }

        // Show error in the list area with retry button
        savedVoicesList.innerHTML = `
            <div class="error-message">
                ${errorMsg}
                <button class="btn btn-secondary retry-btn" onclick="loadSavedVoices()" style="margin-top: 0.5rem;">
                    ${t('misc.retry')}
                </button>
            </div>`;

        // Show the "create new voice" option even on error
        const msgElement = noVoicesMessage.querySelector('p');
        if (msgElement) {
            msgElement.textContent = t('toast.voices_load_failed');
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
            <button class="btn-icon delete-voice-btn" title="${t('misc.delete_voice')}" data-voice-id="${voice.id}">
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
        showToast(t('toast.select_voice'), 'error');
        return;
    }

    const targetText = savedVoiceTargetText.value.trim();
    if (!targetText) {
        showToast(t('toast.no_target_text'), 'error');
        return;
    }

    if (targetText.length > 50000) {
        showToast(t('toast.target_too_long'), 'error');
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
            throw new Error(result.detail || t('toast.synth_error'));
        }

        if (!result.success) {
            throw new Error(result.error || t('toast.synth_error'));
        }

        // Show results
        generatedAudioSessionId = result.audio_session_id;
        generatedAudioPlayer.src = `/api/audio/${result.audio_session_id}`;
        generatedDuration.textContent = formatDuration(result.duration || 0);

        voiceCloneProcessing.classList.add('hidden');
        voiceCloneResults.classList.remove('hidden');

        showToast(t('toast.audio_generated'), 'success');

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
        showToast(t('toast.voice_name_required'), 'error');
        return;
    }

    if (voiceName.length > 50) {
        showToast(t('toast.voice_name_too_long'), 'error');
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
        showToast(t('toast.no_ref_audio'), 'error');
        return;
    }

    // Get reference text
    const refText = refTextInput.value.trim();
    if (!refText) {
        showToast(t('toast.no_ref_text'), 'error');
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
            throw new Error(result.detail || t('toast.voice_save_error'));
        }

        if (!result.success) {
            throw new Error(result.error || t('toast.voice_save_error'));
        }

        showToast(t('toast.voice_saved'), 'success');

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
    if (!confirm(t('confirm.delete_voice'))) {
        return;
    }

    try {
        const response = await fetch(`/api/voices/${voiceId}`, {
            method: 'DELETE',
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || t('toast.voice_delete_error'));
        }

        // Clear selection if deleted voice was selected
        if (selectedVoiceId === voiceId) {
            selectedVoiceId = null;
        }

        showToast(t('toast.voice_deleted'), 'success');
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
    const locale = currentLanguage === 'es' ? 'es-ES' : 'en-US';
    return date.toLocaleDateString(locale, {
        day: '2-digit',
        month: '2-digit',
        year: 'numeric',
    });
}

// Initialize app
init();
