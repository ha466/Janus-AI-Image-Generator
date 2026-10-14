document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const generationForm = document.getElementById('generationForm');
    const promptInput = document.getElementById('promptInput');
    const numImagesSelect = document.getElementById('numImages');
    const resolutionSelect = document.getElementById('resolution');
    const cfgScaleSlider = document.getElementById('cfgScale');
    const cfgValueSpan = document.getElementById('cfgValue');
    const temperatureSlider = document.getElementById('temperature');
    const tempValueSpan = document.getElementById('tempValue');
    const generateBtn = document.getElementById('generateBtn');
    const randomPromptBtn = document.getElementById('randomPromptBtn');
    const promptChipsContainer = document.getElementById('promptChips');

    // Status & Badges
    const systemStatusBadge = document.getElementById('systemStatusBadge');
    const statusText = document.getElementById('statusText');
    const themeToggleBtn = document.getElementById('themeToggleBtn');

    // Loading State Elements
    const loadingSection = document.getElementById('loadingSection');
    const loadingTitle = document.getElementById('loadingTitle');
    const statusDetail = document.getElementById('statusDetail');
    const progressBar = document.getElementById('progressBar');
    const progressPercentage = document.getElementById('progressPercentage');
    const cancelGenBtn = document.getElementById('cancelGenBtn');

    // Results & History Elements
    const resultSection = document.getElementById('resultSection');
    const resultPromptText = document.getElementById('resultPromptText');
    const imageGrid = document.getElementById('imageGrid');
    const downloadAllBtn = document.getElementById('downloadAllBtn');
    const newGenerationBtn = document.getElementById('newGenerationBtn');
    const historyList = document.getElementById('historyList');
    const clearHistoryBtn = document.getElementById('clearHistoryBtn');

    // Error Elements
    const errorSection = document.getElementById('errorSection');
    const errorMessage = document.getElementById('errorMessage');
    const tryAgainBtn = document.getElementById('tryAgainBtn');

    // Modal Elements
    const imageModal = document.getElementById('imageModal');
    const closeModalBtn = document.getElementById('closeModalBtn');
    const modalImage = document.getElementById('modalImage');
    const modalPromptText = document.getElementById('modalPromptText');
    const modalResTag = document.getElementById('modalResTag');
    const modalCfgTag = document.getElementById('modalCfgTag');
    const modalTempTag = document.getElementById('modalTempTag');
    const modalTimeTag = document.getElementById('modalTimeTag');
    const downloadBtn = document.getElementById('downloadBtn');
    const reusePromptBtn = document.getElementById('reusePromptBtn');

    // State Variables
    let currentGenerationId = null;
    let activePollInterval = null;
    let examplePromptsList = [];

    // Initialize System Status Check
    checkSystemStatus();
    loadExamplePrompts();
    loadHistory();

    // Theme Toggle
    themeToggleBtn.addEventListener('click', () => {
        const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', newTheme);
        themeToggleBtn.innerHTML = newTheme === 'dark' ? '<i class="fas fa-moon"></i>' : '<i class="fas fa-sun"></i>';
    });

    // Slider Listeners
    cfgScaleSlider.addEventListener('input', function() {
        cfgValueSpan.textContent = parseFloat(this.value).toFixed(1);
    });

    temperatureSlider.addEventListener('input', function() {
        tempValueSpan.textContent = parseFloat(this.value).toFixed(1);
    });

    // Resolution change listener to adjust max quantity options
    resolutionSelect.addEventListener('change', function() {
        const res = this.value;
        const options = numImagesSelect.options;

        for (let i = 0; i < options.length; i++) {
            const val = parseInt(options[i].value);
            if (res === '4K' && val > 4) {
                options[i].disabled = true;
                if (options[i].selected) numImagesSelect.value = '4';
            } else if (res === '2K' && val > 8) {
                options[i].disabled = true;
                if (options[i].selected) numImagesSelect.value = '8';
            } else {
                options[i].disabled = false;
            }
        }
    });

    // System Status API
    function checkSystemStatus() {
        fetch('/api/system')
            .then(res => res.json())
            .then(data => {
                if (data.gpu_available) {
                    const gpuName = data.gpu_info.name || 'GPU Ready';
                    statusText.textContent = `GPU: ${gpuName}`;
                    systemStatusBadge.style.color = 'var(--success)';
                } else {
                    statusText.textContent = 'CPU Mode (Active)';
                    systemStatusBadge.style.color = '#f59e0b';
                }
            })
            .catch(() => {
                statusText.textContent = 'Server Online';
            });
    }

    // Example Prompts API
    function loadExamplePrompts() {
        fetch('/api/example_prompts')
            .then(res => res.json())
            .then(prompts => {
                examplePromptsList = prompts;
                renderPromptChips(prompts);
            })
            .catch(() => {
                examplePromptsList = [
                    "Cyberpunk samurai in neon rainy street",
                    "Serene Japanese garden with cherry blossoms",
                    "Majestic dragon on misty mountain peak"
                ];
                renderPromptChips(examplePromptsList);
            });
    }

    function renderPromptChips(prompts) {
        promptChipsContainer.innerHTML = '';
        prompts.forEach(p => {
            const chip = document.createElement('span');
            chip.className = 'chip';
            chip.textContent = p;
            chip.title = p;
            chip.addEventListener('click', () => {
                promptInput.value = p;
                promptInput.focus();
            });
            promptChipsContainer.appendChild(chip);
        });
    }

    // Random Prompt Generator
    randomPromptBtn.addEventListener('click', () => {
        if (examplePromptsList.length > 0) {
            const random = examplePromptsList[Math.floor(Math.random() * examplePromptsList.length)];
            promptInput.value = random;
            promptInput.focus();
        }
    });

    // Form Submission
    generationForm.addEventListener('submit', function(e) {
        e.preventDefault();

        const prompt = promptInput.value.trim();
        if (!prompt) return;

        // Hide previous panels
        resultSection.classList.add('hidden');
        errorSection.classList.add('hidden');

        // Show loading state
        loadingSection.classList.remove('hidden');
        progressBar.style.width = '0%';
        progressPercentage.textContent = '0%';
        statusDetail.textContent = 'Initializing generation model...';
        generateBtn.disabled = true;

        const requestData = {
            prompt: prompt,
            num_images: parseInt(numImagesSelect.value),
            resolution: resolutionSelect.value,
            cfg_weight: parseFloat(cfgScaleSlider.value),
            temperature: parseFloat(temperatureSlider.value),
            confirmed: true
        };

        fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                currentGenerationId = data.id;
                pollGenerationStatus(data.id);
            } else {
                throw new Error(data.error || 'Failed to start generation');
            }
        })
        .catch(err => {
            showError(err.message);
            loadingSection.classList.add('hidden');
            generateBtn.disabled = false;
        });
    });

    // Cancel Active Generation
    cancelGenBtn.addEventListener('click', () => {
        if (!currentGenerationId) return;
        fetch(`/api/cancel/${currentGenerationId}`, { method: 'POST' })
            .then(res => res.json())
            .then(() => {
                if (activePollInterval) clearInterval(activePollInterval);
                loadingSection.classList.add('hidden');
                generateBtn.disabled = false;
                statusDetail.textContent = 'Generation cancelled.';
            });
    });

    // Poll Generation Status
    function pollGenerationStatus(id) {
        if (activePollInterval) clearInterval(activePollInterval);

        activePollInterval = setInterval(() => {
            fetch(`/api/status/${id}`)
                .then(res => res.json())
                .then(statusData => {
                    if (statusData.status === 'complete') {
                        clearInterval(activePollInterval);
                        loadingSection.classList.add('hidden');
                        generateBtn.disabled = false;
                        displayResults(statusData);
                        loadHistory();
                    } else if (statusData.status === 'error') {
                        clearInterval(activePollInterval);
                        loadingSection.classList.add('hidden');
                        generateBtn.disabled = false;
                        showError(statusData.message || 'Generation failed');
                        loadHistory();
                    } else if (statusData.status === 'processing') {
                        const progress = statusData.progress || 0;
                        const pct = Math.round(progress * 100);
                        progressBar.style.width = `${pct}%`;
                        progressPercentage.textContent = `${pct}%`;

                        if (statusData.completed !== undefined && statusData.total) {
                            statusDetail.textContent = `Completed ${statusData.completed} / ${statusData.total} image tokens`;
                        } else {
                            statusDetail.textContent = `Processing image tokens... (${pct}%)`;
                        }
                    }
                })
                .catch(err => {
                    clearInterval(activePollInterval);
                    loadingSection.classList.add('hidden');
                    generateBtn.disabled = false;
                    showError('Network error checking status: ' + err.message);
                });
        }, 1500);
    }

    // Display Generated Images
    function displayResults(data) {
        imageGrid.innerHTML = '';
        resultPromptText.textContent = `"${data.prompt}"`;

        const images = data.images || [];
        images.forEach((imgObj, idx) => {
            const card = document.createElement('div');
            card.className = 'image-card';

            const img = document.createElement('img');
            img.src = imgObj.thumbnail;
            img.alt = `Generated Image ${idx + 1}`;
            img.loading = 'lazy';

            const overlay = document.createElement('div');
            overlay.className = 'image-overlay';
            overlay.innerHTML = `
                <div class="overlay-actions">
                    <span class="badge">${data.settings?.resolution || '1K'}</span>
                    <button class="icon-btn" title="View Fullscreen"><i class="fas fa-expand"></i></button>
                </div>
            `;

            card.appendChild(img);
            card.appendChild(overlay);

            card.addEventListener('click', () => {
                openModal(imgObj.full, data);
            });

            imageGrid.appendChild(card);
        });

        resultSection.classList.remove('hidden');
    }

    // Load History
    function loadHistory() {
        fetch('/api/history')
            .then(res => res.json())
            .then(history => {
                renderHistory(history);
            })
            .catch(() => {});
    }

    function renderHistory(history) {
        historyList.innerHTML = '';
        const completedHistory = history.filter(h => h.status === 'complete' && h.images && h.images.length > 0);

        if (completedHistory.length === 0) {
            historyList.innerHTML = `
                <div class="empty-history">
                    <i class="fas fa-palette"></i>
                    <p>No recent generations yet. Create your first artwork above!</p>
                </div>
            `;
            return;
        }

        completedHistory.forEach(item => {
            const row = document.createElement('div');
            row.className = 'history-item';

            const timeStr = item.generation_time ? `${item.generation_time}s` : 'Done';
            const res = item.settings?.resolution || '1K';

            row.innerHTML = `
                <div class="history-info">
                    <div class="history-prompt">${escapeHtml(item.prompt)}</div>
                    <div class="history-meta">
                        <span><i class="fas fa-images"></i> ${item.images.length} imgs</span>
                        <span><i class="fas fa-expand"></i> ${res}</span>
                        <span><i class="fas fa-stopwatch"></i> ${timeStr}</span>
                    </div>
                </div>
                <button class="secondary-btn btn-sm"><i class="fas fa-eye"></i></button>
            `;

            row.addEventListener('click', () => {
                displayResults(item);
                resultSection.scrollIntoView({ behavior: 'smooth' });
            });

            historyList.appendChild(row);
        });
    }

    // Clear History
    clearHistoryBtn.addEventListener('click', () => {
        fetch('/api/clear_history', { method: 'POST' })
            .then(() => loadHistory());
    });

    // Lightbox Modal
    function openModal(imageSrc, data) {
        modalImage.src = imageSrc;
        modalPromptText.textContent = data.prompt || 'Generated Image';
        modalResTag.textContent = data.settings?.resolution || '1K';
        modalCfgTag.textContent = data.settings?.cfg_weight || '7.0';
        modalTempTag.textContent = data.settings?.temperature || '0.8';
        modalTimeTag.textContent = data.generation_time ? `${data.generation_time}s` : '-';

        downloadBtn.onclick = () => downloadSingleImage(imageSrc);
        reusePromptBtn.onclick = () => {
            promptInput.value = data.prompt;
            closeModal();
            promptInput.focus();
        };

        imageModal.classList.add('active');
    }

    function closeModal() {
        imageModal.classList.remove('active');
    }

    closeModalBtn.addEventListener('click', closeModal);
    imageModal.addEventListener('click', (e) => {
        if (e.target === imageModal) closeModal();
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && imageModal.classList.contains('active')) {
            closeModal();
        }
    });

    // Download helpers
    function downloadSingleImage(src) {
        const a = document.createElement('a');
        a.href = src;
        a.download = `janus-ai-${Date.now()}.jpg`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }

    downloadAllBtn.addEventListener('click', () => {
        const imgs = imageGrid.querySelectorAll('.image-card img');
        imgs.forEach((img, idx) => {
            setTimeout(() => {
                downloadSingleImage(img.src);
            }, idx * 400);
        });
    });

    newGenerationBtn.addEventListener('click', () => {
        resultSection.classList.add('hidden');
        promptInput.focus();
    });

    tryAgainBtn.addEventListener('click', () => {
        errorSection.classList.add('hidden');
        promptInput.focus();
    });

    function showError(msg) {
        errorMessage.textContent = msg;
        errorSection.classList.remove('hidden');
    }

    function escapeHtml(str) {
        return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
    }
});
