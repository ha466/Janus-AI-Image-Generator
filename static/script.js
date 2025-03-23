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
    
    const loadingSection = document.getElementById('loadingSection');
    const loadingText = document.getElementById('loadingText');
    const statusDetail = document.getElementById('statusDetail');
    const progressBar = document.getElementById('progressBar');
    
    const resultSection = document.getElementById('resultSection');
    const imageGrid = document.getElementById('imageGrid');
    const downloadAllBtn = document.getElementById('downloadAllBtn');
    const newGenerationBtn = document.getElementById('newGenerationBtn');
    
    const errorSection = document.getElementById('errorSection');
    const errorMessage = document.getElementById('errorMessage');
    const tryAgainBtn = document.getElementById('tryAgainBtn');
    
    const imageModal = document.getElementById('imageModal');
    const modalImage = document.getElementById('modalImage');
    const closeModal = document.querySelector('.close');
    const downloadBtn = document.getElementById('downloadBtn');
    
    // Initialize logo image
    const logoImg = document.querySelector('.logo');
    if (logoImg && !logoImg.getAttribute('src')) {
        logoImg.src = 'https://raw.githubusercontent.com/deepseek-ai/Janus/main/images/logo.png';
    }
    
    // Create favicon if it doesn't exist
    if (!document.querySelector('link[rel="icon"]')) {
        const favicon = document.createElement('link');
        favicon.rel = 'icon';
        favicon.href = 'https://raw.githubusercontent.com/deepseek-ai/Janus/main/images/logo.png';
        document.head.appendChild(favicon);
    }
    
    // Update slider values
    cfgScaleSlider.addEventListener('input', function() {
        cfgValueSpan.textContent = this.value;
    });
    
    temperatureSlider.addEventListener('input', function() {
        tempValueSpan.textContent = this.value;
    });
    
    // Resolution change handler
    resolutionSelect.addEventListener('change', function() {
        const resolution = this.value;
        
        // Adjust num_images options based on resolution
        const options = numImagesSelect.options;
        
        if (resolution === '4K') {
            // Disable options > 4 for 4K
            for (let i = 0; i < options.length; i++) {
                if (parseInt(options[i].value) > 4) {
                    options[i].disabled = true;
                    if (options[i].selected) {
                        numImagesSelect.value = '4';
                    }
                }
            }
        } else if (resolution === '2K') {
            // Disable options > 8 for 2K
            for (let i = 0; i < options.length; i++) {
                if (parseInt(options[i].value) > 8) {
                    options[i].disabled = true;
                    if (options[i].selected) {
                        numImagesSelect.value = '8';
                    }
                } else {
                    options[i].disabled = false;
                }
            }
        } else {
            // Enable all options for 1K
            for (let i = 0; i < options.length; i++) {
                options[i].disabled = false;
            }
        }
    });
    
    // Initialize resolution options on page load
    resolutionSelect.dispatchEvent(new Event('change'));
    
    // Form submission handler
    generationForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const prompt = promptInput.value.trim();
        if (!prompt) {
            showError('Please enter an image description');
            return;
        }
        
        // Hide previous sections
        resultSection.classList.add('hidden');
        errorSection.classList.add('hidden');
        
        // Show loading section
        loadingSection.classList.remove('hidden');
        progressBar.style.width = '0%';
        statusDetail.textContent = 'Starting generation...';
        
        // Disable generation button
        generateBtn.disabled = true;
        
        // Collect form data
        const requestData = {
            prompt: prompt,
            num_images: parseInt(numImagesSelect.value),
            resolution: resolutionSelect.value,
            cfg_weight: parseFloat(cfgScaleSlider.value),
            temperature: parseFloat(temperatureSlider.value)
        };
        
        // Send generation request
        fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Start polling for status
                const generationId = data.id;
                pollGenerationStatus(generationId);
            } else {
                throw new Error(data.error || 'Failed to start generation');
            }
        })
        .catch(error => {
            showError(error.message || 'An error occurred while starting generation');
            loadingSection.classList.add('hidden');
            generateBtn.disabled = false;
        });
    });
    
    // Function to poll generation status
    function pollGenerationStatus(generationId) {
        const statusUrl = `/api/status/${generationId}`;
        let pollInterval;
        
        const checkStatus = () => {
            fetch(statusUrl)
                .then(response => response.json())
                .then(statusData => {
                    if (statusData.status === 'error') {
                        clearInterval(pollInterval);
                        throw new Error(statusData.message || 'Generation failed');
                    } else if (statusData.status === 'complete') {
                        clearInterval(pollInterval);
                        displayGeneratedImages(statusData.images);
                        loadingSection.classList.add('hidden');
                        generateBtn.disabled = false;
                    } else if (statusData.status === 'processing') {
                        // Update progress
                        if (statusData.progress) {
                            const progressPercent = Math.round(statusData.progress * 100);
                            progressBar.style.width = `${progressPercent}%`;
                            statusDetail.textContent = `Processing image tokens... ${progressPercent}%`;
                        }
                        
                        if (statusData.completed && statusData.total) {
                            loadingText.textContent = `Generated ${statusData.completed} of ${statusData.total} images`;
                        }
                    }
                })
                .catch(error => {
                    clearInterval(pollInterval);
                    showError(error.message || 'Failed to check generation status');
                    loadingSection.classList.add('hidden');
                    generateBtn.disabled = false;
                });
        };
        
        // Initial check
        checkStatus();
        
        // Set up polling every 2 seconds
        pollInterval = setInterval(checkStatus, 2000);
    }
    
    // Function to display generated images
    function displayGeneratedImages(images) {
        imageGrid.innerHTML = '';
        
        images.forEach((imageData, index) => {
            const imgContainer = document.createElement('div');
            imgContainer.className = 'image-container';
            
            const img = document.createElement('img');
            img.src = imageData.thumbnail;
            img.alt = `Generated image ${index + 1}`;
            img.dataset.fullSrc = imageData.full;
            
            const imageInfo = document.createElement('div');
            imageInfo.className = 'image-info';
            
            const resolutionLabel = document.createElement('span');
            const resolution = resolutionSelect.options[resolutionSelect.selectedIndex].text;
            resolutionLabel.textContent = resolution;
            
            imageInfo.appendChild(resolutionLabel);
            
            imgContainer.appendChild(img);
            imgContainer.appendChild(imageInfo);
            imageGrid.appendChild(imgContainer);
            
            // Add click event to show modal
            imgContainer.addEventListener('click', function() {
                showImageModal(imageData.full);
            });
        });
        
        resultSection.classList.remove('hidden');
    }
    
    // Image modal functions
    function showImageModal(imageSrc) {
        modalImage.src = imageSrc;
        imageModal.style.display = 'block';
        setTimeout(() => {
            imageModal.classList.add('show');
        }, 10);
        
        // Update download button
        downloadBtn.onclick = function() {
            downloadImage(imageSrc);
        };
    }
    
    closeModal.addEventListener('click', function() {
        imageModal.classList.remove('show');
        setTimeout(() => {
            imageModal.style.display = 'none';
        }, 300);
    });
    
    // Close modal when clicking outside the image
    imageModal.addEventListener('click', function(e) {
        if (e.target === imageModal) {
            closeModal.click();
        }
    });
    
    // Download functions
    function downloadImage(imageSrc) {
        const a = document.createElement('a');
        a.href = imageSrc;
        a.download = `janus-image-${Date.now()}.jpg`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }
    
    downloadAllBtn.addEventListener('click', function() {
        const images = document.querySelectorAll('#imageGrid .image-container img');
        images.forEach((img, index) => {
            setTimeout(() => {
                downloadImage(img.dataset.fullSrc);
            }, index * 500); // Stagger downloads to avoid browser limitations
        });
    });
    
    // New generation button
    newGenerationBtn.addEventListener('click', function() {
        resultSection.classList.add('hidden');
        promptInput.focus();
    });
    
    // Try again button
    tryAgainBtn.addEventListener('click', function() {
        errorSection.classList.add('hidden');
        promptInput.focus();
    });
    
    // Error handling
    function showError(message) {
        errorMessage.textContent = message;
        errorSection.classList.remove('hidden');
    }
    
    // Handle keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // ESC to close modal
        if (e.key === 'Escape' && imageModal.style.display === 'block') {
            closeModal.click();
        }
    });
    
    // Add example prompts
    const examplePrompts = [
        "A stunning princess from Kabul in red, white traditional clothing, blue eyes, brown hair",
        "An astronaut riding a horse on Mars, photorealistic",
        "A futuristic cityscape with flying cars and neon lights",
        "A serene Japanese garden with cherry blossoms and a traditional pagoda",
        "A detailed fantasy dragon perched on a mountain peak at sunset"
    ];
    
    // Create example prompts button if it doesn't exist
    if (!document.getElementById('examplePromptsBtn')) {
        const exampleBtn = document.createElement('button');
        exampleBtn.id = 'examplePromptsBtn';
        exampleBtn.className = 'action-button';
        exampleBtn.innerHTML = '<i class="fas fa-lightbulb"></i> Example Prompts';
        exampleBtn.style.marginBottom = '1rem';
        
        const promptLabel = document.querySelector('label[for="promptInput"]');
        promptLabel.parentNode.insertBefore(exampleBtn, promptLabel.nextSibling);
        
        exampleBtn.addEventListener('click', function() {
            const randomPrompt = examplePrompts[Math.floor(Math.random() * examplePrompts.length)];
            promptInput.value = randomPrompt;
        });
    }
}); 