document.addEventListener('DOMContentLoaded', () => {
    const radioBtns = document.querySelectorAll('input[name="method"]');
    const templateDropzone = document.getElementById('dropzone-template');
    
    // Toggle template dropzone based on method
    radioBtns.forEach(radio => {
        radio.addEventListener('change', (e) => {
            if (e.target.value === 'classical') {
                templateDropzone.classList.remove('hidden');
            } else {
                templateDropzone.classList.add('hidden');
            }
        });
    });

    // Setup drag and drop functionality
    setupDropzone('dropzone-test', 'file-test', 'preview-test');
    setupDropzone('dropzone-template', 'file-template', 'preview-template');

    // Analyze button
    const btnAnalyze = document.getElementById('btn-analyze');
    btnAnalyze.addEventListener('click', runAnalysis);
});

function setupDropzone(dropzoneId, inputId, previewId) {
    const dropzone = document.getElementById(dropzoneId);
    const input = document.getElementById(inputId);
    const preview = document.getElementById(previewId);

    // Click to select
    dropzone.addEventListener('click', () => {
        input.click();
    });

    input.addEventListener('change', () => {
        if (input.files.length) handleFile(input.files[0], dropzone, preview);
    });

    // Drag events
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            input.files = e.dataTransfer.files; // assign files to input
            handleFile(e.dataTransfer.files[0], dropzone, preview);
        }
    });
}

function handleFile(file, dropzone, preview) {
    if (file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => {
            preview.style.backgroundImage = `url(${e.target.result})`;
            dropzone.classList.add('has-file');
        };
        reader.readAsDataURL(file);
    }
}

async function runAnalysis() {
    const btn = document.getElementById('btn-analyze');
    const method = document.querySelector('input[name="method"]:checked').value;
    const fileTest = document.getElementById('file-test').files[0];
    const fileTemplate = document.getElementById('file-template').files[0];

    if (!fileTest) {
        alert("TEST_IMAGE required.");
        return;
    }
    
    if (method === 'classical' && !fileTemplate) {
        alert("TEMPLATE_IMAGE required for classical method.");
        return;
    }

    // Set loading state
    btn.disabled = true;
    btn.classList.add('scanning');
    document.querySelector('.btn-text').textContent = "SCANNING...";

    const formData = new FormData();
    formData.append('method', method);
    formData.append('test_img', fileTest);
    if (method === 'classical') {
        formData.append('template_img', fileTemplate);
    }

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || "Unknown error occurred.");
        }

        displayResults(data);
    } catch (error) {
        alert("SCAN_ERROR: " + error.message);
        console.error(error);
    } finally {
        // Reset state
        btn.disabled = false;
        btn.classList.remove('scanning');
        document.querySelector('.btn-text').textContent = "INITIATE_SCAN()";
    }
}

function displayResults(data) {
    const outputPanel = document.getElementById('output-panel');
    const resultsContainer = document.getElementById('results-container');
    const logContainer = document.getElementById('execution-log');
    const methodSpan = document.getElementById('output-method');

    // Show panel
    outputPanel.classList.remove('hidden');
    methodSpan.textContent = `[${data.method.toUpperCase()}]`;
    
    // Clear previous
    resultsContainer.innerHTML = '';
    
    // Determine which outputs we got
    // Add cache buster to images so they reload if filenames are the same
    const ts = new Date().getTime();
    
    if (data.method === 'dl') {
        const imgPath = data.outputs.result + `?t=${ts}`;
        resultsContainer.innerHTML = `
            <div class="result-item">
                <div class="result-label">RESULT_IMAGE</div>
                <img class="result-image" src="${imgPath}" alt="Result">
            </div>
        `;
    } else {
        const resultPath = data.outputs.result + `?t=${ts}`;
        const maskPath = data.outputs.mask + `?t=${ts}`;
        const alignedPath = data.outputs.aligned + `?t=${ts}`;
        
        resultsContainer.innerHTML = `
            <div class="result-item">
                <div class="result-label">RESULT_IMAGE (BOUNDING BOXES)</div>
                <img class="result-image" src="${resultPath}" alt="Result">
            </div>
            <div style="display: flex; gap: 20px;">
                <div class="result-item" style="flex:1;">
                    <div class="result-label">MASK_IMAGE</div>
                    <img class="result-image" src="${maskPath}" alt="Mask">
                </div>
                <div class="result-item" style="flex:1;">
                    <div class="result-label">ALIGNED_IMAGE</div>
                    <img class="result-image" src="${alignedPath}" alt="Aligned">
                </div>
            </div>
        `;
    }

    logContainer.textContent = data.logs;
    
    // Scroll down to results
    outputPanel.scrollIntoView({ behavior: 'smooth' });
}
