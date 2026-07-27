document.addEventListener('DOMContentLoaded', () => {
    const radioBtns = document.querySelectorAll('input[name="method"]');
    const templateDropzone = document.getElementById('dropzone-template');
    
    // Toggle template dropzone based on method
    radioBtns.forEach(radio => {
        radio.addEventListener('change', (e) => {
            if (e.target.value === 'classical' || e.target.value === 'classical_topological') {
                templateDropzone.classList.remove('hidden');
            } else {
                templateDropzone.classList.add('hidden');
            }
        });
    });

    // Setup drag and drop functionality
    setupDropzone('dropzone-test', 'file-test', 'preview-test', 'filename-test');
    setupDropzone('dropzone-template', 'file-template', 'preview-template', 'filename-template');

    // Analyze button
    const btnAnalyze = document.getElementById('btn-analyze');
    btnAnalyze.addEventListener('click', runAnalysis);
});

function setupDropzone(dropzoneId, inputId, previewId, nameId) {
    const dropzone = document.getElementById(dropzoneId);
    const input = document.getElementById(inputId);
    const preview = document.getElementById(previewId);
    const nameEl = document.getElementById(nameId);

    // Click to select
    dropzone.addEventListener('click', () => {
        input.click();
    });

    input.addEventListener('change', () => {
        if (input.files.length) handleFile(input.files[0], dropzone, preview, nameEl);
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
            handleFile(e.dataTransfer.files[0], dropzone, preview, nameEl);
        }
    });
}

function handleFile(file, dropzone, preview, nameEl) {
    if (file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => {
            preview.style.backgroundImage = `url(${e.target.result})`;
            dropzone.classList.add('has-file');
            if (nameEl) nameEl.textContent = file.name;
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
    
    if ((method === 'classical' || method === 'classical_topological') && !fileTemplate) {
        alert("TEMPLATE_IMAGE required for classical methods.");
        return;
    }

    // Set loading state
    btn.disabled = true;
    btn.classList.add('scanning');
    document.querySelector('.btn-text').textContent = "SCANNING...";

    const formData = new FormData();
    formData.append('method', method);
    formData.append('test_img', fileTest);
    if (method === 'classical' || method === 'classical_topological') {
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
        const name = data.outputs.result.split('/').pop();
        resultsContainer.innerHTML = `
            <div class="result-item">
                <div class="result-label">RESULT_IMAGE</div>
                <img class="result-image" src="${imgPath}" alt="Result">
                <div style="text-align:center; margin-top:5px; font-family: monospace; color:#0f0;">${name}</div>
            </div>
        `;
    } else {
        const resultPath = data.outputs.result + `?t=${ts}`;
        const maskPath = data.outputs.mask + `?t=${ts}`;
        const alignedPath = data.outputs.aligned + `?t=${ts}`;
        
        const rName = data.outputs.result.split('/').pop();
        const mName = data.outputs.mask.split('/').pop();
        const aName = data.outputs.aligned.split('/').pop();
        
        resultsContainer.innerHTML = `
            <div class="result-item">
                <div class="result-label">RESULT_IMAGE (BOUNDING BOXES)</div>
                <img class="result-image" src="${resultPath}" alt="Result">
                <div style="text-align:center; margin-top:5px; font-family: monospace; color:#0f0;">${rName}</div>
            </div>
            <div style="display: flex; gap: 20px;">
                <div class="result-item" style="flex:1;">
                    <div class="result-label">MASK_IMAGE</div>
                    <img class="result-image" src="${maskPath}" alt="Mask">
                    <div style="text-align:center; margin-top:5px; font-family: monospace; color:#0f0;">${mName}</div>
                </div>
                <div class="result-item" style="flex:1;">
                    <div class="result-label">ALIGNED_IMAGE</div>
                    <img class="result-image" src="${alignedPath}" alt="Aligned">
                    <div style="text-align:center; margin-top:5px; font-family: monospace; color:#0f0;">${aName}</div>
                </div>
            </div>
        `;
    }

    logContainer.textContent = data.logs;
    
    // Scroll down to results
    outputPanel.scrollIntoView({ behavior: 'smooth' });
}
