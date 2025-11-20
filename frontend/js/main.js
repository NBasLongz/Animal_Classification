// Navigation functionality
const navItems = document.querySelectorAll('.nav-item');
const pages = document.querySelectorAll('.page');
const menuToggle = document.getElementById('menuToggle');
const sidebar = document.getElementById('sidebar');

navItems.forEach(item => {
    item.addEventListener('click', () => {
        const targetPage = item.getAttribute('data-page');
        
        // Update active nav item
        navItems.forEach(nav => nav.classList.remove('active'));
        item.classList.add('active');
        
        // Update active page
        pages.forEach(page => page.classList.remove('active'));
        document.getElementById(targetPage).classList.add('active');
        
        // Close sidebar on mobile after navigation
        if (window.innerWidth <= 768) {
            sidebar.classList.remove('active');
        }
    });
});

// Mobile menu toggle
menuToggle.addEventListener('click', () => {
    sidebar.classList.toggle('active');
});

// Tab functionality
const tabs = document.querySelectorAll('.tab');
const tabContents = document.querySelectorAll('.tab-content');

tabs.forEach(tab => {
    tab.addEventListener('click', () => {
        const targetTab = tab.getAttribute('data-tab');
        
        // Update active tab
        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        
        // Update active tab content
        tabContents.forEach(content => content.classList.remove('active'));
        document.getElementById(targetTab).classList.add('active');
        
        // Reset visualizations when switching tabs
        resetPipelineVisualizations();
    });
});

// File upload functionality
const uploadButton = document.getElementById('uploadButton');
const fileInput = document.getElementById('fileInput');
const imagePreview = document.getElementById('imagePreview');
const uploadArea = document.getElementById('uploadArea');

uploadButton.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block';
            
            // Reset visualizations to placeholder state
            resetPipelineVisualizations();
        };
        reader.readAsDataURL(file);
    }
});

// Drag and drop functionality
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.backgroundColor = 'rgba(52, 152, 219, 0.1)';
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.backgroundColor = 'rgba(52, 152, 219, 0.05)';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.backgroundColor = 'rgba(52, 152, 219, 0.05)';
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        fileInput.files = e.dataTransfer.files;
        const event = new Event('change', { bubbles: true });
        fileInput.dispatchEvent(event);
    }
});

// API Configuration
const API_URL = 'http://localhost:5000/api';

// Prediction buttons functionality
const handcraftedPredictButton = document.getElementById('handcraftedPredictButton');
const deepPredictButton = document.getElementById('deepPredictButton');

// Helper function to convert image to base64
function getImageBase64() {
    return imagePreview.src;
}

// Function to call prediction API
async function predictImage(method, modelType) {
    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: getImageBase64(),
                method: method,
                model: modelType
            })
        });

        if (!response.ok) {
            throw new Error('Prediction failed');
        }

        const result = await response.json();
        return result;
    } catch (error) {
        console.error('Error:', error);
        alert('Lỗi kết nối với server. Vui lòng đảm bảo backend đang chạy!');
        throw error;
    }
}

handcraftedPredictButton.addEventListener('click', async () => {
    // Check if image is uploaded
    if (!imagePreview.src || imagePreview.style.display === 'none') {
        alert('Vui lòng tải ảnh lên trước khi dự đoán!');
        return;
    }
    
    // Show loading state
    handcraftedPredictButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Đang xử lý...';
    handcraftedPredictButton.disabled = true;
    
    try {
        // Update visualizations first
        await updatePipelineVisualizationsFromAPI(imagePreview.src);
        
        // Get selected model
        const modelType = document.getElementById('handcraftedModel').value;
        
        // Call API
        const result = await predictImage('handcrafted', modelType);
        
        // Update UI
        const handcraftedConfidence = document.getElementById('handcraftedConfidence');
        const handcraftedResultLabel = document.getElementById('handcraftedResultLabel');
        const handcraftedProcessingTime = document.getElementById('handcraftedProcessingTime');
        
        // Reset width first
        handcraftedConfidence.style.width = '0%';
        
        // Then animate to final values
        setTimeout(() => {
            const confidenceValue = (result.confidence * 100).toFixed(2);
            handcraftedConfidence.style.width = confidenceValue + '%';
            handcraftedResultLabel.textContent = result.prediction.toUpperCase();
            handcraftedProcessingTime.textContent = `Thời gian xử lý: ${result.processing_time}s`;
        }, 100);
        
    } catch (error) {
        // Error already handled in predictImage function
    } finally {
        // Reset button
        handcraftedPredictButton.innerHTML = '<i class="fas fa-magic"></i> Thực hiện Dự đoán';
        handcraftedPredictButton.disabled = false;
    }
});

deepPredictButton.addEventListener('click', async () => {
    // Check if image is uploaded
    if (!imagePreview.src || imagePreview.style.display === 'none') {
        alert('Vui lòng tải ảnh lên trước khi dự đoán!');
        return;
    }
    
    // Show loading state
    deepPredictButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Đang xử lý...';
    deepPredictButton.disabled = true;
    
    try {
        // Update visualizations first
        await updatePipelineVisualizationsFromAPI(imagePreview.src);
        
        // Get selected model
        const modelType = document.getElementById('deepModel').value;
        
        // Call API
        const result = await predictImage('deep', modelType);
        
        // Update UI
        const deepConfidence = document.getElementById('deepConfidence');
        const deepResultLabel = document.getElementById('deepResultLabel');
        const deepProcessingTime = document.getElementById('deepProcessingTime');
        
        // Reset width first
        deepConfidence.style.width = '0%';
        
        // Then animate to final values
        setTimeout(() => {
            const confidenceValue = (result.confidence * 100).toFixed(2);
            deepConfidence.style.width = confidenceValue + '%';
            deepResultLabel.textContent = result.prediction.toUpperCase();
            deepProcessingTime.textContent = `Thời gian xử lý: ${result.processing_time}s`;
        }, 100);
        
    } catch (error) {
        // Error already handled in predictImage function
    } finally {
        // Reset button
        deepPredictButton.innerHTML = '<i class="fas fa-magic"></i> Thực hiện Dự đoán';
        deepPredictButton.disabled = false;
    }
});

// Draw color histogram on canvas
function drawColorHistogram() {
    const canvas = document.getElementById('colorHistogram');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Draw axes
    ctx.strokeStyle = '#ddd';
    ctx.beginPath();
    ctx.moveTo(20, height - 20);
    ctx.lineTo(width - 10, height - 20);
    ctx.moveTo(20, 10);
    ctx.lineTo(20, height - 20);
    ctx.stroke();
    
    // Generate random histogram data for demo
    const rData = Array.from({length: 20}, () => Math.random() * 80 + 20);
    const gData = Array.from({length: 20}, () => Math.random() * 80 + 20);
    const bData = Array.from({length: 20}, () => Math.random() * 80 + 20);
    
    // Draw histogram lines
    const drawLine = (data, color) => {
        ctx.strokeStyle = color;
        ctx.beginPath();
        const stepX = (width - 40) / data.length;
        
        data.forEach((value, index) => {
            const x = 20 + index * stepX;
            const y = height - 20 - (value / 100) * (height - 40);
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        
        ctx.stroke();
    };
    
    drawLine(rData, 'rgba(255, 0, 0, 0.7)');
    drawLine(gData, 'rgba(0, 255, 0, 0.7)');
    drawLine(bData, 'rgba(0, 0, 255, 0.7)');
}

// Initialize histogram when page loads
window.addEventListener('load', () => {
    drawColorHistogram();
    loadTrainingResults();
});

// Redraw histogram when switching to handcrafted tab
document.querySelector('[data-tab="handcrafted"]').addEventListener('click', () => {
    setTimeout(drawColorHistogram, 100);
});

// Load training results from API
async function loadTrainingResults() {
    try {
        const response = await fetch(`${API_URL}/training-results`);
        if (!response.ok) {
            console.error('Failed to load training results');
            return;
        }
        
        const data = await response.json();
        
        // Update chart bars
        updateChart(data.models);
        
        // Update table
        updateResultsTable(data.models);
        
        // Draw confusion matrices from data
        if (data.labels && data.models && data.models[0].confusion_matrix) {
            updateConfusionMatrix(data.models, data.labels);
        }
        
    } catch (error) {
        console.error('Error loading training results:', error);
    }
}

function updateChart(models) {
    const chartContainer = document.querySelector('.chart-container');
    if (!chartContainer) return;
    
    chartContainer.innerHTML = '';
    
    models.forEach(model => {
        const barDiv = document.createElement('div');
        barDiv.className = 'bar';
        barDiv.style.height = model.accuracy + '%';
        
        const valueDiv = document.createElement('div');
        valueDiv.className = 'bar-value';
        valueDiv.textContent = model.accuracy + '%';
        
        const labelDiv = document.createElement('div');
        labelDiv.className = 'bar-label';
        labelDiv.textContent = model.display_name;
        
        barDiv.appendChild(valueDiv);
        barDiv.appendChild(labelDiv);
        chartContainer.appendChild(barDiv);
    });
}

function updateResultsTable(models) {
    const tbody = document.querySelector('.results-table tbody');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    models.forEach(model => {
        const row = document.createElement('tr');
        
        row.innerHTML = `
            <td>${model.display_name}</td>
            <td>${model.accuracy}%</td>
            <td>${model.training_time}</td>
            <td>${model.note}</td>
        `;
        
        tbody.appendChild(row);
    });
}

function drawConfusionMatrix(model, containerDiv) {
    const cm = model.confusion_matrix;
    const labels = window.trainingLabels || ['cat', 'dog', 'wild'];
    const n = labels.length;
    
    // Create table
    const table = document.createElement('table');
    table.style.cssText = 'border-collapse: collapse; margin: 10px auto; font-size: 16px;';
    
    // Header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    headerRow.appendChild(document.createElement('th')); // Empty corner
    
    const predictedHeader = document.createElement('th');
    predictedHeader.colSpan = n;
    predictedHeader.textContent = 'Predicted';
    predictedHeader.style.cssText = 'text-align: center; padding: 10px; font-weight: bold; font-size: 14px;';
    headerRow.appendChild(predictedHeader);
    thead.appendChild(headerRow);
    
    const labelRow = document.createElement('tr');
    labelRow.appendChild(document.createElement('th')); // Empty corner
    labels.forEach(label => {
        const th = document.createElement('th');
        th.textContent = label;
        th.style.cssText = 'padding: 10px; text-align: center; font-weight: bold; font-size: 14px;';
        labelRow.appendChild(th);
    });
    thead.appendChild(labelRow);
    table.appendChild(thead);
    
    // Body
    const tbody = document.createElement('tbody');
    const maxVal = Math.max(...cm.flat());
    
    cm.forEach((row, i) => {
        const tr = document.createElement('tr');
        
        // True label
        if (i === 0) {
            const trueLabelCell = document.createElement('th');
            trueLabelCell.textContent = 'True';
            trueLabelCell.rowSpan = n;
            trueLabelCell.style.cssText = 'writing-mode: vertical-rl; transform: rotate(180deg); padding: 10px; font-weight: bold; text-align: center; font-size: 14px;';
            tr.appendChild(trueLabelCell);
        }
        
        row.forEach((val, j) => {
            const td = document.createElement('td');
            td.textContent = val;
            
            // Color intensity based on value
            const intensity = val / maxVal;
            const blue = Math.floor(255 - intensity * 120);
            const bgColor = i === j 
                ? `rgb(${blue}, ${blue}, 255)` // Diagonal: darker blue
                : `rgb(${255 - intensity * 50}, ${255 - intensity * 50}, 255)`; // Off-diagonal: lighter
            
            td.style.cssText = `
                padding: 20px;
                text-align: center;
                background-color: ${bgColor};
                border: 1px solid #ddd;
                font-weight: bold;
                min-width: 80px;
                font-size: 18px;
            `;
            tr.appendChild(td);
        });
        
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    
    // Add to container
    containerDiv.appendChild(table);
}

function updateConfusionMatrix(models, labels) {
    const container = document.getElementById('confusionMatrixContainer');
    if (!container) return;
    
    container.innerHTML = '';
    window.trainingLabels = labels;
    
    // Create grid for 2x2 confusion matrices
    const grid = document.createElement('div');
    grid.style.cssText = 'display: grid; grid-template-columns: 1fr 1fr; gap: 40px; max-width: 100%; margin: 0 auto;';
    
    models.forEach(model => {
        const modelDiv = document.createElement('div');
        modelDiv.style.cssText = 'text-align: center;';
        
        const title = document.createElement('h4');
        title.textContent = `${model.display_name}`;
        title.style.cssText = 'margin-bottom: 8px; color: #2c3e50; font-size: 18px;';
        modelDiv.appendChild(title);
        
        const accuracy = document.createElement('p');
        accuracy.textContent = `Accuracy: ${model.accuracy}%`;
        accuracy.style.cssText = 'margin: 8px 0 15px 0; color: #7f8c8d; font-size: 16px; font-weight: 500;';
        modelDiv.appendChild(accuracy);
        
        drawConfusionMatrix(model, modelDiv);
        grid.appendChild(modelDiv);
    });
    
    container.appendChild(grid);
}

// Pipeline Visualization Functions - Call API for accurate processing
async function updatePipelineVisualizationsFromAPI(imageSrc) {
    try {
        const response = await fetch(`${API_URL}/visualize-pipeline`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: imageSrc
            })
        });

        if (!response.ok) {
            throw new Error('Visualization failed');
        }

        const visualizations = await response.json();
        
        // Update handcrafted pipeline
        updateHandcraftedPipelineFromAPI(visualizations);
        
        // Update deep learning pipeline
        updateDeepLearningPipelineFromAPI(visualizations);
        
    } catch (error) {
        console.error('Error generating visualizations:', error);
        alert('Lỗi kết nối với server. Vui lòng đảm bảo backend đang chạy!');
    }
}

function updateHandcraftedPipelineFromAPI(visualizations) {
    // 1. Original image (128x128)
    const originalImg = document.getElementById('handPipelineOriginal');
    const originalPlaceholder = document.getElementById('handPipelineOriginalPlaceholder');
    originalImg.src = visualizations.original_128;
    originalImg.style.display = 'block';
    originalPlaceholder.style.display = 'none';
    
    // 2. Grayscale
    const grayImg = document.getElementById('handPipelineGray');
    const grayCtx = grayImg.getContext('2d');
    const grayImage = new Image();
    grayImage.onload = function() {
        grayImg.width = grayImage.width;
        grayImg.height = grayImage.height;
        grayCtx.drawImage(grayImage, 0, 0);
    };
    grayImage.src = visualizations.grayscale;
    
    // 3. HOG Visualization
    const hogImg = document.getElementById('handPipelineHOG');
    const hogCtx = hogImg.getContext('2d');
    const hogImage = new Image();
    hogImage.onload = function() {
        hogImg.width = hogImage.width;
        hogImg.height = hogImage.height;
        hogCtx.drawImage(hogImage, 0, 0);
    };
    hogImage.src = visualizations.hog;
    
    // 4. Color Histogram
    const histImg = document.getElementById('colorHistogram');
    const histCtx = histImg.getContext('2d');
    const histImage = new Image();
    histImage.onload = function() {
        histImg.width = histImage.width;
        histImg.height = histImage.height;
        histCtx.drawImage(histImage, 0, 0);
    };
    histImage.src = visualizations.color_histogram;
}

function updateDeepLearningPipelineFromAPI(visualizations) {
    // 1. Original image
    const originalImg = document.getElementById('deepPipelineOriginal');
    const originalPlaceholder = document.getElementById('deepPipelineOriginalPlaceholder');
    originalImg.src = imagePreview.src; // Use the uploaded image
    originalImg.style.display = 'block';
    originalPlaceholder.style.display = 'none';
    
    // 2. ResNet preprocessed input
    const processedImg = document.getElementById('deepPipelineProcessed');
    const processedCtx = processedImg.getContext('2d');
    const resnetImage = new Image();
    resnetImage.onload = function() {
        processedImg.width = resnetImage.width;
        processedImg.height = resnetImage.height;
        processedCtx.drawImage(resnetImage, 0, 0);
    };
    resnetImage.src = visualizations.resnet_input;
}

// Reset visualizations to placeholder state
function resetPipelineVisualizations() {
    // Reset handcrafted pipeline
    const handOriginalImg = document.getElementById('handPipelineOriginal');
    const handOriginalPlaceholder = document.getElementById('handPipelineOriginalPlaceholder');
    handOriginalImg.style.display = 'none';
    handOriginalPlaceholder.style.display = 'flex';
    
    // Clear canvases
    const grayCanvas = document.getElementById('handPipelineGray');
    const hogCanvas = document.getElementById('handPipelineHOG');
    const histCanvas = document.getElementById('colorHistogram');
    
    if (grayCanvas) {
        const ctx = grayCanvas.getContext('2d');
        ctx.clearRect(0, 0, grayCanvas.width, grayCanvas.height);
    }
    if (hogCanvas) {
        const ctx = hogCanvas.getContext('2d');
        ctx.clearRect(0, 0, hogCanvas.width, hogCanvas.height);
    }
    if (histCanvas) {
        const ctx = histCanvas.getContext('2d');
        ctx.clearRect(0, 0, histCanvas.width, histCanvas.height);
    }
    
    // Reset deep learning pipeline
    const deepOriginalImg = document.getElementById('deepPipelineOriginal');
    const deepOriginalPlaceholder = document.getElementById('deepPipelineOriginalPlaceholder');
    deepOriginalImg.style.display = 'none';
    deepOriginalPlaceholder.style.display = 'flex';
    
    const processedCanvas = document.getElementById('deepPipelineProcessed');
    if (processedCanvas) {
        const ctx = processedCanvas.getContext('2d');
        ctx.clearRect(0, 0, processedCanvas.width, processedCanvas.height);
    }
}

// Old client-side functions (keeping for fallback, but not used)
function updatePipelineVisualizations(imageSrc) {
    // Create an image object to load the uploaded image
    const img = new Image();
    img.onload = function() {
        // Update handcrafted pipeline
        updateHandcraftedPipeline(img, imageSrc);
        
        // Update deep learning pipeline
        updateDeepLearningPipeline(img, imageSrc);
    };
    img.src = imageSrc;
}

function updateHandcraftedPipeline(img, imageSrc) {
    // 1. Show original image (128x128)
    const originalImg = document.getElementById('handPipelineOriginal');
    const originalPlaceholder = document.getElementById('handPipelineOriginalPlaceholder');
    originalImg.src = imageSrc;
    originalImg.style.display = 'block';
    originalPlaceholder.style.display = 'none';
    
    // 2. Draw grayscale version
    const grayCanvas = document.getElementById('handPipelineGray');
    const grayCtx = grayCanvas.getContext('2d');
    grayCanvas.width = 150;
    grayCanvas.height = 120;
    
    // Draw and convert to grayscale
    grayCtx.drawImage(img, 0, 0, 150, 120);
    const imageData = grayCtx.getImageData(0, 0, 150, 120);
    const data = imageData.data;
    
    for (let i = 0; i < data.length; i += 4) {
        const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
        data[i] = data[i + 1] = data[i + 2] = gray;
    }
    
    grayCtx.putImageData(imageData, 0, 0);
    
    // 3. HOG Visualization (simplified representation)
    const hogCanvas = document.getElementById('handPipelineHOG');
    const hogCtx = hogCanvas.getContext('2d');
    hogCanvas.width = 150;
    hogCanvas.height = 120;
    
    // Draw grayscale image first
    hogCtx.drawImage(grayCanvas, 0, 0);
    
    // Add HOG-like visualization overlay (gradient directions)
    hogCtx.strokeStyle = 'rgba(255, 165, 0, 0.6)';
    hogCtx.lineWidth = 1;
    
    const cellSize = 10;
    for (let y = cellSize; y < 120; y += cellSize) {
        for (let x = cellSize; x < 150; x += cellSize) {
            // Get gradient direction (simplified)
            const angle = Math.random() * Math.PI;
            const len = 5;
            const dx = Math.cos(angle) * len;
            const dy = Math.sin(angle) * len;
            
            hogCtx.beginPath();
            hogCtx.moveTo(x - dx, y - dy);
            hogCtx.lineTo(x + dx, y + dy);
            hogCtx.stroke();
        }
    }
    
    // 4. Color Histogram
    const histCanvas = document.getElementById('colorHistogram');
    const histCtx = histCanvas.getContext('2d');
    histCanvas.width = 150;
    histCanvas.height = 120;
    
    // Clear canvas
    histCtx.fillStyle = 'white';
    histCtx.fillRect(0, 0, 150, 120);
    
    // Calculate color histogram
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    tempCanvas.width = img.width;
    tempCanvas.height = img.height;
    tempCtx.drawImage(img, 0, 0);
    
    const pixelData = tempCtx.getImageData(0, 0, img.width, img.height).data;
    const histR = new Array(16).fill(0);
    const histG = new Array(16).fill(0);
    const histB = new Array(16).fill(0);
    
    for (let i = 0; i < pixelData.length; i += 4) {
        const binR = Math.floor(pixelData[i] / 16);
        const binG = Math.floor(pixelData[i + 1] / 16);
        const binB = Math.floor(pixelData[i + 2] / 16);
        histR[binR]++;
        histG[binG]++;
        histB[binB]++;
    }
    
    // Normalize
    const maxCount = Math.max(...histR, ...histG, ...histB);
    const barWidth = 150 / 16;
    const maxHeight = 100;
    
    // Draw histograms
    for (let i = 0; i < 16; i++) {
        const x = i * barWidth;
        const heightR = (histR[i] / maxCount) * maxHeight;
        const heightG = (histG[i] / maxCount) * maxHeight;
        const heightB = (histB[i] / maxCount) * maxHeight;
        
        // Red
        histCtx.fillStyle = 'rgba(255, 0, 0, 0.5)';
        histCtx.fillRect(x, 120 - heightR, barWidth / 3, heightR);
        
        // Green
        histCtx.fillStyle = 'rgba(0, 255, 0, 0.5)';
        histCtx.fillRect(x + barWidth / 3, 120 - heightG, barWidth / 3, heightG);
        
        // Blue
        histCtx.fillStyle = 'rgba(0, 0, 255, 0.5)';
        histCtx.fillRect(x + 2 * barWidth / 3, 120 - heightB, barWidth / 3, heightB);
    }
}

function updateDeepLearningPipeline(img, imageSrc) {
    // 1. Show original image
    const originalImg = document.getElementById('deepPipelineOriginal');
    const originalPlaceholder = document.getElementById('deepPipelineOriginalPlaceholder');
    originalImg.src = imageSrc;
    originalImg.style.display = 'block';
    originalPlaceholder.style.display = 'none';
    
    // 2. Show ResNet preprocessing (Resize 256 -> Center Crop 224 -> Normalize)
    const processedCanvas = document.getElementById('deepPipelineProcessed');
    const ctx = processedCanvas.getContext('2d');
    processedCanvas.width = 300;
    processedCanvas.height = 200;
    
    // Calculate scaling to resize to 256 (maintaining aspect ratio)
    const scale = 256 / Math.min(img.width, img.height);
    const scaledWidth = img.width * scale;
    const scaledHeight = img.height * scale;
    
    // Center crop coordinates for 224x224
    const cropX = (scaledWidth - 224) / 2;
    const cropY = (scaledHeight - 224) / 2;
    
    // Draw the center-cropped region
    // First, draw to a temp canvas at scaled size
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    tempCanvas.width = scaledWidth;
    tempCanvas.height = scaledHeight;
    tempCtx.drawImage(img, 0, 0, scaledWidth, scaledHeight);
    
    // Then crop and draw to display canvas
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, 300, 200);
    
    // Draw cropped region (scaled to fit canvas)
    const displayScale = Math.min(300 / 224, 200 / 224);
    const displayWidth = 224 * displayScale;
    const displayHeight = 224 * displayScale;
    const offsetX = (300 - displayWidth) / 2;
    const offsetY = (200 - displayHeight) / 2;
    
    ctx.drawImage(tempCanvas, 
        cropX, cropY, 224, 224,
        offsetX, offsetY, displayWidth, displayHeight
    );
    
    // Add border to show the crop area
    ctx.strokeStyle = '#3498db';
    ctx.lineWidth = 2;
    ctx.strokeRect(offsetX, offsetY, displayWidth, displayHeight);
    
    // Add text overlay
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(0, 0, 300, 25);
    ctx.fillStyle = 'white';
    ctx.font = '12px Arial';
    ctx.fillText('224x224 Center Crop + Normalized', 10, 17);
}
