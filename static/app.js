// RAG System Frontend

// --- Navigation ---
document.querySelectorAll('.nav-tab[data-panel]').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.nav-tab[data-panel]').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById(tab.dataset.panel).classList.add('active');

        if (tab.dataset.panel === 'monitoring-panel') loadMonitoring();
    });
});

// --- File Upload ---
const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');

uploadArea.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('dragover'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop', e => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    handleFiles(e.dataTransfer.files);
});
fileInput.addEventListener('change', () => handleFiles(fileInput.files));

async function handleFiles(files) {
    for (const file of files) {
        const formData = new FormData();
        formData.append('file', file);
        try {
            showToast(`Uploading ${file.name}...`);
            const res = await fetch('/api/upload', { method: 'POST', body: formData });
            const data = await res.json();
            if (res.ok) {
                showToast(`${file.name}: ${data.chunks_created} chunks indexed`, 'success');
            } else {
                showToast(`Error: ${data.detail}`, 'error');
            }
        } catch (e) {
            showToast(`Upload failed: ${e.message}`, 'error');
        }
    }
    loadDocuments();
}

// --- Documents ---
async function loadDocuments() {
    try {
        const res = await fetch('/api/stats');
        const data = await res.json();
        const container = document.getElementById('doc-list-items');

        if (!data.documents || data.documents.length === 0) {
            container.innerHTML = '<p style="font-size:13px; color:var(--text-dim); padding:8px;">No documents yet. Upload some files to get started.</p>';
            return;
        }

        container.innerHTML = data.documents.map(doc => `
            <div class="doc-item">
                <span class="name" title="${doc.name}">${doc.name}</span>
                <span class="size">${formatSize(doc.size)}</span>
                <button class="delete-btn" onclick="deleteDoc('${doc.name}')" title="Delete">&#10005;</button>
            </div>
        `).join('');
    } catch (e) {
        document.getElementById('doc-list-items').innerHTML = '<p style="color:var(--danger); font-size:13px; padding:8px;">Failed to load documents</p>';
    }
}

async function deleteDoc(filename) {
    if (!confirm(`Delete ${filename}?`)) return;
    try {
        await fetch(`/api/documents/${encodeURIComponent(filename)}`, { method: 'DELETE' });
        showToast(`Deleted ${filename}`, 'success');
        loadDocuments();
    } catch (e) {
        showToast('Delete failed', 'error');
    }
}

async function ingestAll() {
    showToast('Re-indexing all documents...');
    try {
        const res = await fetch('/api/ingest', { method: 'POST' });
        const data = await res.json();
        showToast(`Indexed ${data.stats.total_chunks} chunks from ${data.stats.total_docs} documents`, 'success');
        loadDocuments();
    } catch (e) {
        showToast('Ingestion failed: ' + e.message, 'error');
    }
}

// --- Query ---
let queryCount = 0;

async function submitQuery() {
    const input = document.getElementById('query-input');
    const query = input.value.trim();
    if (!query) return;

    const btn = document.getElementById('submit-btn');
    btn.disabled = true;
    btn.textContent = 'Searching...';

    const container = document.getElementById('answer-container');
    container.innerHTML = '<div class="loading"><div class="spinner"></div> Retrieving and generating answer...</div>';

    try {
        const res = await fetch('/api/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                top_k: parseInt(document.getElementById('top-k').value) || 5,
                use_reranking: document.getElementById('use-reranking').checked,
            }),
        });
        const data = await res.json();
        renderAnswer(query, data);
    } catch (e) {
        container.innerHTML = `<div class="answer-card"><div class="answer-body" style="color:var(--danger)">Error: ${e.message}</div></div>`;
    }

    btn.disabled = false;
    btn.textContent = 'Search';
}

function renderAnswer(query, data) {
    const container = document.getElementById('answer-container');
    const idx = queryCount++;
    const statusClass = data.status || 'success';
    const statusLabel = { success: 'Answer Generated', error: 'Error', blocked: 'Blocked by Guardrails' }[statusClass] || statusClass;

    let html = `
        <div class="answer-card">
            <div class="answer-header">
                <div class="answer-status">
                    <span class="status-dot ${statusClass}"></span>
                    <span>${statusLabel}</span>
                </div>
                <div class="answer-meta">
                    <span>Latency: ${data.latency_ms || '?'}ms</span>
                    <span>Confidence: ${(data.confidence * 100).toFixed(1)}%</span>
                    <span>Chunks: ${data.chunks_used || 0}</span>
                </div>
            </div>
            <div class="answer-body">${formatAnswer(data.answer || 'No answer available.')}</div>
            ${data.citations && data.citations.length ? `<div style="margin-top:12px; font-size:12px; color:var(--text-dim);">Citations: ${data.citations.join(', ')}</div>` : ''}
            ${data.query_rewritten && data.query_rewritten !== query ? `<div style="margin-top:8px; font-size:12px; color:var(--text-dim);">Query rewritten: "${data.query_rewritten}"</div>` : ''}
            <div class="feedback-buttons">
                <button class="feedback-btn" onclick="sendFeedback(${idx}, 'thumbs_up', this)">&#128077; Helpful</button>
                <button class="feedback-btn" onclick="sendFeedback(${idx}, 'thumbs_down', this)">&#128078; Not Helpful</button>
            </div>
        </div>
    `;

    if (data.sources && data.sources.length) {
        html += '<h3 style="font-size:14px; margin: 16px 0 8px; color:var(--text-dim);">Retrieved Sources</h3><div class="sources-list">';
        data.sources.forEach((src, i) => {
            html += `
                <div class="source-card">
                    <div class="source-header">
                        <span>[${i+1}] ${src.source} (Section ${src.chunk_index})</span>
                        <span class="source-score">Score: ${(src.score * 100).toFixed(1)}%</span>
                    </div>
                    <div class="source-preview">${escapeHtml(src.preview)}</div>
                </div>
            `;
        });
        html += '</div>';
    }

    container.innerHTML = html;
}

function formatAnswer(text) {
    // Escape HTML, then convert markdown-like formatting
    let safe = escapeHtml(text);
    // Bold
    safe = safe.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    // Citations highlighting
    safe = safe.replace(/\[Source:\s*([^\]]+)\]/g, '<span style="color:var(--accent); font-weight:600;">[Source: $1]</span>');
    // Newlines
    safe = safe.replace(/\n/g, '<br>');
    return safe;
}

async function sendFeedback(idx, feedback, btn) {
    try {
        await fetch('/api/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query_index: idx, feedback }),
        });
        btn.parentElement.querySelectorAll('.feedback-btn').forEach(b => b.classList.remove('selected'));
        btn.classList.add('selected');
    } catch (e) { /* ignore */ }
}

// --- Monitoring ---
async function loadMonitoring() {
    try {
        const [statsRes, logsRes] = await Promise.all([
            fetch('/api/stats'),
            fetch('/api/logs?n=50'),
        ]);
        const stats = await statsRes.json();
        const logs = await logsRes.json();

        renderMetrics(stats);
        renderConfig(stats.config);
        renderLogs(logs.logs);
    } catch (e) {
        document.getElementById('metrics-container').innerHTML = `<p style="color:var(--danger)">Failed to load: ${e.message}</p>`;
    }
}

function renderMetrics(stats) {
    const m = stats.metrics;
    const idx = stats.index;

    document.getElementById('metrics-container').innerHTML = `
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">${idx.total_chunks}</div>
                <div class="metric-label">Indexed Chunks</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${m.total_queries}</div>
                <div class="metric-label">Total Queries</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${m.avg_latency_ms.toFixed(0)}ms</div>
                <div class="metric-label">Avg Latency</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${(m.avg_retrieval_score * 100).toFixed(1)}%</div>
                <div class="metric-label">Avg Retrieval Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${m.avg_chunks_used.toFixed(1)}</div>
                <div class="metric-label">Avg Chunks Used</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${m.avg_citations_per_answer.toFixed(1)}</div>
                <div class="metric-label">Avg Citations</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${(m.abstention_rate * 100).toFixed(1)}%</div>
                <div class="metric-label">Abstention Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${(m.positive_feedback_rate * 100).toFixed(0)}%</div>
                <div class="metric-label">Positive Feedback</div>
            </div>
        </div>
    `;
}

function renderConfig(cfg) {
    if (!cfg) return;
    document.getElementById('config-container').innerHTML = `
        <div class="config-grid">
            ${Object.entries(cfg).map(([k, v]) => `
                <div class="config-item">
                    <span class="key">${k.replace(/_/g, ' ')}</span>
                    <span class="value">${v}</span>
                </div>
            `).join('')}
        </div>
    `;
}

function renderLogs(logs) {
    if (!logs || logs.length === 0) {
        document.getElementById('logs-container').innerHTML = '<p style="color:var(--text-dim); font-size:13px;">No queries logged yet.</p>';
        return;
    }

    document.getElementById('logs-container').innerHTML = `
        <table class="logs-table">
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Query</th>
                    <th>Status</th>
                    <th>Score</th>
                    <th>Chunks</th>
                    <th>Latency</th>
                    <th>Feedback</th>
                </tr>
            </thead>
            <tbody>
                ${logs.map(log => `
                    <tr>
                        <td>${new Date(log.timestamp).toLocaleTimeString()}</td>
                        <td title="${escapeHtml(log.query)}">${escapeHtml(log.query.substring(0, 50))}</td>
                        <td><span class="status-dot ${log.status}" style="display:inline-block; margin-right:4px;"></span>${log.status}</td>
                        <td>${(log.top_score * 100).toFixed(1)}%</td>
                        <td>${log.num_chunks_retrieved}</td>
                        <td>${log.latency_ms.toFixed(0)}ms</td>
                        <td>${log.user_feedback || '-'}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
}

// --- Utilities ---
function formatSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(1) + ' MB';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showToast(msg, type = '') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = msg;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}

// --- Init ---
loadDocuments();
