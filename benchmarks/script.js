// TODO: Make this configurable into a separate file
const defaultReferenceCommit = 'c1bdc24';
const benchmarkPrimaryBase = 'https://raw.githubusercontent.com/linkedin/Liger-Kernel/refs/heads/gh-pages/benchmarks';
const benchmarkFallbackBase = 'https://cdn.jsdelivr.net/gh/linkedin/Liger-Kernel@gh-pages/benchmarks';

// Kernel categories configuration
const kernelCategories = {
  "pretraining": [
    "cross_entropy",
    "fused_linear_cross_entropy",
    "softmax",
    "sparsemax",
    "rope",
    "geglu",
    "swiglu",
    "layer_norm",
    "rms_norm",
    "group_norm",
    "multi_token_attention",
    "sparse_multi_token_attention",
    "fused_neighborhood_attention"
  ],
  "distillation": [
    "distill_jsd_loss",
    "distill_cosine_loss", 
    "jsd",
    "kl_div",
    "tvd",
    "fused_linear_jsd",
    "dyt_beta=True",
    "dyt_beta=False"
  ],
  "post_training": [
    "dpo_loss",
    "kto_loss",
    "fused_linear_cpo_loss",
    "fused_linear_simpo_loss"
  ]
};

let allDataByCommit = {};
let kernelMeta = {};
let configData = {}; // Store configuration data for tooltips - format: {commit: {kernelKey: config}}
let goldenCommits = {};

async function fetchWithFallback(path, responseType = 'text') {
  const urls = [
    `${benchmarkPrimaryBase}/${path}`,
    `${benchmarkFallbackBase}/${path}`,
  ];
  let lastErr;
  for (const url of urls) {
    try {
      const res = await fetch(url);
      if (res.ok) {
        if (responseType === 'json') return await res.json();
        return await res.text();
      }
      lastErr = new Error(`HTTP ${res.status} for ${url}`);
    } catch (e) {
      lastErr = e;
    }
  }
  throw lastErr || new Error('Failed to fetch with fallback');
}

async function loadGoldenCommits() {
  try {
    const data = await fetchWithFallback('golden_commits.json', 'json');
    return data.golden_commits || {};
  } catch (e) {
    console.warn('Could not load golden commits from JSON');
  }
  return {};
}

function getGoldenCommit(kernelName) {
  // Extract base kernel name by removing operation mode and metric suffixes
  let baseKernelName = kernelName;
  
  const suffixes = ['_forward_speed', '_backward_speed', '_full_speed', '_full_memory', '_forward_memory', '_backward_memory'];
  for (const suffix of suffixes) {
    if (baseKernelName.endsWith(suffix)) {
      baseKernelName = baseKernelName.slice(0, -suffix.length);
      break;
    }
  }
  
  // Check for exact base kernel match
  if (goldenCommits[baseKernelName]) {
    return goldenCommits[baseKernelName];
  }
  
  // Check for partial matches - find the longest matching key
  let bestMatch = '';
  let bestCommit = defaultReferenceCommit;
  
  for (const [key, commit] of Object.entries(goldenCommits)) {
    if (kernelName.includes(key) && key.length > bestMatch.length) {
      bestMatch = key;
      bestCommit = commit;
    }
  }
  
  return bestCommit;
}

function getKernelCategory(kernelName) {
  // Extract base kernel name by removing operation mode and metric suffixes
  let baseKernelName = kernelName;
  
  const suffixes = ['_forward_speed', '_backward_speed', '_full_speed', '_full_memory', '_forward_memory', '_backward_memory'];
  for (const suffix of suffixes) {
    if (baseKernelName.endsWith(suffix)) {
      baseKernelName = baseKernelName.slice(0, -suffix.length);
      break;
    }
  }
  
  // Find which category this kernel belongs to
  for (const [category, kernels] of Object.entries(kernelCategories)) {
    for (const kernelPattern of kernels) {
      if (baseKernelName === kernelPattern || baseKernelName.includes(kernelPattern)) {
        return category;
      }
    }
  }
  
  return 'other'; // Default category for unmatched kernels
}

function toggleTheme() {
  const body = document.body;
  const themeButton = document.querySelector('.theme-toggle');
  const isDark = body.getAttribute('data-theme') === 'dark';
  body.setAttribute('data-theme', isDark ? 'light' : 'dark');
  themeButton.textContent = isDark ? '☀️' : '🌙';
}

function goToDetailedView() {
    const baseUrl = window.location.href.endsWith('/') ? window.location.href : window.location.href + '/';
    window.location.href = baseUrl + 'detailed';
}

async function fetchCommits() {
  const text = await fetchWithFallback('commits.txt', 'text');
  const all = text.trim().split('\n');
  const commits = all.filter(line => !line.includes('-'));
  const versions = all.filter(line => line.includes('-'));
  return { commits, versions };
}

async function fetchCSV(commit) {
  const text = await fetchWithFallback(`${commit}/benchmark.csv`, 'text');
  const rows = text.trim().split('\n');
  const headers = rows[0].split(',');
  const data = {};

  for (let i = 1; i < rows.length; i++) {
    const rowText = rows[i];
    const cols = [];
    let currentCol = '';
    let inQuotes = false;
    let quoteChar = '';
    
    // Parse CSV row properly handling quoted fields
    for (let j = 0; j < rowText.length; j++) {
      const char = rowText[j];
      
      if ((char === '"' || char === "'") && !inQuotes) {
        inQuotes = true;
        quoteChar = char;
      } else if (char === quoteChar && inQuotes) {
        inQuotes = false;
        quoteChar = '';
      } else if (char === ',' && !inQuotes) {
        cols.push(currentCol.trim());
        currentCol = '';
      } else {
        currentCol += char;
      }
    }
    cols.push(currentCol.trim());
    
    const row = Object.fromEntries(headers.map((h, idx) => [h.trim(), cols[idx]?.trim()]));
    if (row.kernel_provider !== 'liger') continue;

    const kernelKey = `${row.kernel_name}_${row.kernel_operation_mode}_${row.metric_name}`;

    if (!data[kernelKey]) {
      data[kernelKey] = parseFloat(row.y_value_50);
      if (!kernelMeta[kernelKey]) {
        kernelMeta[kernelKey] = {
          x_label: row.x_label,
          x_value: row.x_value
        };
      }
      // Store configuration data for tooltips
      if (!configData[commit]) {
        configData[commit] = {};
      }
      if (!configData[commit][kernelKey]) {
        configData[commit][kernelKey] = {
          extra_benchmark_config_str: row.extra_benchmark_config_str || 'N/A',
          gpu_name: row.gpu_name || 'N/A',
          liger_version: row.liger_version || 'N/A'
        };
      }
    }
  }
  return data;
}

async function loadData() {
  document.getElementById('speedTable').innerHTML = 'Loading...';
  document.getElementById('memoryTable').innerHTML = 'Loading...';

  goldenCommits = await loadGoldenCommits();

  const count = document.getElementById('commitCount').value;
  const viewType = document.getElementById('viewType').value;
  const { commits, versions } = await fetchCommits();

  let selectedList = viewType === 'versions' ? versions : commits;
  const selectedCommits = count === 'all' ? selectedList : selectedList.slice(-parseInt(count));

  // Collect all unique commits that need to be loaded
  const commitsToLoad = new Set(selectedCommits);
  
  // Add reference commits only if they're not already in the selected commits
  Object.values(goldenCommits).forEach(commit => {
    if (!selectedCommits.includes(commit)) {
      commitsToLoad.add(commit);
    }
  });
  
  // Also add the default reference commit if not already included
  if (!selectedCommits.includes(defaultReferenceCommit)) {
    commitsToLoad.add(defaultReferenceCommit);
  }

  allDataByCommit = {};
  kernelMeta = {};
  configData = {};

  await Promise.all([...commitsToLoad].map(async (commit) => {
    const data = await fetchCSV(commit);
    if (data) allDataByCommit[commit] = data;
  }));

  await renderTables();
}

function compareMetric(current, reference) {
  const THRESHOLD = 0.10;
  if (current == null || reference == null) return '';
  const delta = (current - reference) / reference;
  if (delta > THRESHOLD) return 'red';
  if (delta < 0) return 'green';
  return '';
}

function formatConfigForTooltip(config) {
  try {
    const parsed = JSON.parse(config.extra_benchmark_config_str);
    const configStr = Object.entries(parsed)
      .map(([key, value]) => `${key}: ${value}`)
      .join('\n');
    return `Extra Config:\n${configStr}\n\nGPU: ${config.gpu_name}\nLiger Version: ${config.liger_version}`;
  } catch (e) {
    return `Extra Config: ${config.extra_benchmark_config_str}\n\nGPU: ${config.gpu_name}\nLiger Version: ${config.liger_version}`;
  }
}

async function renderTables() {
  const metricType = document.getElementById('metricType').value;
  const searchQuery = document.getElementById('kernelSearch').value.toLowerCase();
  const categoryFilter = document.getElementById('categoryFilter').value;
  const speedTable = document.getElementById('speedTable');
  const memoryTable = document.getElementById('memoryTable');
  const viewType = document.getElementById('viewType').value;

  speedTable.innerHTML = '';
  memoryTable.innerHTML = '';

  const count = document.getElementById('commitCount').value;
  const { commits: allCommitsList, versions: allVersionsList } = await fetchCommits();
  const selectedList = viewType === 'versions' ? allVersionsList : allCommitsList;
  const selectedCommits = count === 'all' ? selectedList : selectedList.slice(-parseInt(count));
  
  const allLoadedCommits = Object.keys(allDataByCommit);
  const commits = selectedCommits.filter(commit => allLoadedCommits.includes(commit)).reverse();

  function createTable(table, filterMetric) {
    let header = `<tr><th>Kernel</th><th>X</th>`;
    if (viewType === 'commits') {
      header += `<th>Golden Reference</th>`;
    }
    for (const commit of commits) {
      header += `<th>${commit}</th>`;
    }
    header += '</tr>';
    table.innerHTML += header;

    const allKeys = new Set();
    for (const commit of commits) {
      for (const k in allDataByCommit[commit]) {
        if (k.includes(`_${metricType}_${filterMetric}`)) allKeys.add(k);
      }
    }

    [...allKeys].forEach((kernelKey) => {
      if (searchQuery && !kernelKey.toLowerCase().includes(searchQuery)) return;
      
      if (categoryFilter !== 'all') {
        const kernelCategory = getKernelCategory(kernelKey);
        if (kernelCategory !== categoryFilter) return;
      }
      
      const kernelCategory = getKernelCategory(kernelKey);
      let row = `<tr class="category-${kernelCategory}">`;
      row += `<td>${kernelKey}</td><td>${(kernelMeta[kernelKey]?.x_label || '?')}=${(kernelMeta[kernelKey]?.x_value || '?')}</td>`;
      if (viewType === 'commits') {
        const goldenCommit = getGoldenCommit(kernelKey);
        let refVal = allDataByCommit[goldenCommit]?.[kernelKey];
        if (refVal == null) {
          const versionedKey = Object.keys(allDataByCommit).find(k => k.startsWith(goldenCommit + '-'));
          if (versionedKey && allDataByCommit[versionedKey]?.[kernelKey] != null) {
            refVal = allDataByCommit[versionedKey][kernelKey];
          }
        }
        const refConfig = configData[goldenCommit]?.[kernelKey] || { extra_benchmark_config_str: 'N/A', gpu_name: 'N/A', liger_version: 'N/A' };
        const refTooltipText = formatConfigForTooltip(refConfig) + `\n\nCategory: ${kernelCategory}\nGolden Commit: ${goldenCommit}`;
        row += `<td title="${goldenCommit}" data-tooltip="${refTooltipText}" style="cursor: pointer;">${refVal != null ? refVal.toFixed(2) : 'N/A'}</td>`;
      }
      for (const commit of commits) {
        const val = allDataByCommit[commit]?.[kernelKey];
        let cellClass = '';
        if (viewType === 'commits') {
          const goldenCommit = getGoldenCommit(kernelKey);
          const refVal = allDataByCommit[goldenCommit]?.[kernelKey];
          const cls = compareMetric(val, refVal);
          const isGolden = commit === goldenCommit;
          cellClass = isGolden ? 'golden' : cls;
        }
        
        const config = configData[commit]?.[kernelKey] || { extra_benchmark_config_str: 'N/A', gpu_name: 'N/A', liger_version: 'N/A' };
        const tooltipText = formatConfigForTooltip(config) + `\n\nCategory: ${kernelCategory}\nCommit: ${commit}`;
        row += `<td class="${cellClass}" data-tooltip="${tooltipText}" style="cursor: pointer;">${val != null ? val.toFixed(2) : 'N/A'}</td>`;
      }
      row += '</tr>';
      table.innerHTML += row;
    });
  }

  createTable(memoryTable, 'memory');
  createTable(speedTable, 'speed');
  
  addTooltipListeners();
}

let tooltipEl;

function addTooltipListeners() {
  if (!tooltipEl) {
    tooltipEl = document.createElement('div');
    tooltipEl.id = 'custom-tooltip';
    tooltipEl.className = 'custom-tooltip';
    document.body.appendChild(tooltipEl);
  }

  const tooltipCells = document.querySelectorAll('td[data-tooltip]');
  
  tooltipCells.forEach(cell => {
    cell.addEventListener('mouseenter', showTooltip);
    cell.addEventListener('mousemove', moveTooltip);
    cell.addEventListener('mouseleave', hideTooltip);
  });
}

function showTooltip(event) {
  tooltipEl.textContent = event.currentTarget.getAttribute('data-tooltip');
  tooltipEl.style.display = 'block';
  moveTooltip(event);
}

function moveTooltip(event) {
    if (!tooltipEl || tooltipEl.style.display === 'none') return;
    
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;

    let left = event.clientX + 15;
    let top = event.clientY + 15;

    const tooltipWidth = tooltipEl.offsetWidth;
    const tooltipHeight = tooltipEl.offsetHeight;

    if (left + tooltipWidth > viewportWidth) {
        left = event.clientX - tooltipWidth - 15;
    }

    if (top + tooltipHeight > viewportHeight) {
        top = event.clientY - tooltipHeight - 15;
    }

    if (left < 0) {
        left = 5;
    }

    if (top < 0) {
        top = 5;
    }

    tooltipEl.style.left = `${left}px`;
    tooltipEl.style.top = `${top}px`;
}

function hideTooltip() {
  if (tooltipEl) {
    tooltipEl.style.display = 'none';
  }
}

window.onload = loadData;