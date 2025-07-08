// TODO: Make this configurable into a separate file
const defaultReferenceCommit = 'c1bdc24';
const benchmarkBase = 'https://raw.githubusercontent.com/linkedin/Liger-Kernel/refs/heads/gh-pages/benchmarks';

let allCommits = [];
let allDataByCommit = {};
let kernelMeta = {};
let configData = {}; // Store configuration data for tooltips
let goldenCommits = {};

async function loadGoldenCommits() {
  try {
    const res = await fetch('https://raw.githubusercontent.com/linkedin/Liger-Kernel/refs/heads/gh-pages/benchmark/golden_commits.json');
    if (res.ok) {
      const data = await res.json();
      return data.golden_commits || {};
    }
  } catch (e) {
    console.warn('Could not load golden commits from JSON, using fallback');
  }
  return {};
}

function getGoldenCommit(kernelName) {
  // Extract base kernel name by removing operation mode and metric suffixes
  let baseKernelName = kernelName;
  
  // Remove common suffixes
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
  const res = await fetch(`${benchmarkBase}/commits.txt`);
  const text = await res.text();
  return text.trim().split('\n');
}

async function fetchCSV(commit) {
  const res = await fetch(`${benchmarkBase}/${commit}/benchmark.csv`);
  if (!res.ok) return null;
  const text = await res.text();
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
    cols.push(currentCol.trim()); // Add the last column
    
    const row = Object.fromEntries(headers.map((h, idx) => [h.trim(), cols[idx]?.trim()]));
    if (row.kernel_provider !== 'liger') continue;

    const baseKey = `${row.kernel_name}_${row.kernel_operation_mode}_${row.metric_name}`;
    const kernelKey = `${baseKey}`;

    if (!data[kernelKey]) {
      data[kernelKey] = parseFloat(row.y_value_50);
      if (!kernelMeta[kernelKey]) {
        kernelMeta[kernelKey] = {
          x_label: row.x_label,
          x_value: row.x_value
        };
      }
      // Store configuration data for tooltips
      if (!configData[kernelKey]) {
        // TODO: Compare for same GPU type only.
        configData[kernelKey] = {
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
  allCommits = await fetchCommits();
  const selectedCommits = count === 'all' ? allCommits : allCommits.slice(-parseInt(count));

  allDataByCommit = {};
  kernelMeta = {};
  configData = {}; // Reset config data

  await Promise.all(selectedCommits.map(async (commit) => {
    const data = await fetchCSV(commit);
    if (data) allDataByCommit[commit] = data;
  }));

  renderTables();
}

function compareMetric(current, reference) {
  // Tried running benchmarks on the same commit 4 times, and there was a 10% variance in the results.
  // So we're using a 10% threshold to highlight the difference.
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

function renderTables() {
  const metricType = document.getElementById('metricType').value;
  const searchQuery = document.getElementById('kernelSearch').value.toLowerCase();
  const speedTable = document.getElementById('speedTable');
  const memoryTable = document.getElementById('memoryTable');

  speedTable.innerHTML = '';
  memoryTable.innerHTML = '';

  const commits = Object.keys(allDataByCommit);

  function createTable(table, filterMetric) {
    let header = `<tr><th>Kernel</th><th>X</th><th>Golden Reference</th>`
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
      
      // Get kernel-specific golden commit
      const goldenCommit = getGoldenCommit(kernelKey);
      const refVal = allDataByCommit[goldenCommit]?.[kernelKey];
      const meta = kernelMeta[kernelKey] || { x_label: '?', x_value: '?' };
      const config = configData[kernelKey] || { extra_benchmark_config_str: 'N/A', gpu_name: 'N/A', liger_version: 'N/A' };
      
      const tooltipText = formatConfigForTooltip(config) + `\n\nGolden Commit: ${goldenCommit}`;

      let row = `<tr data-tooltip="${tooltipText}" style="cursor: pointer;">`;
      row += `<td>${kernelKey}</td><td>${meta.x_label}=${meta.x_value}</td><td title="${goldenCommit}">${refVal != null ? refVal.toFixed(2) : 'N/A'}</td>`;
      for (const commit of commits) {
        const val = allDataByCommit[commit]?.[kernelKey];
        const cls = compareMetric(val, refVal);
        const isGolden = commit === goldenCommit;
        const cellClass = isGolden ? 'golden' : cls;
        row += `<td class="${cellClass}" ${isGolden ? 'title="Golden Reference"' : ''}>${val != null ? val.toFixed(2) : 'N/A'}</td>`;
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

  const tooltipRows = document.querySelectorAll('tr[data-tooltip]');
  
  tooltipRows.forEach(row => {
    row.addEventListener('mouseenter', showTooltip);
    row.addEventListener('mousemove', moveTooltip);
    row.addEventListener('mouseleave', hideTooltip);
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

    // Use clientX and clientY which are relative to the viewport for fixed positioning
    let left = event.clientX + 15;
    let top = event.clientY + 15;

    const tooltipWidth = tooltipEl.offsetWidth;
    const tooltipHeight = tooltipEl.offsetHeight;

    // Prevent tooltip from going off the right edge of the viewport
    if (left + tooltipWidth > viewportWidth) {
        left = event.clientX - tooltipWidth - 15;
    }

    // Prevent tooltip from going off the bottom edge of the viewport
    if (top + tooltipHeight > viewportHeight) {
        top = event.clientY - tooltipHeight - 15;
    }

    // Prevent tooltip from going off the left edge
    if (left < 0) {
        left = 5;
    }

    // Prevent tooltip from going off the top edge
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