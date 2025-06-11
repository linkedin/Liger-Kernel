let chartSpeed;
let chartMemory;
let chart2Speed;
let chart2Memory;

document.addEventListener("DOMContentLoaded", async () => {
  const commits = await fetchCommitHashes();
  // const commits = ["a96bdc1", "bc105fd", "cbe66ed"]
  populateCommitDropdown(commits, "commit");
  populateCommitDropdown(commits, "commit2");

  document.getElementById("commit").addEventListener("change", (e) => {
    loadCSV(e.target.value, 1);
  });

  document.getElementById("commit2").addEventListener("change", (e) => {
    loadCSV(e.target.value, 2);
  });

  if (commits.length > 0) {
    loadCSV(commits[0], 1);
    loadCSV(commits[0], 2);
  }
});

async function fetchCommitHashes() {
  try {
    const response = await fetch(
      "https://raw.githubusercontent.com/linkedin/Liger-Kernel/refs/heads/gh-pages/benchmarks/commits.txt"
    );
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const text = await response.text();
    // Split by newlines and filter out empty lines
    return text.split('\n')
      .map(line => line.trim())
      .filter(line => line.length > 0);
  } catch (err) {
    console.error("Failed to fetch commit hashes:", err);
    return [];
  }
}

function populateCommitDropdown(commits, selectId) {
  const select = document.getElementById(selectId);
  select.innerHTML = "";
  commits.forEach(commit => {
    const opt = document.createElement("option");
    opt.value = opt.textContent = commit;
    select.appendChild(opt);
  });
}

function loadCSV(commit, panel) {
  Papa.parse(`https://raw.githubusercontent.com/linkedin/Liger-Kernel/refs/heads/gh-pages/benchmarks/${commit}/benchmark.csv`, {
    download: true,
    header: true,
    dynamicTyping: true,
    complete: (result) => {
      const data = result.data.filter(d => d.kernel_provider);
      if (panel === 1) {
        setupControls(data, 1);
        renderCharts(data, 1);
      } else {
        setupControls(data, 2);
        renderCharts(data, 2);
      }
    },
    error: (err) => {
      alert("Failed to load CSV for commit: " + commit);
      console.error(err);
    }
  });
}

function setupControls(data, panel) {
  const kernelSet = new Set(data.map(d => d.kernel_name));
  const modeSet = new Set(data.map(d => d.kernel_operation_mode));

  const kernelSelect = document.getElementById(panel === 1 ? "kernel" : "kernel2");
  const modeSelect = document.getElementById(panel === 1 ? "mode" : "mode2");

  kernelSelect.innerHTML = "";
  modeSelect.innerHTML = "";

  kernelSet.forEach(k => {
    const opt = document.createElement("option");
    opt.value = opt.textContent = k;
    kernelSelect.appendChild(opt);
  });

  modeSet.forEach(m => {
    const opt = document.createElement("option");
    opt.value = opt.textContent = m;
    modeSelect.appendChild(opt);
  });

  kernelSelect.addEventListener("change", () => renderCharts(data, panel));
  modeSelect.addEventListener("change", () => renderCharts(data, panel));
}

function renderCharts(data, panel) {
  const kernel = document.getElementById(panel === 1 ? "kernel" : "kernel2").value;
  const mode = document.getElementById(panel === 1 ? "mode" : "mode2").value;

  // Render speed chart
  renderChart(data, panel, "speed", kernel, mode);
  // Render memory chart
  renderChart(data, panel, "memory", kernel, mode);
}

function renderChart(data, panel, metric, kernel, mode) {
  console.log(`Rendering ${metric} chart for panel ${panel}`);
  console.log('Available metrics:', [...new Set(data.map(d => d.metric_name))]);
  
  const filtered = data.filter(
    d =>
      d.kernel_name === kernel &&
      d.metric_name.toLowerCase() === metric.toLowerCase() &&
      d.kernel_operation_mode === mode
  );

  console.log(`Filtered data for ${metric}:`, filtered);

  const batchSizes = [...new Set(filtered.map(d => d.x_value))].sort((a, b) => a - b);
  const providers = [...new Set(filtered.map(d => d.kernel_provider))];

  console.log(`Batch sizes for ${metric}:`, batchSizes);
  console.log(`Providers for ${metric}:`, providers);

  const datasets = providers.map(provider => {
    const values = batchSizes.map(bs => {
      const row = filtered.find(d => d.kernel_provider === provider && d.x_value === bs);
      return row ? row.y_value_50 : null;
    });

    let borderColor;
    if (provider === "liger") {
      borderColor = "orange";
    } else if (provider === "huggingface") {
      borderColor = "steelblue";
    } else {
      borderColor = "green";  // For any third provider (like torch_compile)
    }

    return {
      label: provider,
      data: values,
      borderColor: borderColor,
      backgroundColor: "transparent",
      tension: 0.2,
      pointRadius: 4,
      pointHoverRadius: 6,
    };
  });

  const chartId = panel === 1 
    ? (metric === "speed" ? "benchmarkChartSpeed" : "benchmarkChartMemory")
    : (metric === "speed" ? "benchmarkChart2Speed" : "benchmarkChart2Memory");

  const ctx = document.getElementById(chartId).getContext("2d");
  
  // Destroy existing chart if it exists
  if (panel === 1) {
    if (metric === "speed" && chartSpeed) chartSpeed.destroy();
    if (metric === "memory" && chartMemory) chartMemory.destroy();
  } else {
    if (metric === "speed" && chart2Speed) chart2Speed.destroy();
    if (metric === "memory" && chart2Memory) chart2Memory.destroy();
  }

  const newChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: batchSizes,
      datasets: datasets,
    },
    options: {
      responsive: true,
      plugins: {
        title: {
          display: true,
          text: `Benchmark - ${kernel} - ${metric} (${mode})`,
        },
        legend: {
          position: "top",
        }
      },
      scales: {
        x: {
          title: { display: true, text: "Batch Size (B)" },
        },
        y: {
          title: { display: true, text: metric },
          beginAtZero: true
        }
      }
    }
  });

  // Store the chart reference
  if (panel === 1) {
    if (metric === "speed") chartSpeed = newChart;
    else chartMemory = newChart;
  } else {
    if (metric === "speed") chart2Speed = newChart;
    else chart2Memory = newChart;
  }
}
