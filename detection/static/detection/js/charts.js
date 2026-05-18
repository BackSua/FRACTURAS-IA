/**
 * charts.js — Lógica de inicialización de gráficas Chart.js para el dashboard.
 *
 * Las gráficas se renderizan en el NAVEGADOR del usuario, no en el servidor.
 * Esto significa cero carga de CPU/RAM en el servidor — solo se envía el JSON
 * con los datos y Chart.js (~60KB de JS) hace todo el trabajo visual.
 *
 * Los datos vienen del backend Django como JSON embebido en el template HTML.
 */

/**
 * Gráfica 1 — Curvas de Loss por Época.
 *
 * Muestra cómo evoluciona el error (loss) durante el entrenamiento.
 * Train Loss debe bajar consistentemente.
 * Val Loss debe bajar también, pero si sube mientras train baja => overfitting.
 */
function initLossChart(data) {
    const ctx = document.getElementById('lossChart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.epochs,
            datasets: [
                {
                    label: 'Train Box Loss',
                    data: data.train_box_loss,
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.3,
                    pointRadius: 0,
                    pointHitRadius: 10,
                },
                {
                    label: 'Val Box Loss',
                    data: data.val_box_loss,
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.3,
                    pointRadius: 0,
                    pointHitRadius: 10,
                },
                {
                    label: 'Train Cls Loss',
                    data: data.train_cls_loss,
                    borderColor: '#e67e22',
                    borderWidth: 1.5,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.3,
                    pointRadius: 0,
                    pointHitRadius: 10,
                },
                {
                    label: 'Val Cls Loss',
                    data: data.val_cls_loss,
                    borderColor: '#2ecc71',
                    borderWidth: 1.5,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.3,
                    pointRadius: 0,
                    pointHitRadius: 10,
                },
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { usePointStyle: true, padding: 15 }
                },
                tooltip: { mode: 'index', intersect: false }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Época' }
                },
                y: {
                    title: { display: true, text: 'Loss' },
                    beginAtZero: false,
                }
            },
            interaction: { mode: 'nearest', axis: 'x', intersect: false }
        }
    });
}


/**
 * Gráfica 2 — Métricas por Época (Precision, Recall, mAP).
 *
 * Muestra la evolución de las métricas de evaluación durante el entrenamiento.
 * Todas deberían subir y estabilizarse. Si bajan al final => overfitting.
 */
function initMetricsChart(data) {
    const ctx = document.getElementById('metricsChart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.epochs,
            datasets: [
                {
                    label: 'mAP@50',
                    data: data.mAP50,
                    borderColor: '#1a5276',
                    borderWidth: 2.5,
                    fill: false,
                    tension: 0.3,
                    pointRadius: 0,
                    pointHitRadius: 10,
                },
                {
                    label: 'mAP@50:95',
                    data: data.mAP50_95,
                    borderColor: '#8e44ad',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.3,
                    pointRadius: 0,
                    pointHitRadius: 10,
                },
                {
                    label: 'Precision',
                    data: data.precision,
                    borderColor: '#27ae60',
                    borderWidth: 1.5,
                    borderDash: [4, 4],
                    fill: false,
                    tension: 0.3,
                    pointRadius: 0,
                    pointHitRadius: 10,
                },
                {
                    label: 'Recall',
                    data: data.recall,
                    borderColor: '#e67e22',
                    borderWidth: 1.5,
                    borderDash: [4, 4],
                    fill: false,
                    tension: 0.3,
                    pointRadius: 0,
                    pointHitRadius: 10,
                },
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { usePointStyle: true, padding: 15 }
                },
                tooltip: { mode: 'index', intersect: false }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Época' }
                },
                y: {
                    title: { display: true, text: 'Valor' },
                    min: 0,
                    max: 1,
                }
            },
            interaction: { mode: 'nearest', axis: 'x', intersect: false }
        }
    });
}


/**
 * Gráfica 3 — Métricas Finales del Modelo (Barras).
 *
 * Resumen visual del rendimiento final sobre el conjunto de test.
 * Estos son los números que importan: rendimiento en datos NUNCA VISTOS.
 */
function initFinalMetricsChart(evalData) {
    const ctx = document.getElementById('finalMetricsChart');
    if (!ctx || !evalData.metrics) return;

    const metrics = evalData.metrics;
    const labels = ['Precision', 'Recall', 'F1-Score', 'mAP@50', 'mAP@50:95'];
    const values = [
        metrics.precision,
        metrics.recall,
        metrics.f1_score,
        metrics.mAP50,
        metrics.mAP50_95,
    ];
    const colors = ['#27ae60', '#e67e22', '#2980b9', '#1a5276', '#8e44ad'];

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Valor',
                data: values,
                backgroundColor: colors.map(c => c + 'CC'),
                borderColor: colors,
                borderWidth: 2,
                borderRadius: 6,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return (context.raw * 100).toFixed(1) + '%';
                        }
                    }
                }
            },
            scales: {
                x: {
                    min: 0,
                    max: 1,
                    ticks: {
                        callback: function(value) {
                            return (value * 100) + '%';
                        }
                    },
                    title: { display: true, text: 'Porcentaje' }
                },
            }
        }
    });
}


/**
 * Actualiza las tarjetas de métricas clave en la parte superior del dashboard.
 */
function updateMetricCards(evalData) {
    if (!evalData.metrics) return;

    const m = evalData.metrics;

    const setCard = (id, value) => {
        const el = document.getElementById(id);
        if (el) el.textContent = (value * 100).toFixed(1) + '%';
    };

    setCard('metric-map50', m.mAP50);
    setCard('metric-precision', m.precision);
    setCard('metric-recall', m.recall);
    setCard('metric-f1', m.f1_score);
}
