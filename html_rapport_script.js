$(document).ready(function () {

    $("table")
        .removeAttr("border")
        .addClass("table table-striped table-hover table-sm");

    $("button.toggle-button")
        .on("click", function (e) {
            $(e.target).siblings().closest(".toggle-visible").toggle();
        });

    chart_general_accuracies(document.getElementById("chart_general_accuracies"), general_acuracies_data.count);
    chart_confusion_matrices(document.getElementById("chart_confusion_matrices"), confusion_matrices_data);
    chart_confusion_matrices(document.getElementById("chart_confusion_matrices_consolidated"), confusion_matrices_consolidated_data);

    function chart_general_accuracies(elem, data) {
        var chartLegend = [];
        var chartData = [];

        Object.keys(data).forEach(function (key, index) {
            chartLegend.push(key);
            chartData.push({ name: key, value: data[key] });
        });

        var chart = echarts.init(elem);
        var chartOptions = {
            tooltip: {
                formatter: "{a} <br/>{b}: {c} ({d}%)"
            },
            legend: {
                orient: 'vertical',
                x: 'left',
                data: chartLegend
            },
            toolbox: {
                show: true,
                feature: {
                    dataView: { show: true, readOnly: false },
                    restore: { show: true },
                    saveAsImage: { show: true }
                }
            },
            calculable: true,
            series: {
                name: 'accuracies',
                type: 'pie',
                radius: ['50%', '70%'],
                avoidLabelOverlap: false,
                label: {
                    normal: {
                        show: false,
                        position: 'center'
                    },
                    emphasis: {
                        show: true,
                        textStyle: {
                            fontSize: '30',
                            fontWeight: 'bold'
                        }
                    }
                },
                labelLine: {
                    normal: {
                        show: false
                    }
                },
                data: chartData
            }
        };

        if (chartOptions && typeof chartOptions === "object") {
            chart.setOption(chartOptions, true);
        }
    }
    function chart_confusion_matrices(elem, data) {

        // categories opvullen
        var rows = Object.keys(data).filter(x => !x.startsWith("nb_") && !x.startsWith("pct_"));
        var series = [];
        var categories = [];

        // data is geformateerd per column

        rows.forEach(function (rowKey, rowIndex) {
            categories.push(rowKey);

            rows.forEach(function (columnKey, columnIndex) {
                // per rij loopen en de kolom waarde (=predicted) ophalen
                series.push([rowIndex, columnIndex, data[rowKey][columnKey] || "-"]);
            });
        });

        var chart = echarts.init(elem);
        var chartOptions = {
            tooltip: {
                position: 'top',
                formatter: function (params) {
                    var categoryX = categories[params.value[0]];
                    var categoryY = categories[params.value[1]];
                    var nb_input = data.nb_input[categoryX];
                    var nb_predicted = data.nb_predicted[categoryX];
                    var pct_predicted_correct = data.pct_predicted_correct[categoryX];
                    var pct_predicted_wrong = data.pct_predicted_wrong[categoryX];

                    return categoryX + ", " + categoryY + ": " + params.value[2] + "<br/>" +
                        "input: " + nb_input + ", predicted: " + nb_predicted + "<br/>" +
                        "correct: " + (pct_predicted_correct || 0).toFixed(2) + "%, wrong " + (pct_predicted_wrong || 0).toFixed(2) + "%";
                }
            },
            legend: {
                data: categories
            },
            grid: {
                left: 2,
                bottom: 10,
                right: 10,
                containLabel: true
            },
            xAxis: {
                type: 'category',
                data: categories,
                boundaryGap: false
            },
            yAxis: {
                type: 'category',
                data: categories
            },
            series: {
                name: "confusion matrices",
                type: "scatter",
                data: series,
                large: true,
                itemStyle: {
                    color: function (params) {
                        return params.value[0] === params.value[1]
                            ? "green"
                            : "red";
                    }
                },
                symbolSize: function (val) {
                    if (val[2] === "-") return 0;
                    //if (val[0] === val[1]) return 10;

                    var total = data.nb_input[categories[val[0]]];
                    var pct = (val[2] / total) * 100;
                    return pct;
                }
            }
        };

        if (chartOptions && typeof chartOptions === "object") {
            chart.setOption(chartOptions, true);
        }
    }
 });