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

        // data verwerken
        var rows = Object.keys(data).filter(x => !x.startsWith("nb_") && !x.startsWith("pct_"));
        var series = [];
        var categories = [];

        rows.forEach(function (rowKey, rowIndex) {
            categories.push(rowKey);

            rows.forEach(function (columnKey, columnIndex) {
                // per rij loopen en de kolom waarde (=predicted value) ophalen
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

                    var nb_input_x = data.nb_input[categoryX];
                    var nb_predicted_x = data.nb_predicted[categoryX];
                    var nb_predicted_x_correct = data.nb_predicted_correct[categoryX];
                    var nb_predicted_x_wrong = data.nb_predicted_wrong[categoryX];
                    var pct_predicted_x_correct = data.pct_predicted_correct[categoryX];
                    var pct_predicted_x_wrong = data.pct_predicted_wrong[categoryX];

                    var nb_input_y = data.nb_input[categoryY];
                    var nb_predicted_y = data.nb_predicted[categoryY];
                    var nb_predicted_y_correct = data.nb_predicted_correct[categoryY];
                    var nb_predicted_y_wrong = data.nb_predicted_wrong[categoryY];
                    var pct_predicted_y_correct = data.pct_predicted_correct[categoryY];
                    var pct_predicted_y_wrong = data.pct_predicted_wrong[categoryY];

                    return params.value[2] + "/" + nb_input_x + "<br/>" +
                        categoryX + " (input: " + nb_input_x + ", predicted: " + nb_predicted_x + ")<br/>" +
                        "&nbsp;&nbsp;&nbsp;<span class='oi oi-thumb-up text-success'></span>" + (pct_predicted_x_correct || 0).toFixed(2) + "%" + 
                            "(" + nb_predicted_x_correct + "/" + nb_input_x + "), <span class='oi oi-thumb-down text-danger'></span>" + (pct_predicted_x_wrong || 0).toFixed(2) + "% (" + nb_predicted_x_wrong + "/" + nb_input_x + ")<br/>" +
                        categoryY + " (input: " + nb_input_y + ", predicted: " + nb_predicted_y + ")<br/>" +
                        "&nbsp;&nbsp;&nbsp;<span class='oi oi-thumb-up text-success'></span>" + (pct_predicted_y_correct || 0).toFixed(2) + "%" +
                            "(" + nb_predicted_y_correct + "/" + nb_input_y + "), <span class='oi oi-thumb-down text-danger'></span>" + (pct_predicted_y_wrong || 0).toFixed(2) + "% (" + nb_predicted_y_wrong + "/" + nb_input_y + ")<br/>";
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

                    var denomitor = data.nb_predicted[categories[val[0]]];
                    if (categories[val[0]] === "DOUBT") {
                        // NOTE: If we're comparing with DOUBT, then use the predicted value of the comparing crop. This provides more information.
                        denomitor = data.nb_predicted[categories[val[1]]];
                    }

                    return (val[2] / denomitor) * 100;
                }
            }
        };

        if (chartOptions && typeof chartOptions === "object") {
            chart.setOption(chartOptions, true);
        }
    }
 });