import os
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import norm
import numpy as np

import pandas as pd
import os
import plotly.express as px

def plot_error_distributions_map(best_sensor_rmse, best_sensor_mae, best_sensor_accuracy, best_sensor_r2, best_sensor_variance, output_path, sensor_ids, geocoordinates):
    import plotly.graph_objects as go

    save_path = os.path.join(output_path, "stats")

    metrics_to_plot = [
        ("Test Root Mean Square Error (RMSE)", "RMSE", best_sensor_rmse),
        ("Test Mean Absolute Error (MAE)", "MAE", best_sensor_mae),
        ("Test Accuracy (Acc)", "Acc", best_sensor_accuracy),
        ("Test Coefficient of Determination (R²)", "R2", best_sensor_r2),
        ("Test Variance (Var)", "Var", best_sensor_variance)
    ]

    # 准备经纬度
    lats, lons = [], []
    for sensor_id in sensor_ids:
        row = geocoordinates.loc[sensor_id]
        lats.append(row["lat"])
        lons.append(row["long"])

    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)

    for metric_label, metric_name, metric_values in metrics_to_plot:
        metric_values_sorted = list(metric_values)
        print(f"Generating {metric_name} stats...")

        color_min = 0 if any(v < 0 for v in metric_values_sorted) else min(metric_values_sorted)
        color_max = max(metric_values_sorted)

        fig = go.Figure(go.Scattergeo(
            lat=lats,
            lon=lons,
            mode='markers',
            marker=dict(
                size=12,
                color=metric_values_sorted,
                colorscale='Viridis',
                cmin=color_min,
                cmax=color_max,
                colorbar=dict(
                    title=metric_name,
                    tickfont=dict(size=14)
                ),
                showscale=True
            ),
            text=[f"Sensor {sid}<br>{metric_name}: {v:.4f}"
                  for sid, v in zip(sensor_ids, metric_values_sorted)],
            hoverinfo='text'
        ))

        fig.update_layout(
            title=dict(
                text=f"Heatmap of {metric_label}",
                yanchor="top", y=0.99,
                xanchor="center", x=0.5,
                font=dict(size=24)
            ),
            font_family="Times New Roman",
            font_color="#333333",
            geo=dict(
                scope='usa',
                projection_type='albers usa',
                showland=True,
                landcolor='rgb(243, 243, 243)',
                showocean=True,
                oceancolor='rgb(204, 230, 255)',
                showlakes=True,
                lakecolor='rgb(204, 230, 255)',
                showrivers=True,
                rivercolor='rgb(204, 230, 255)',
                showcountries=True,
                countrycolor='rgb(200, 200, 200)',
                showsubunits=True,
                subunitcolor='rgb(200, 200, 200)',
                center=dict(lat=center_lat, lon=center_lon),
                lonaxis=dict(range=[center_lon - 0.5, center_lon + 0.5]),
                lataxis=dict(range=[center_lat - 0.3, center_lat + 0.3]),
            ),
            margin=dict(l=0, r=0, t=60, b=0)
        )

        fig.write_image(
            save_path + f'/{metric_name.lower()}_heatmap.jpg',
            format="jpeg", width=2000, height=1200, scale=2
        )


def plot_error_distributions(best_sensor_rmse, best_sensor_mae, best_sensor_accuracy, best_sensor_r2, best_sensor_variance, output_path):
    save_path = os.path.join(output_path, "stats")

    metrics_to_plot = [
        ("Test Root Mean Square Error (RMSE)", "RMSE", best_sensor_rmse),
        ("Test Mean Absolute Error (MAE)", "MAE", best_sensor_mae),
        ("Test Accuracy (Acc)", "Acc",best_sensor_accuracy),
        ("Test Coefficient of Determination (R²)", "R²", best_sensor_r2),        
        ("Test Variance (Var)", "Var", best_sensor_variance)
    ]

    for metric_label, metric_name, metric_values in metrics_to_plot:
        metric_values = sorted(metric_values)

        print(f"Generating {metric_name} stats...")

        fit_data = stats.norm.cdf(metric_values, np.mean(metric_values), np.std(metric_values))

        plt.title(f"Normal Distribution of the Min {metric_name} Values for All Sensors", fontweight="bold",
                  color="#333333", pad=10, fontname='Times New Roman', fontdict={'fontsize': 12})
        plt.yticks(fontweight="bold", color="#333333",
                   fontname='Times New Roman', fontsize=8)
        plt.xticks(fontweight="bold", color="#333333",
                   fontname='Times New Roman', fontsize=8)

        plt.plot(metric_values, fit_data, linewidth=1, marker=".", markersize=2, color="#ffd700", label=f"{metric_name}, σ = " +
                 str(round(np.std(metric_values), 2)) + ", µ = " + str(round(np.mean(metric_values), 2)),)
        plt.legend(loc='best', fontsize=8)

        plt.axvline(x=np.mean(metric_values), linewidth=1,
                    color='#ffd700', ls='--', label='axvline - full height')

        plt.xlabel(f'{metric_label}', fontweight="bold", color="#333333",
                   fontname='Times New Roman', fontdict={'fontsize': 10}, labelpad=8)
        plt.ylabel('Frequency in %', fontweight="bold", color="#333333",
                   fontname='Times New Roman', fontdict={'fontsize': 10}, labelpad=8)

        plt.savefig(save_path+f'/{metric_name.lower()}_distribution.jpg', dpi=500)
        plt.close()





def plot_additional_errors(test_rmse, train_rmse, train_loss, alpha1, save_path):
    # plot additional errors for the model
    # train_rmse & test_rmse
    fig1 = plt.figure(figsize=(6, 4))
    plt.title(str("Training vs Test Root Mean Square Error (RMSE)"), fontweight="bold",
              color="#333333", pad=10, fontname='Times New Roman', fontdict={'fontsize': 12})
    # x y labels
    plt.xlabel('Epochs', fontweight="bold", color="#333333",
               fontname='Times New Roman', fontdict={'fontsize': 10}, labelpad=8)
    plt.ylabel('RMSE', fontweight="bold", color="#333333",
               fontname='Times New Roman', fontdict={'fontsize': 10}, labelpad=8)
    # x y ticks
    plt.yticks(fontweight="bold", color="#333333",
               fontname='Times New Roman', fontsize=8)
    plt.xticks(fontweight="bold", color="#333333",
               fontname='Times New Roman', fontsize=8)
    plt.plot(train_rmse, label="train_rmse", color="#ffd700")
    plt.plot(test_rmse, label="test_rmse", color="#333333")
    plt.legend(loc='best', fontsize=8)
    # save the figure
    plt.savefig(save_path+'/rmse.jpg', dpi=500)
    # plt.show()
    # close the figure
    plt.close()

    # train_loss
    fig1 = plt.figure(figsize=(6, 4))
    plt.title(str("Training Loss"), fontweight="bold", color="#333333",
              pad=10, fontname='Times New Roman', fontdict={'fontsize': 10})
    # x y labels
    plt.xlabel('Epochs', fontweight="bold", color="#333333",
               fontname='Times New Roman', fontdict={'fontsize': 10}, labelpad=8)
    plt.ylabel('Loss', fontweight="bold", color="#333333",
               fontname='Times New Roman', fontdict={'fontsize': 10}, labelpad=8)
    # x y ticks
    plt.yticks(fontweight="bold", color="#333333",
               fontname='Times New Roman', fontsize=8)
    plt.xticks(fontweight="bold", color="#333333",
               fontname='Times New Roman', fontsize=8)
    plt.plot(train_loss, label='train_loss', color="#333333")
    plt.yscale('log')
    plt.legend(loc='best', fontsize=8)
    plt.savefig(save_path+'/train_loss.jpg', dpi=500)
    # save the figure
    # plt.show()
    # close the figure
    plt.close()

    # train_rmse
    fig1 = plt.figure(figsize=(6, 4))
    plt.title(str("Training Root Mean Square Error (RMSE)"), fontweight="bold",
              color="#333333", pad=10, fontname='Times New Roman', fontdict={'fontsize': 12})
    # x y labels
    plt.xlabel('Epochs', fontweight="bold", color="#333333",
               fontname='Times New Roman', fontdict={'fontsize': 10}, labelpad=8)
    plt.ylabel('RMSE', fontweight="bold", color="#333333",
               fontname='Times New Roman', fontdict={'fontsize': 10}, labelpad=8)
    # x y ticks
    plt.yticks(fontweight="bold", color="#333333",
               fontname='Times New Roman', fontsize=8)
    plt.xticks(fontweight="bold", color="#333333",
               fontname='Times New Roman', fontsize=8)
    plt.plot(train_rmse, label='train_rmse', color="#333333")
    plt.legend(loc='best', fontsize=8)
    plt.savefig(save_path+'/train_rmse.jpg', dpi=500)
    # save the figure
    # plt.show()
    # close the figure
    plt.close()

    # alpha
    fig1 = plt.figure(figsize=(6, 4))
    ax1 = fig1.add_subplot(1, 1, 1)
    plt.title(str("Test Alpha"), fontweight="bold", color="#333333",
              pad=10, fontname='Times New Roman', fontdict={'fontsize': 10})
    # x y labels
    # plt.xlabel('Epochs',fontweight="bold", color="#333333", fontname = 'Times New Roman', fontdict={'fontsize':18}, labelpad=8)
    # plt.ylabel('Alpha',fontweight="bold", color="#333333", fontname = 'Times New Roman', fontdict={'fontsize':18}, labelpad=8)
    # x y ticks
    plt.yticks(fontweight="bold", color="#333333",
               fontname='Times New Roman', fontsize=8)
    plt.xticks(fontweight="bold", color="#333333",
               fontname='Times New Roman', fontsize=8)
    plt.plot(np.sum(alpha1, 0), label="alpha", color="#333333")
    plt.legend(loc='best', fontsize=8)
    plt.savefig(save_path+'/alpha.jpg', dpi=500)
    # save the figure
    # plt.show()
    # close the figure
    plt.close()

    # alpha bar
    plt.title(str("Test Alpha"), fontweight="bold", color="#333333",
              pad=10, fontname='Times New Roman', fontdict={'fontsize': 10})
    # x y labels
    # plt.xlabel('Epochs',fontweight="bold", color="#333333", fontname = 'Times New Roman', fontdict={'fontsize':18}, labelpad=8)
    # plt.ylabel('Alpha',fontweight="bold", color="#333333", fontname = 'Times New Roman', fontdict={'fontsize':18}, labelpad=8)
    # x y ticks
    plt.yticks(fontweight="bold", color="#333333",
               fontname='Times New Roman', fontsize=8)
    plt.xticks(fontweight="bold", color="#333333",
               fontname='Times New Roman', fontsize=8)
    plt.imshow(np.mat(np.sum(alpha1, 0)))
    plt.legend(loc='best', fontsize=8)
    plt.savefig(save_path+'/alpha11.jpg', dpi=500)
    # save the figure
    # plt.show()
    # close the figure
    plt.close()