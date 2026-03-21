# Author 2023 Thomas Fink

import os
from datetime import datetime
import torch
import torch.nn as nn

import helpers.metrics as metrics
import helpers.visualisation as visualisation
import helpers.data as data
import helpers.stats as stats
import helpers.output as output

from model_config import init_model


if __name__ == "__main__":

    OS_PATH = "./"
    DATA_SET = "metr-la"

    data_csv_file      = f"{OS_PATH}/data/{DATA_SET}/weather/merged_speed_traffic_and_air_temperature_data.csv"
    adjacency_csv_file = f"{OS_PATH}/data/{DATA_SET}/traffic/adj.csv"

    data_df = data.load_data(data_csv_file)[1:]
    data_df = data_df.iloc[:, 1:]

    adjacency_df = data.load_adjacency_matrix(adjacency_csv_file)
    sensor_ids   = adjacency_df.iloc[:, 0].tolist()
    adjacency_df = adjacency_df.iloc[:, 1:]

    data_normalized, scaler = data.normalize_data(data_df)
    traffic_data = torch.transpose(data_normalized, 0, 1)

    adjacency_matrix = torch.tensor(adjacency_df.values, dtype=torch.float)
    edge_index  = adjacency_matrix.nonzero(as_tuple=False).t().contiguous()
    edge_weight = adjacency_matrix[edge_index[0], edge_index[1]]

    timestamp           = datetime.now().strftime('%Y%m%d_%H%M%S')
    timestamped_dataset = f"{timestamp}_{DATA_SET}"
    output_path = output.create_output_directories(OS_PATH, timestamped_dataset, sensor_ids)

    num_epochs      = 15
    num_predictions = 288
    train_data = traffic_data[:, :-num_predictions]
    test_data  = traffic_data[:, -num_predictions:]

    model = init_model(
        model_type="GCN_LSTM_BI_Multi_Attention_Weather_Separate",
        train_data=train_data,
        num_predictions=num_predictions,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.027, weight_decay=0)

    loss_list, rmse_list, mae_list = [], [], []
    sensor_rmse_lists, sensor_mae_lists = [], []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        predictions = model(train_data, edge_index, edge_weight)
        loss = criterion(predictions, test_data)
        loss.backward()
        optimizer.step()

        if epoch == num_epochs - 1:
            predictions = torch.where(predictions < 0, torch.zeros_like(predictions), predictions)

        rmse, mae = metrics.evaluate(
            predictions.detach().numpy().T,
            test_data.detach().numpy().T
        )
        print(f"Epoch: {epoch + 1}, Loss: {loss.item():.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        loss_list.append(loss.item())
        rmse_list.append(rmse)
        mae_list.append(mae)

        sensor_rmses, sensor_maes = [], []
        for sensor_idx in range(len(sensor_ids)):
            sensor_pred   = predictions[sensor_idx, :].detach().numpy()
            sensor_actual = test_data[sensor_idx, :].detach().numpy()
            s_rmse, s_mae = metrics.evaluate(sensor_pred, sensor_actual)
            sensor_rmses.append(s_rmse)
            sensor_maes.append(s_mae)

        sensor_rmse_lists.append(sensor_rmses)
        sensor_mae_lists.append(sensor_maes)

    metrics.visualize_metric(loss_list, 'Loss', output_path)
    metrics.visualize_metric(rmse_list, 'RMSE', output_path)
    metrics.visualize_metric(mae_list,  'MAE',  output_path)

    best_sensor_rmse = [float('inf')] * len(sensor_ids)
    best_sensor_mae  = [float('inf')] * len(sensor_ids)

    for sensor_idx in range(len(sensor_ids)):
        for epoch in range(num_epochs):
            best_sensor_rmse[sensor_idx] = min(best_sensor_rmse[sensor_idx], sensor_rmse_lists[epoch][sensor_idx])
            best_sensor_mae[sensor_idx]  = min(best_sensor_mae[sensor_idx],  sensor_mae_lists[epoch][sensor_idx])

    stats.plot_error_distributions(best_sensor_rmse, best_sensor_mae, output_path)

    geocoordinates_csv_file = f"{OS_PATH}/data/{DATA_SET}/sensors/metr_la_sensors_traffic.csv"
    geocoordinates_df = None

    if os.path.exists(geocoordinates_csv_file):
        geocoordinates_df = data.load_geocoordinates(geocoordinates_csv_file)
        stats.plot_error_distributions_map_osm(
            best_sensor_rmse, best_sensor_mae,
            output_path, sensor_ids, geocoordinates_df
        )
    else:
        print(f"File {geocoordinates_csv_file} does not exist! No heat maps will be created!")

    actual_data_np = scaler.inverse_transform(test_data.detach().numpy().T).T
    predictions_np = scaler.inverse_transform(predictions.detach().numpy().T).T

    if geocoordinates_df is not None:
        stats.plot_predicted_speed_map(
            predictions_np, output_path, sensor_ids, geocoordinates_df,
            timestep_interval_min=5, target_hour=12
        )

    print("Saving predictions...")
    visualisation.save_predictions_to_csv(predictions_np, output_path)

    print("Generating sensor predictions...")
    for sensor_idx, (sensor_id, (pred, actual)) in enumerate(zip(sensor_ids, zip(predictions_np, actual_data_np))):
        sensor_id_str = str(int(sensor_id)) if isinstance(sensor_id, float) and sensor_id.is_integer() else str(sensor_id)
        sensor_folder = os.path.join(output_path, "sensors", f"sensor_{sensor_id_str}")
        visualisation.plot_prediction(pred, actual, sensor_id_str, sensor_folder)

    print("Generating sensor metrics...")
    for sensor_idx, sensor_id in enumerate(sensor_ids):
        sensor_id_str = str(int(sensor_id)) if isinstance(sensor_id, float) and sensor_id.is_integer() else str(sensor_id)
        sensor_folder = os.path.join(output_path, "sensors", f"sensor_{sensor_id_str}")
        metrics.visualize_metric([x[sensor_idx] for x in sensor_rmse_lists], 'RMSE', sensor_folder)
        metrics.visualize_metric([x[sensor_idx] for x in sensor_mae_lists],  'MAE',  sensor_folder)