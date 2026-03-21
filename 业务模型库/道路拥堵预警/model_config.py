import torch

models_map = {
        'GCN_LSTM_BI_Multi_Attention_Weather_Separate': ("models.GCN_LSTM_BI_Multi_Attention_Weather_Separate.GCN_LSTM_BI_Multi_Attention_Weather_Separate", {
            "in_channels": None,
            "hidden_channels": 64,
            "num_gcn_layers": 64,
            "num_rnn_layers": 3,          
            "dropout": 0,
            "num_lags": 8,
        }),
       
}

def init_model(model_type, train_data, num_predictions, dropout=0):
    model_path, default_params = models_map[model_type]
    model_module, model_name = model_path.rsplit('.', 1)
    model_class = getattr(__import__(model_module, fromlist=[model_name]), model_name)

    # Set in_channels to the number of input features
    if "in_channels" in default_params:
        default_params["in_channels"] = train_data.size(1)
    if "speed_channels" in default_params:
        default_params["speed_channels"] = train_data.size(1)
    if "temp_channels" in default_params:
        default_params["temp_channels"] = train_data.size(1)
    default_params["num_predictions"] = num_predictions


    # Merge default params from models_map with provided params, with the latter taking precedence
    params = {
        **default_params  # Overwrite values from models_map with provided values
    }

    # Print params for debugging purposes
    print(f"Parameters being used: {params}")

    model = model_class(**params)

    # Post-processing for specific models
    if model_type == 'ARIMA_NN':
        train_data = train_data.to(dtype=torch.float32)
        numpy_train_data = train_data.numpy()
        model.arima.fit(numpy_train_data)

    elif model_type == 'SVR':
        train_data = train_data.to(dtype=torch.float32)
        numpy_train_data = train_data.numpy()
        model.svr.fit(numpy_train_data)

    return model
