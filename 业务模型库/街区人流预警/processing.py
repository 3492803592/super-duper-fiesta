import pandas as pd
import numpy as np
import chinese_calendar as cc
#人流数据处理
def people_data_processing(df,area_df):
    print("——————————开始处理人流数据——————————")
    df.columns = df.columns.astype(str)
    time_col = df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.dropna(subset=[time_col]).sort_values(time_col)

    value_cols = df.columns[1:]
    df[value_cols] = df[value_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    df[value_cols] = df[value_cols].clip(lower=0, upper=50000)

    #  读入面积表
    area_df.columns = area_df.columns.astype(str)

    id_col = area_df.columns[0]
    area_col = area_df.columns[6]

    # 将面积数据改成字典：{摄像头编号: area值}
    area_map = dict(zip(area_df[id_col].astype(str), area_df[area_col]))

    # 按摄像头编号对齐面积 
    camera_areas = []
    for cam in value_cols:
        if cam in area_map:
            camera_areas.append(area_map[cam])
        else:
            camera_areas.append(1)

    camera_areas = pd.Series(camera_areas, index=value_cols)

    # 计算密度：密度 = count / area 
    density_df_1m = df[value_cols].div(camera_areas, axis=1)*100
    density_df_1m.columns = [f"{c}_density" for c in value_cols]
    density_df_1m[time_col] = df[time_col]
    #10min聚合密度计算
    df = df.set_index(time_col)
    freq = '10min'  
    agg_df = df[value_cols].resample(freq).mean().reset_index()

    density_df_10m = agg_df[value_cols].div(camera_areas, axis=1)*100
    density_df_10m.columns = [f"{c}_density" for c in value_cols]
    density_df_10m[time_col] = agg_df[time_col]
   
    
    print("人流数据处理完成！")

    return density_df_1m, density_df_10m


#天气数据处理
def weather_data_processing(weather_df):
    print("——————————开始处理天气数据——————————"    )
    weather_df.columns = weather_df.columns.astype(str)
    time_col = weather_df.columns[0]
    weather_df[time_col] = pd.to_datetime(weather_df[time_col], errors='coerce')
    weather_df = weather_df.dropna(subset=[time_col]).sort_values(time_col)
    value_cols = weather_df.columns[1:]
    weather_df[value_cols] = weather_df[value_cols].apply(pd.to_numeric, errors='coerce').fillna(method='ffill')
    #重采样为1分钟/10分钟频率
    weather_df_1m = weather_df.set_index(time_col).resample('1min').ffill().reset_index()
    weather_df_10m = weather_df.set_index(time_col).resample('10min').ffill().reset_index()
    weather_df_1m = add_time_features(weather_df_1m, time_col)
    weather_df_10m = add_time_features(weather_df_10m, time_col)
    print("天气数据处理完成！")
    return weather_df_1m,weather_df_10m

#加入时间特征
def add_time_features(df, time_col="Time"):
    
    if time_col not in df.columns:
        raise ValueError(f"{time_col} 不存在")

    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df["year"] = df[time_col].dt.year
    df["month"] = df[time_col].dt.month
    df["day"] = df[time_col].dt.day
    df["hour"] = df[time_col].dt.hour
    df["minute"] = df[time_col].dt.minute
    df["weekday"] = df[time_col].dt.weekday  # 0=周一，6=周日

    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    df["is_holiday"] = df[time_col].apply(lambda x: int(cc.is_holiday(x)))
    df["is_workday"] = df[time_col].apply(lambda x: int(cc.is_workday(x)))

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

    day_minutes = df["hour"] * 60 + df["minute"]
    df["day_sin"] = np.sin(2 * np.pi * day_minutes / (24 * 60))
    df["day_cos"] = np.cos(2 * np.pi * day_minutes / (24 * 60))
    print("时间特征添加完成！")

    return df

#特征数据拼接
def merge_features(density_df, weather_df, time_col="Time", freq="1m"):
    print("——————————开始合并特征数据——————————")
    density_df = density_df.copy()
    weather_df = weather_df.copy()

    merge_df = pd.merge(density_df, weather_df, on=time_col, how='inner')

    if freq == "10m":
        print("特征数据合并完成！")

    return merge_df


def process_all_data(df,area_df,weather_df):

    density_df_1m,density_df_10m = people_data_processing(df,area_df)
    weather_df_1m,weather_df_10m=weather_data_processing(weather_df)
    merge_df_1m=merge_features(density_df_1m,weather_df_1m,freq="1m")
    merge_df_10m=merge_features(density_df_10m,weather_df_10m,freq="10m")
    return merge_df_1m,merge_df_10m
    
        



if __name__ == "__main__":
    
    data_path='./data(real)/data.xlsx'
    df= pd.read_excel(data_path, sheet_name="count")
    area_df = pd.read_excel(data_path, sheet_name="detector")
    weather_df=pd.read_csv('./data(real)/weather.csv')
    
    merge_df_1m,merge_df_10m=process_all_data (df,area_df,weather_df)
    #density_df_1m,density_df_10m = people_data_processing(df,area_df)
    #weather_df_1m,weather_df_10m=weather_data_processing(weather_df)
   # merge_df_1m=merge_features("density_df_1m","weather_df_1m")
    #merge_df_10m=merge_features("density_df_10m","weather_df_10m")
    
   
    
    """density_df_1m.to_csv("./density_df_1m.csv",index=False)
    density_df_10m.to_csv("./density_10m_df.csv",index=False)
    weather_df_1m.to_csv("./weather_df_1m.csv",index=False)
    weather_df_10m.to_csv("./weather_df_10m.csv",index=False)"""
    merge_df_1m.to_csv("./data_out/merge_df_1m.csv",index=False)
    merge_df_10m.to_csv("./data_out/merge_df_10m.csv",index=False)
 
