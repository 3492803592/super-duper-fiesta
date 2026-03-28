import arcpy

# 设置工作空间(可选)
#arcpy.env.workspace = r"D:/提取地块"
#arcpy.env.overwriteOutput = True

# 输入文件
input_shp = "./out-2023/成都/chengdu_centerline.shp"

# 步骤1: 合并所有线要素为一个要素并输出
merged_output_shp = "./out-2023/成都/merged_chengdu.shp"  # 合并后的输出文件
arcpy.management.Dissolve(
    in_features=input_shp,
    out_feature_class=merged_output_shp,
    dissolve_field=[],  # 空列表,合并所有要素
    multi_part="MULTI_PART"
)

print(f"步骤1完成: 已合并所有线要素,输出文件: {merged_output_shp}")

# 步骤2: 在交点处打断并输出
split_output_shp = "./out-2023/成都/split_chengdu.shp"  # 打断后的输出文件
arcpy.management.FeatureToLine(
    in_features=merged_output_shp,
    out_feature_class=split_output_shp,
    cluster_tolerance="",
    attributes="NO_ATTRIBUTES"  # 或 "ATTRIBUTES" 如果需要保留属性
)

print(f"步骤2完成: 已在交点处打断,输出文件: {split_output_shp}")
print("\n最终输出两个文件:")
print(f"1. 合并后的文件: {merged_output_shp}")
print(f"2. 打断后的文件: {split_output_shp}")