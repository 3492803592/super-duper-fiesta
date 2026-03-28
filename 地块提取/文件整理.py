# -*- coding: utf-8 -*-
#该脚本用于将不同城市文件夹中包含 "_centerline" 关键字的 shp 文件重命名为 "城市名_路网2023" 并复制到指定输出文件夹。

import arcpy
import os
import glob
import shutil

# ==================== 配置参数 ====================
# 根文件夹路径（包含各个城市子文件夹）
root_folder = r"./out-2023"  # 请修改为实际路径

# 输出文件夹路径（重命名后的文件将保存到这里）
output_folder = r"./2023-parcel"  # 请修改为实际路径

# 需要匹配的关键字
keyword = "_parcel.shp"

# 新文件名的后缀
name_suffix = "_parcel2023"

# ==================== 开始处理 ====================

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"已创建输出文件夹: {output_folder}")

# 计数器
success_count = 0
error_count = 0

# 遍历根文件夹下的所有子文件夹
for city_folder in os.listdir(root_folder):
    city_path = os.path.join(root_folder, city_folder)
    
    # 跳过文件，只处理文件夹
    if not os.path.isdir(city_path):
        continue
    
    print(f"\n正在处理城市: {city_folder}")
    
    # 在城市文件夹中查找包含 keyword 的 shp 文件
    # 注意：shp 文件的主文件名以 .shp 结尾，但我们需要找到包含 keyword 的
    shp_files = glob.glob(os.path.join(city_path, f"*{keyword}*"))
    
    if not shp_files:
        print(f"  未找到包含 '{keyword}' 的 shp 文件，跳过")
        continue
    
    # 理论上每个城市文件夹只有一个符合条件的 shp 文件，但如果有多个也处理
    for shp_file in shp_files:
        # 获取 shp 文件所在的目录和文件名
        shp_dir = os.path.dirname(shp_file)
        shp_basename = os.path.basename(shp_file)
        shp_name_without_ext = os.path.splitext(shp_basename)[0]
        
        # 构建新文件名
        new_name = f"{city_folder}{name_suffix}"
        
        print(f"  找到文件: {shp_basename}")
        print(f"  将重命名为: {new_name}.shp")
        
        # 获取该 shp 文件的所有配套文件（.shp, .shx, .dbf, .prj, .sbn, .sbx, .xml 等）
        # shp 文件通常有多个同名但不同扩展名的文件
        file_pattern = os.path.join(shp_dir, f"{shp_name_without_ext}.*")
        companion_files = glob.glob(file_pattern)
        
        # 复制并重命名所有配套文件到输出文件夹
        for src_file in companion_files:
            src_ext = os.path.splitext(src_file)[1]
            dst_file = os.path.join(output_folder, f"{new_name}{src_ext}")
            
            try:
                shutil.copy2(src_file, dst_file)
                print(f"    已复制: {src_ext} 文件")
            except Exception as e:
                print(f"    复制失败 {src_ext}: {e}")
                error_count += 1
        
        success_count += 1

# ==================== 输出统计 ====================
print("\n" + "="*50)
print(f"处理完成！")
print(f"成功处理: {success_count} 个文件")
print(f"失败: {error_count} 个文件")
print(f"输出位置: {output_folder}")