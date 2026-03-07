# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon, Point, box
from shapely.ops import linemerge, unary_union
from scipy.ndimage import gaussian_filter1d

import arcpy
import os
import tempfile
import shutil




def smooth_line(line, sigma=0.8):
    """使用高斯滤波平滑线型几何（保留首尾点）"""
    if line.geom_type != 'LineString':
        return line

    points = np.array(line.coords)
    if len(points) < 4:  # 过短的线不处理
        return line

    # 对x/y坐标分别应用高斯滤波（保留首尾点位置不变）
    x = points[:, 0]
    y = points[:, 1]

    # 仅平滑中间点（保护起点和终点）
    x[1:-1] = gaussian_filter1d(x, sigma=sigma)[1:-1]
    y[1:-1] = gaussian_filter1d(y, sigma=sigma)[1:-1]

    return LineString(np.column_stack([x, y]))


def remove_dangling_lines_by_polygon_conversion(lines, tolerance=0.01):
    """
    使用线转面再面转线的方法去除悬挂线
    :param lines: LineString或MultiLineString几何对象
    :param tolerance: 坐标容差，用于处理浮点精度问题
    :return: 清理后的LineString或MultiLineString
    """
    if lines.is_empty or lines.geom_type not in ['LineString', 'MultiLineString']:
        return lines

    # 将线转换为面（悬挂线无法形成面，会被自动丢弃）
    try:
        # 使用多边形化操作将线转换为面
        polygon = PolygonizeResult(lines, tolerance=tolerance)

        # 如果成功生成面，再从面提取边界线
        if not polygon.is_empty:
            # 面转线 - 提取面的边界
            boundary_lines = polygon.boundary

            # 确保返回的是线类型
            if boundary_lines.geom_type in ['LineString', 'MultiLineString']:
                return boundary_lines

    except Exception as e:
        print(f"线转面面转线过程中出现错误: {e}")

    # 如果转换失败，返回原始线
    return lines


def PolygonizeResult(geometry, tolerance=0.01):
    """
    将线几何转换为面几何
    """
    if geometry.is_empty:
        return geometry

    # 如果是MultiLineString，尝试合并
    if geometry.geom_type == 'MultiLineString':
        merged = linemerge(geometry)
        if merged.geom_type == 'LineString':
            geometry = merged

    # 尝试创建面
    try:
        if geometry.geom_type == 'LineString' and not geometry.is_closed:
            # 如果线未闭合，尝试闭合它
            coords = list(geometry.coords)
            if coords[0] != coords[-1]:
                coords.append(coords[0])
                geometry = LineString(coords)

        # 创建面
        if geometry.geom_type == 'LineString' and geometry.is_closed:
            return Polygon(geometry)
        else:
            # 使用多边形化方法
            from shapely.ops import polygonize
            polygons = list(polygonize([geometry]))
            if polygons:
                if len(polygons) == 1:
                    return polygons[0]
                else:
                    return MultiPolygon(polygons)

    except Exception as e:
        print(f"多边形化失败: {e}")

    return geometry  # 返回原始几何


def morphological_operations_rectangular(geometry, buffer_distance, dilation_factor, erosion_factor):
    """
    对几何图形进行直角膨胀腐蚀操作，保持直角不变圆角
    :param geometry: 输入的几何图形
    :param buffer_distance: 原始缓冲区距离
    :param dilation_factor: 膨胀系数
    :param erosion_factor: 腐蚀系数
    :return: 处理后的几何图形
    """
    if geometry.is_empty:
        return geometry

    try:
        # 第一步：直角膨胀操作（扩大几何体，填充小空隙，保持直角）
        dilation_distance = buffer_distance * dilation_factor

        # 使用 cap_style=2 (flat) 和 join_style=2 (mitre) 来保持直角
        # cap_style=2: 平坦端点
        # join_style=2: 斜接连接，保持直角
        # mitre_limit=5.0: 斜接限制，防止过长的尖角
        dilated = geometry.buffer(
            dilation_distance,
            quad_segs=1,  # 减少线段数量，有助于保持直角
            cap_style=2,  # 平坦端点
            join_style=2,  # 斜接连接
            mitre_limit=5.0
        )

        #print(f"直角膨胀操作完成: 膨胀距离={dilation_distance:.2f}m")

        # 第二步：直角腐蚀操作（收缩回大致原始大小，保持直角）
        erosion_distance = buffer_distance * erosion_factor

        # 对膨胀后的几何体进行直角腐蚀
        eroded = dilated.buffer(
            -erosion_distance,
            quad_segs=1,  # 减少线段数量，有助于保持直角
            cap_style=2,  # 平坦端点
            join_style=2,  # 斜接连接
            mitre_limit=5.0
        )

        #print(f"直角腐蚀操作完成: 腐蚀距离={erosion_distance:.2f}m")
        print(f"形态学操作完成: 膨胀距离={dilation_distance:.2f}m, 腐蚀距离={erosion_distance:.2f}m (保持直角)")

        return eroded

    except Exception as e:
        print(f"直角形态学操作失败: {e}")
        # 如果直角操作失败，回退到原始方法
        try:
            return morphological_operations(geometry, buffer_distance, dilation_factor, erosion_factor)
        except Exception as e2:
            print(f"回退到原始方法也失败: {e2}")
            return geometry


def morphological_operations(geometry, buffer_distance, dilation_factor, erosion_factor):
    """
    对几何图形进行膨胀腐蚀操作，去除小空隙（原始方法，会产生圆角）
    """
    if geometry.is_empty:
        return geometry

    try:
        # 第一步：膨胀操作（扩大几何体，填充小空隙）
        dilation_distance = buffer_distance * dilation_factor
        dilated = geometry.buffer(dilation_distance)

        # 第二步：腐蚀操作（收缩回大致原始大小）
        erosion_distance = buffer_distance * erosion_factor
        eroded = dilated.buffer(-erosion_distance)

        print(f"形态学操作完成: 膨胀距离={dilation_distance:.2f}m, 腐蚀距离={erosion_distance:.2f}m")

        return eroded

    except Exception as e:
        print(f"形态学操作失败: {e}")
        return geometry


def simplify_polygon_keep_rectangular(polygon, tolerance=0.1):
    """
    简化多边形但尽量保持直角
    """
    if polygon.is_empty or polygon.geom_type not in ['Polygon', 'MultiPolygon']:
        return polygon

    try:
        # 使用Douglas-Peucker算法简化
        simplified = polygon.simplify(tolerance, preserve_topology=True)

        # 如果简化后的多边形仍然有效，返回简化结果
        if not simplified.is_empty:
            return simplified
        else:
            return polygon
    except Exception as e:
        print(f"多边形简化失败: {e}")
        return polygon


def save_for_arcpy_processing(geometry, output_path, crs):
    """
    专门为ArcPy处理保存几何数据，使用arcpy.management.CreateFeatureclass创建要素类
    """
    try:
        # 获取输出目录和文件名
        output_dir = os.path.dirname(output_path)
        output_name = os.path.basename(output_path)

        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 创建空间参考
        sr = arcpy.SpatialReference()

        # 根据CRS设置空间参考
        if crs is not None:
            try:
                epsg_code = crs.to_epsg()
                if epsg_code:
                    sr.factoryCode = epsg_code
                    sr.create()
                else:
                    # 使用WGS84作为默认
                    sr.factoryCode = 4326
                    sr.create()
            except:
                # 如果无法获取EPSG代码，使用Web Mercator
                sr.factoryCode = 3857
                sr.create()
        else:
            # 默认使用Web Mercator
            sr.factoryCode = 3857
            sr.create()

        print(f"创建ArcPy要素类: {output_path}")
        print(f"空间参考: {sr.name} (EPSG:{sr.factoryCode})")

        # 创建要素类
        arcpy.management.CreateFeatureclass(
            output_dir,
            output_name,
            "POLYGON" if geometry.geom_type in ['Polygon', 'MultiPolygon'] else "POLYLINE",
            spatial_reference=sr
        )

        # 添加一个ID字段
        arcpy.management.AddField(output_path, "ID", "LONG")

        # 准备几何数据
        geometries = []
        if geometry.geom_type in ['Polygon', 'LineString']:
            geometries = [geometry]
        elif geometry.geom_type in ['MultiPolygon', 'MultiLineString']:
            geometries = list(geometry.geoms)

        # 插入几何数据
        with arcpy.da.InsertCursor(output_path, ["SHAPE@", "ID"]) as cursor:
            for i, geom in enumerate(geometries, 1):
                if geom.is_empty:
                    continue

                # 创建arcpy几何对象
                if geom.geom_type == 'Polygon':
                    # 创建多边形
                    array = arcpy.Array()

                    # 外环
                    exterior_ring = arcpy.Array()
                    for coord in geom.exterior.coords:
                        exterior_ring.add(arcpy.Point(coord[0], coord[1]))
                    array.add(exterior_ring)

                    # 内环（孔洞）
                    for interior in geom.interiors:
                        interior_ring = arcpy.Array()
                        for coord in interior.coords:
                            interior_ring.add(arcpy.Point(coord[0], coord[1]))
                        array.add(interior_ring)

                    arcpy_geom = arcpy.Polygon(array, sr)

                elif geom.geom_type == 'LineString':
                    # 创建线
                    array = arcpy.Array()
                    for coord in geom.coords:
                        array.add(arcpy.Point(coord[0], coord[1]))
                    arcpy_geom = arcpy.Polyline(array, sr)

                elif geom.geom_type == 'MultiPolygon':
                    # 对于MultiPolygon中的每个多边形
                    for poly in geom.geoms:
                        if poly.is_empty:
                            continue

                        array = arcpy.Array()

                        # 外环
                        exterior_ring = arcpy.Array()
                        for coord in poly.exterior.coords:
                            exterior_ring.add(arcpy.Point(coord[0], coord[1]))
                        array.add(exterior_ring)

                        # 内环（孔洞）
                        for interior in poly.interiors:
                            interior_ring = arcpy.Array()
                            for coord in interior.coords:
                                interior_ring.add(arcpy.Point(coord[0], coord[1]))
                            array.add(interior_ring)

                        arcpy_geom = arcpy.Polygon(array, sr)
                        cursor.insertRow([arcpy_geom, i])

                    continue  # 已经插入了所有多边形，继续下一个几何

                elif geom.geom_type == 'MultiLineString':
                    # 对于MultiLineString中的每条线
                    for line in geom.geoms:
                        if line.is_empty:
                            continue

                        array = arcpy.Array()
                        for coord in line.coords:
                            array.add(arcpy.Point(coord[0], coord[1]))
                        arcpy_geom = arcpy.Polyline(array, sr)
                        cursor.insertRow([arcpy_geom, i])

                    continue  # 已经插入了所有线，继续下一个几何
                else:
                    print(f"警告: 不支持的几何类型: {geom.geom_type}")
                    continue

                # 插入当前几何
                cursor.insertRow([arcpy_geom, i])

        print(f"成功创建ArcPy要素类，包含 {len(geometries)} 个几何要素")
        return output_path

    except Exception as e:
        print(f"使用ArcPy创建要素类失败: {e}")

        # 回退到GeoPandas方法
        print("使用GeoPandas回退方法...")
        try:
            gdf = gpd.GeoDataFrame(geometry=[geometry], crs=crs)
            gdf['ID'] = 1
            gdf.to_file(output_path, encoding='utf-8')
            return output_path
        except Exception as e2:
            print(f"GeoPandas回退方法也失败: {e2}")
            return None


def extract_centerline_with_arcpy(polygon_shp, output_centerline_shp):
    """
    使用ArcPy的PolygonToCenterline工具提取中心线
    :param polygon_shp: 输入多边形要素类路径
    :param output_centerline_shp: 输出中心线要素类路径
    :return: 中心线几何对象列表或None
    """
    try:
        # 检查ArcGIS许可
        if arcpy.CheckExtension("Foundation") == "Available":
            arcpy.CheckOutExtension("Foundation")
            print("已获取Foundation扩展许可")
        else:
            print("警告: 没有Foundation扩展许可，将尝试使用其他方法")
            return None

        # 确保输出目录存在
        output_dir = os.path.dirname(output_centerline_shp)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 使用FeatureClassToFeatureClass创建干净的要素类
        temp_gdb = os.path.join(output_dir, "temp.gdb")
        if not arcpy.Exists(temp_gdb):
            arcpy.management.CreateFileGDB(output_dir, "temp.gdb")

        temp_fc = os.path.join(temp_gdb, "temp_polygon")

        print(f"转换要素类到地理数据库: {polygon_shp} -> {temp_fc}")

        # 转换要素类到地理数据库
        arcpy.conversion.FeatureClassToFeatureClass(
            polygon_shp,
            temp_gdb,
            "temp_polygon"
        )

        # 确保有OBJECTID字段（地理数据库中会自动创建）
        print("检查字段...")
        desc = arcpy.Describe(temp_fc)
        fields = [field.name for field in desc.fields]
        print(f"字段列表: {fields}")

        # 执行PolygonToCenterline工具
        print("使用ArcPy的PolygonToCenterline工具提取中心线...")
        temp_centerline_fc = os.path.join(temp_gdb, "temp_centerline")
        arcpy.topographic.PolygonToCenterline(temp_fc, temp_centerline_fc)

        # 导出到shapefile
        print(f"导出中心线到shapefile: {output_centerline_shp}")
        output_basename = os.path.basename(output_centerline_shp).replace('.shp', '')  # 去掉.shp扩展名
        arcpy.conversion.FeatureClassToFeatureClass(
            temp_centerline_fc,
            output_dir,
            output_basename
        )

        # 检查输出是否成功创建
        if arcpy.Exists(output_centerline_shp):
            print(f"成功提取中心线到: {output_centerline_shp}")

            # 读取中心线数据并转换为shapely几何对象
            centerline_gdf = gpd.read_file(output_centerline_shp)

            if len(centerline_gdf) == 0:
                print("警告: 中心线输出为空!")
                return None

            # 直接返回几何对象列表，而不是尝试合并
            lines = list(centerline_gdf.geometry)

            if lines:
                print(f"成功提取 {len(lines)} 条中心线")
                return lines  # 返回几何对象列表而不是合并的线

        else:
            print("警告: 中心线提取失败，输出文件不存在")
            return None

    except arcpy.ExecuteError as e:
        print(f"ArcPy执行错误: {e}")
        error_msgs = arcpy.GetMessages(2)
        print(f"详细错误信息: {error_msgs}")

        # 尝试使用FeatureToLine工具作为备选
        try:
            print("尝试使用FeatureToLine工具作为备选中心线提取方法...")
            temp_line = os.path.join(output_dir, "temp_line.shp")

            # 先确保多边形是有效的
            arcpy.management.RepairGeometry(polygon_shp)

            # 使用FeatureToLine
            arcpy.management.FeatureToLine(polygon_shp, temp_line)

            if arcpy.Exists(temp_line):
                # 读取并处理
                line_gdf = gpd.read_file(temp_line)
                if len(line_gdf) > 0:
                    # 获取最长的线作为中心线
                    line_gdf['length'] = line_gdf.length
                    longest_line = line_gdf.loc[line_gdf['length'].idxmax()].geometry
                    print("已使用FeatureToLine工具提取中心线")

                    # 清理临时文件
                    try:
                        base_name = os.path.splitext(temp_line)[0]
                        for ext in ['.shp', '.dbf', '.shx', '.prj', '.cpg', '.sbn', '.sbx']:
                            related_file = base_name + ext
                            if os.path.exists(related_file):
                                os.remove(related_file)
                    except:
                        pass

                    return [longest_line]  # 返回列表
        except Exception as e2:
            print(f"备选方法也失败: {e2}")

        return None

    except Exception as e:
        print(f"提取中心线时发生错误: {e}")
        return None
    finally:
        # 清理临时地理数据库
        try:
            if 'temp_gdb' in locals() and arcpy.Exists(temp_gdb):
                arcpy.management.Delete(temp_gdb)
                print("已清理临时地理数据库")
        except Exception as e:
            print(f"清理临时地理数据库时出错: {e}")
            pass

        # 释放扩展许可
        try:
            arcpy.CheckInExtension("Foundation")
            print("已释放Foundation扩展许可")
        except:
            pass


def flatten_geometry_list(geometries):
    """
    展平几何对象列表，将MultiLineString和MultiPolygon拆分为单个几何对象
    """
    flattened = []
    for geom in geometries:
        if geom is None or geom.is_empty:
            continue
        elif geom.geom_type in ['MultiLineString', 'MultiPolygon']:
            flattened.extend(list(geom.geoms))
        else:
            flattened.append(geom)
    return flattened


def create_rectangular_buffer(geometry, distance, join_style=2, cap_style=2):
    """
    创建直角缓冲区
    """
    if geometry.is_empty:
        return geometry

    try:
        # 使用join_style=2 (mitre) 和 cap_style=2 (flat) 来创建直角缓冲区
        buffered = geometry.buffer(
            distance,
            quad_segs=1,  # 减少线段数量，有助于保持直角
            cap_style=cap_style,
            join_style=join_style,
            mitre_limit=5.0  # 斜接限制
        )
        return buffered
    except Exception as e:
        print(f"创建直角缓冲区失败: {e}")
        # 回退到普通缓冲区
        return geometry.buffer(distance)


def generate_parcels_with_fixed_params(input_shp, output_path_buffer, output_path_centerline=None,
                                       admin_boundary_path=None, buffer_distance=20, buffer_distance2=20,
                                       use_morphological_ops=True, dilation_factor=1.2, erosion_factor=1.2,
                                       output_path_dissolved_buffer_after_morph=None, use_rectangular_morph=True,output_path_precise_buffer=None):
    """
    使用固定参数生成地块，使用ArcPy提取中心线但使用GeoPandas创建缓冲区

    :param input_shp: 输入道路数据路径
    :param output_path_buffer: 输出地块数据路径
    :param output_path_centerline: 输出中心线数据路径
    :param admin_boundary_path: 行政区划边界路径
    :param buffer_distance: 缓冲区距离
    :param output_path_dissolved_buffer_after_morph: 输出形态学操作后融合缓冲区路径（新增参数）
    :param use_morphological_ops: 是否使用形态学操作
    :param dilation_factor: 膨胀系数
    :param erosion_factor: 腐蚀系数
    :param use_rectangular_morph: 是否使用直角形态学操作（新增参数）
    """

    # 1. 读取原始矢量数据
    gdf = gpd.read_file(input_shp)
    print("已读取矢量数据")
    
    # OSM数据通常用 fclass 字段区分等级
    # 保留三级及以上道路
    keep_classes = [
        'motorway', 'motorway_link', 
        'trunk', 'trunk_link', 
        'primary', 'primary_link', 
        'secondary', 'secondary_link',
        'tertiary', 'tertiary_link'
    ]

    if 'fclass' in gdf.columns:
        initial_count = len(gdf)
        # 筛选在列表中的道路级别
        gdf = gdf[gdf['fclass'].isin(keep_classes)]
        print(f"道路等级过滤完成：保留了 {keep_classes}")
        print(f"要素数量从 {initial_count} 减少至 {len(gdf)}")
    else:
        # 如果不是OSM标准数据，打印属性表列名帮你排查
        print(f"警告：未找到 'fclass' 字段。当前字段有: {gdf.columns.tolist()}")
    
    # 保存原始属性数据用于后续空间连接
    original_roads_gdf = gdf.copy()

    # 新增步骤：使用行政区划进行裁剪（如果提供了行政区划路径）
    if admin_boundary_path:
        try:
            # 读取行政区划数据
            admin_gdf = gpd.read_file(admin_boundary_path)
            print("已读取行政区划数据")

            # 确保两个数据集的坐标系一致
            if gdf.crs != admin_gdf.crs:
                admin_gdf = admin_gdf.to_crs(gdf.crs)
                print("已将行政区划数据转换为与输入数据相同的坐标系")

            # 进行裁剪操作
            gdf = gpd.clip(gdf, admin_gdf)
            print("已完成行政区划裁剪")

            # 检查裁剪后的数据是否为空
            if len(gdf) == 0:
                print("警告: 裁剪后数据为空，请检查行政区划范围!")
                return

        except Exception as e:
            print(f"行政区划裁剪失败: {e}")
            print("将继续使用原始数据进行处理")
    else:
        print("未提供行政区划路径，将使用完整数据进行处理")

    # 2. 保持Web Mercator投影(EPSG:3857)
    target_crs = 'EPSG:3857'
    original_crs = gdf.crs
    gdf = gdf.to_crs(target_crs)
    original_roads_gdf = original_roads_gdf.to_crs(target_crs)
    print(f"已投影到Web Mercator(EPSG:3857)")

    # 3. 使用GeoPandas的buffer方法创建初始缓冲区并融合
    print("步骤3: 使用GeoPandas的buffer方法创建并融合缓冲区...")

    # 使用直角缓冲区创建初始缓冲区
    if use_rectangular_morph:
        print("使用直角缓冲区创建初始缓冲区...")
        gdf['geometry'] = gdf.geometry.apply(
            lambda geom: create_rectangular_buffer(geom, buffer_distance, join_style=2, cap_style=2)
        )
    else:
        gdf['geometry'] = gdf.buffer(buffer_distance, join_style=1)

    dissolved = gdf.dissolve()
    polygon = dissolved.geometry.values[0]
    print("已使用GeoPandas生成并融合缓冲区")

    # 保存形态学操作前的多边形用于后续比较
    polygon_before_morph = polygon

    # 4. 对第一次融合缓冲区进行膨胀腐蚀操作
    if use_morphological_ops:
        print("步骤4: 对第一次融合缓冲区进行形态学操作（膨胀腐蚀）...")

        if use_rectangular_morph:
            print("使用直角形态学操作...")
            polygon = morphological_operations_rectangular(
                polygon,
                buffer_distance,
                dilation_factor=dilation_factor,
                erosion_factor=erosion_factor
            )
        else:
            print("使用普通形态学操作...")
            polygon = morphological_operations(
                polygon,
                buffer_distance,
                dilation_factor=dilation_factor,
                erosion_factor=erosion_factor
            )

        print("已完成第一次融合缓冲区的形态学操作")

        # 在形态学操作后保存融合缓冲区到文件
        if output_path_dissolved_buffer_after_morph:
            try:
                # 创建形态学操作后的融合缓冲区的GeoDataFrame
                dissolved_gdf_after_morph = gpd.GeoDataFrame(geometry=[polygon], crs=target_crs)

                # 添加一些有用的属性信息
                dissolved_gdf_after_morph['area_m2'] = dissolved_gdf_after_morph.area
                dissolved_gdf_after_morph['perimeter_m'] = dissolved_gdf_after_morph.length
                dissolved_gdf_after_morph['buffer_distance'] = buffer_distance
                dissolved_gdf_after_morph['dilation_factor'] = dilation_factor
                dissolved_gdf_after_morph['erosion_factor'] = erosion_factor
                dissolved_gdf_after_morph['operation'] = 'after_morphological'
                dissolved_gdf_after_morph['dilation_dist'] = buffer_distance * dilation_factor
                dissolved_gdf_after_morph['erosion_dist'] = buffer_distance * erosion_factor
                dissolved_gdf_after_morph['rectangular'] = use_rectangular_morph

                # 计算形态学操作前后的面积变化
                area_before = polygon_before_morph.area
                area_after = polygon.area
                area_change = ((area_after - area_before) / area_before) * 100
                dissolved_gdf_after_morph['area_before_m2'] = area_before
                dissolved_gdf_after_morph['area_change_percent'] = area_change

                # 保存到文件
                dissolved_gdf_after_morph.to_file(output_path_dissolved_buffer_after_morph, encoding='utf-8')
                print(f"已保存形态学操作后的融合缓冲区到: {output_path_dissolved_buffer_after_morph}")
                #print(f"形态学操作前缓冲区面积: {area_before:.2f} 平方米")
                #print(f"形态学操作后缓冲区面积: {area_after:.2f} 平方米")
                #print(f"面积变化: {area_change:.2f}%")
            except Exception as e:
                print(f"保存形态学操作后融合缓冲区失败: {e}")

    # 5. 使用ArcPy的PolygonToCenterline工具提取中心线
    centerline_geoms = None  # 修改为geoms
    temp_dir_obj = None

    try:
        print("步骤5: 使用ArcPy的PolygonToCenterline工具提取中心线...")

        # 创建临时目录
        temp_dir_obj = tempfile.mkdtemp()
        temp_polygon_shp = os.path.join(temp_dir_obj, "temppoly.shp")
        temp_centerline_shp = os.path.join(temp_dir_obj, "tempcenterline.shp")

        # 使用专门的函数保存多边形，确保字段正确
        saved_path = save_for_arcpy_processing(polygon, temp_polygon_shp, target_crs)

        if saved_path is None:
            print("警告: 无法为ArcPy处理保存多边形!")
            raise Exception("无法保存多边形用于ArcPy处理")

        print(f"已保存临时多边形文件: {temp_polygon_shp}")

        # 使用ArcPy提取中心线
        centerline_geoms = extract_centerline_with_arcpy(temp_polygon_shp, temp_centerline_shp)

        if centerline_geoms is None or len(centerline_geoms) == 0:
            print("警告: ArcPy未能提取到中心线!")

            # 尝试使用多边形边界简化作为中心线
            print("尝试使用多边形边界简化作为备选中心线...")
            # 获取多边形边界
            boundary = polygon.boundary

            if boundary.geom_type == 'LineString':
                centerline_geoms = [boundary]
            elif boundary.geom_type == 'MultiLineString':
                # 直接获取所有线
                centerline_geoms = list(boundary.geoms)

            print("已使用多边形边界作为备选中心线")
        else:
            print(f"成功使用ArcPy提取中心线，共 {len(centerline_geoms)} 条")

    except Exception as e:
        print(f"中心线提取失败: {e}")
        # 使用多边形边界作为最后的备选方案
        try:
            print("使用多边形边界作为中心线...")
            boundary = polygon.boundary
            if boundary.geom_type == 'LineString':
                centerline_geoms = [boundary]
            elif boundary.geom_type == 'MultiLineString':
                centerline_geoms = list(boundary.geoms)
            else:
                print("无法提取中心线，退出处理")
                if temp_dir_obj and os.path.exists(temp_dir_obj):
                    shutil.rmtree(temp_dir_obj, ignore_errors=True)
                return
        except Exception as e2:
            print(f"所有中心线提取方法都失败: {e2}")
            if temp_dir_obj and os.path.exists(temp_dir_obj):
                shutil.rmtree(temp_dir_obj, ignore_errors=True)
            return
    finally:
        # 清理临时目录
        if temp_dir_obj and os.path.exists(temp_dir_obj):
            try:
                shutil.rmtree(temp_dir_obj, ignore_errors=True)
                print("已清理临时目录")
            except:
                pass

    # 处理中心线结果
    if centerline_geoms is None or len(centerline_geoms) == 0:
        print("警告: 未生成任何中心线!")
        return

    # 展平几何对象列表
    centerline_geoms = flatten_geometry_list(centerline_geoms)
    print(f"展平后共 {len(centerline_geoms)} 条线段")

    # 6. 处理中心线（平滑+使用线转面面转线方法去除悬挂线）
    # 直接使用几何对象列表，不需要linemerge
    lines = centerline_geoms

    # 使用线转面面转线方法去除悬挂线
    # 创建MultiLineString来处理多条线
    if len(lines) > 1:
        # 确保所有几何对象都是LineString
        line_strings = []
        for line in lines:
            if line.geom_type == 'LineString':
                line_strings.append(line)
            elif line.geom_type == 'MultiLineString':
                line_strings.extend(list(line.geoms))

        if line_strings:
            multi_line = MultiLineString(line_strings)
            cleaned_line = remove_dangling_lines_by_polygon_conversion(multi_line, tolerance=0.01)
        else:
            print("警告: 没有有效的LineString几何对象!")
            return
    else:
        # 处理单个几何对象
        if lines[0].geom_type == 'LineString':
            cleaned_line = remove_dangling_lines_by_polygon_conversion(lines[0], tolerance=0.01)
        elif lines[0].geom_type == 'MultiLineString':
            # 将MultiLineString转换为单个LineString列表
            line_strings = list(lines[0].geoms)
            if len(line_strings) == 1:
                cleaned_line = remove_dangling_lines_by_polygon_conversion(line_strings[0], tolerance=0.01)
            else:
                multi_line = MultiLineString(line_strings)
                cleaned_line = remove_dangling_lines_by_polygon_conversion(multi_line, tolerance=0.01)
        else:
            print(f"警告: 不支持的几何类型: {lines[0].geom_type}")
            return

    if cleaned_line is None or cleaned_line.is_empty:
        print("警告: 所有线段都被识别为悬挂线!")
        return

    # 7. 创建精确的道路缓冲区（基于处理后的中心线）- 使用GeoPandas
    print("步骤7: 创建精确的道路缓冲区（使用GeoPandas）...")

    # 处理清理后的线几何
    if cleaned_line.geom_type == 'MultiLineString':
        centerlines = [smooth_line(line) for line in cleaned_line.geoms]
        # 使用Geopandas创建精确道路缓冲区
        if use_rectangular_morph:
            road_buffers = [create_rectangular_buffer(line, buffer_distance2, join_style=2, cap_style=2)
                            for line in centerlines]
        else:
            road_buffers = [line.buffer(buffer_distance2, join_style=1) for line in centerlines]
    elif cleaned_line.geom_type == 'LineString':
        centerline = smooth_line(cleaned_line)
        # 使用Geopandas创建缓冲区
        if use_rectangular_morph:
            road_buffers = [create_rectangular_buffer(centerline, buffer_distance2, join_style=2, cap_style=2)]
        else:
            road_buffers = [centerline.buffer(buffer_distance2, join_style=1)]
    else:
        # 如果是几何对象列表
        centerlines = []
        for line in lines:
            if line.geom_type == 'LineString':
                centerlines.append(smooth_line(line))
            elif line.geom_type == 'MultiLineString':
                centerlines.extend([smooth_line(l) for l in line.geoms])

        if use_rectangular_morph:
            road_buffers = [create_rectangular_buffer(line, buffer_distance2, join_style=2, cap_style=2)
                            for line in centerlines]
        else:
            road_buffers = [line.buffer(buffer_distance2, join_style=1) for line in centerlines]

    merged_road_buffer = unary_union(road_buffers)
    print("已创建精确道路缓冲区")
    # =========== 保存第七步精确缓冲区结果 ===========
    try:
        # 使用传入的输出路径，如果没有传入则使用默认命名规则
        if output_path_precise_buffer:
            precise_buffer_path = output_path_precise_buffer
        else:
            # 如果没有指定精确缓冲区输出路径，则使用默认命名规则
            buffer_dir = os.path.dirname(output_path_buffer)
            buffer_name = os.path.basename(output_path_buffer)
            name_without_ext = os.path.splitext(buffer_name)[0]
            precise_buffer_path = os.path.join(buffer_dir, f"{name_without_ext}_precise_buffer.shp")

        # 创建精确缓冲区的GeoDataFrame
        if merged_road_buffer.geom_type == 'MultiPolygon':
            buffer_geoms = list(merged_road_buffer.geoms)
        elif merged_road_buffer.geom_type == 'Polygon':
            buffer_geoms = [merged_road_buffer]
        else:
            buffer_geoms = [merged_road_buffer]

        precise_buffer_gdf = gpd.GeoDataFrame(geometry=buffer_geoms, crs=target_crs)

        # 添加属性信息
        precise_buffer_gdf['area_m2'] = precise_buffer_gdf.area
        precise_buffer_gdf['perimeter_m'] = precise_buffer_gdf.length
        precise_buffer_gdf['buffer_distance'] = buffer_distance2
        precise_buffer_gdf['rectangular'] = use_rectangular_morph
        precise_buffer_gdf['geometry_type'] = precise_buffer_gdf.geometry.type

        # 保存到文件
        precise_buffer_gdf.to_file(precise_buffer_path, encoding='utf-8')
        print(f"已保存第七步精确缓冲区结果到: {precise_buffer_path}")
        print(f"精确缓冲区包含 {len(precise_buffer_gdf)} 个要素，总面积: {precise_buffer_gdf.area.sum():.2f} 平方米")

    except Exception as e:
        print(f"保存第七步精确缓冲区结果失败: {e}")

    # 8. 创建研究区域边界（使用原始缓冲区的凸包）
    study_area =polygon.convex_hull
    print("已创建研究区域边界")

    # 9. 计算地块区域（研究区域减去道路缓冲区）
    parcels = study_area.difference(merged_road_buffer)
    print("已计算地块区域")

    # 10. 处理并保存结果
    if parcels.is_empty:
        print("警告: 生成的地块区域为空!")
        return

    # 确保几何类型正确
    if parcels.geom_type == 'Polygon':
        parcels = [parcels]
    elif parcels.geom_type == 'MultiPolygon':
        parcels = list(parcels.geoms)
    else:
        print(f"警告: 意外的几何类型: {parcels.geom_type}")
        return

    # 步骤1：删除面积最大的面
    if len(parcels) > 1:
        # 计算每个面的面积并找到最大面积的索引
        areas = [p.area for p in parcels]
        max_area_idx = np.argmax(areas)

        # 移除面积最大的面
        parcels = [p for i, p in enumerate(parcels) if i != max_area_idx]
        #print("已移除面积最大的地块")

    # 步骤2：去除面积小于10000平方米的面
    min_area = 10000  # 最小面积阈值（平方米）
    parcels = [p for p in parcels if p.area >= min_area]
    print(f"已去除面积小于{min_area}平方米的地块，剩余{len(parcels)}个地块")

    # 检查是否还有剩余地块
    if not parcels:
        print("警告: 所有地块都被过滤掉了!")
        return

    # 创建GeoDataFrame并保存
    parcel_gdf = gpd.GeoDataFrame(geometry=parcels, crs=target_crs)
    parcel_gdf.to_file(output_path_buffer, encoding='utf-8')
    print(f"已保存 {len(parcels)} 个地块要素到: {output_path_buffer}")

    # 11. 新增：将原始OSM数据与处理后的中心线进行空间连接
    if output_path_centerline:
        try:
            print("步骤11: 开始空间连接，为处理后的中心线赋予原始属性...")

            # 创建中心线的GeoDataFrame
            centerline_geoms_for_attr = []
            if cleaned_line.geom_type == 'MultiLineString':
                centerline_geoms_for_attr = list(cleaned_line.geoms)
            elif cleaned_line.geom_type == 'LineString':
                centerline_geoms_for_attr = [cleaned_line]
            else:
                # 展平所有几何对象
                centerline_geoms_for_attr = []
                for geom in lines:
                    if geom.geom_type == 'LineString':
                        centerline_geoms_for_attr.append(geom)
                    elif geom.geom_type == 'MultiLineString':
                        centerline_geoms_for_attr.extend(list(geom.geoms))

            centerline_gdf = gpd.GeoDataFrame(geometry=centerline_geoms_for_attr, crs=target_crs)

            # 进行空间连接 - 使用最近邻连接
            centerline_with_attrs = gpd.sjoin_nearest(
                centerline_gdf,
                original_roads_gdf,
                how='left',
                distance_col='distance_to_original'
            )

            # 去除重复的几何图形（如果有）
            centerline_with_attrs = centerline_with_attrs.drop_duplicates(subset=['geometry'])

            # 保存带有属性的中心线数据
            centerline_with_attrs.to_file(output_path_centerline, encoding='utf-8')
            print(f"已保存带有属性的中心线数据到: {output_path_centerline}")
            print(f"中心线数据包含 {len(centerline_with_attrs)} 条线段")

        except Exception as e:
            print(f"空间连接过程中出现错误: {e}")
            print("将继续执行，但不输出带属性的中心线数据")


    # 清理临时文件
    try:
        import glob
        temp_dir = tempfile.gettempdir()
        # 清理其他临时文件
        for pattern in ["temp_centerline_*.shp", "temp_buffer_*.shp"]:
            for temp_file in glob.glob(os.path.join(temp_dir, pattern)):
                base_name = os.path.splitext(temp_file)[0]
                for ext in ['.shp', '.dbf', '.shx', '.prj', '.cpg', '.sbn', '.sbx']:
                    related_file = base_name + ext
                    if os.path.exists(related_file):
                        try:
                            os.remove(related_file)
                        except:
                            pass

        print("已清理临时文件")
    except Exception as e:
        print(f"清理临时文件时出错: {e}")


# 示例调用
if __name__ == "__main__":
    arcpy.env.overwriteOutput = True
    input_shp = r"./hunan-260107-free.shp/gis_osm_roads_free_1.shp"#输入OSM路网数据
    admin_boundary_path = r"./行政边界/长沙市/长沙市.shp"  # 输入研究区域矢量数据
    output_buffer = r"./out/changsha/changsha_parcel2.shp"#输出社区地块面要素
    output_centerline = r"./out/changsha/changsha_centerline3.shp"#输出道路中心线
    output_dissolved_buffer_after_morph = r"./out/changsha/单线双线结果图/changsha_dissolved_buffer.shp"
    output_precise_buffer = r"./out/changsha/单线双线结果图/changsha_buffer.shp"#输出道路双线缓冲区


    # 使用固定参数调用，启用直角形态学操作
    generate_parcels_with_fixed_params(
        input_shp=input_shp,
        output_path_buffer=output_buffer,
        output_path_centerline=output_centerline,
        output_path_dissolved_buffer_after_morph=output_dissolved_buffer_after_morph,
        output_path_precise_buffer=output_precise_buffer,
        admin_boundary_path=admin_boundary_path,
        buffer_distance=20,
        buffer_distance2=20,
        use_morphological_ops=True,
        dilation_factor=1.2,
        erosion_factor=1.2,
        use_rectangular_morph=True  # 启用直角形态学操作
    )