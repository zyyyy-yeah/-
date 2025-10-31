import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import warnings
import time
import os
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
from shapely.geometry import Point, Polygon
from folium.plugins import HeatMap
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

warnings.filterwarnings('ignore')

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def save_map_as_image(html_file, output_image):
    """
    使用Selenium将HTML地图保存为图片
    """
    # 设置Chrome选项
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # 无界面模式
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")  # 设置窗口大小
    
    try:
        # 启动浏览器
        driver = webdriver.Chrome(options=chrome_options)
        
        # 打开本地HTML文件
        driver.get(f"file:///{os.path.abspath(html_file)}")
        
        # 等待地图加载完成
        time.sleep(5)
        
        # 截图保存
        driver.save_screenshot(output_image)
        print(f"✓ 地图已保存为: {output_image}")
        
    except Exception as e:
        print(f"截图失败: {e}")
    finally:
        driver.quit()


def log_progress(message):
    """进度报告函数"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def load_and_preprocess_bike_data():
    """
    加载和预处理共享单车数据
    """
    try:
        log_progress("开始加载共享单车数据...")
        
        # 使用您的确切文件路径
        file_path = r"E:/10-大三上/1-学习/1-信管/3-大数据系统原理与应用/期中作业/bike.csv"
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            log_progress(f"错误：找不到文件 {file_path}")
            log_progress("请检查文件路径是否正确")
            return None
        
        log_progress(f"正在从 {file_path} 读取数据...")
        
        # 首先读取前几行来查看数据结构
        try:
            sample_data = pd.read_csv(file_path, nrows=5)
            log_progress("文件列名: " + ", ".join(sample_data.columns.tolist()))
        except Exception as e:
            log_progress(f"读取文件样本时出错: {e}")
            return None
        
        # 尝试不同的编码方式
        encodings = ['utf-8', 'gbk', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                log_progress(f"尝试使用 {encoding} 编码读取...")
                bike_data = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                log_progress(f"成功使用 {encoding} 编码读取数据")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                log_progress(f"{encoding} 编码读取失败: {e}")
                continue
        else:
            log_progress("所有编码方式都失败了，尝试默认读取...")
            try:
                bike_data = pd.read_csv(file_path, low_memory=False)
            except Exception as e:
                log_progress(f"最终读取失败: {e}")
                return None
        
        log_progress(f"原始数据形状: {bike_data.shape}")
        log_progress(f"数据列: {list(bike_data.columns)}")
        
        # 数据清洗
        initial_count = len(bike_data)
        log_progress(f"初始数据量: {initial_count:,} 行")
        
        # 检查必要的列是否存在
        required_columns = ['start_lat', 'start_lng', 'end_lat', 'end_lng']
        missing_columns = [col for col in required_columns if col not in bike_data.columns]
        if missing_columns:
            log_progress(f"警告：缺少必要的列: {missing_columns}")
            log_progress("可用的列: " + ", ".join(bike_data.columns.tolist()))
            return None
        
        # 移除坐标缺失的行
        before_missing = len(bike_data)
        bike_data = bike_data.dropna(subset=['start_lat', 'start_lng', 'end_lat', 'end_lng'])
        after_missing = len(bike_data)
        log_progress(f"移除坐标缺失行: {before_missing} -> {after_missing}")
        
        # NYC坐标范围过滤
        before_coord = len(bike_data)
        bike_data = bike_data[
            (bike_data['start_lat'].between(40.4, 41.0)) & 
            (bike_data['start_lng'].between(-74.3, -73.6)) &
            (bike_data['end_lat'].between(40.4, 41.0)) & 
            (bike_data['end_lng'].between(-74.3, -73.6))
        ]
        after_coord = len(bike_data)
        log_progress(f"坐标范围过滤: {before_coord} -> {after_coord}")
        
        # 如果有duration_minutes和distance_km列，进行过滤
        if 'duration_minutes' in bike_data.columns and 'distance_km' in bike_data.columns:
            before_filter = len(bike_data)
            bike_data = bike_data[
                (bike_data['duration_minutes'].between(1, 180)) & 
                (bike_data['distance_km'].between(0.1, 20))
            ]
            after_filter = len(bike_data)
            log_progress(f"时长距离过滤: {before_filter} -> {after_filter}")
        
        # 检查是否有时间列，用于区分早晚高峰
        time_columns = ['started_at', 'start_time', 'starttime']
        time_column = None
        for col in time_columns:
            if col in bike_data.columns:
                time_column = col
                break
        
        if time_column:
            log_progress(f"找到时间列: {time_column}")
            # 转换时间格式
            try:
                bike_data[time_column] = pd.to_datetime(bike_data[time_column])
                bike_data['hour'] = bike_data[time_column].dt.hour
                bike_data['is_morning_peak'] = bike_data['hour'].between(7, 10)  # 早高峰 7-10点
                bike_data['is_evening_peak'] = bike_data['hour'].between(17, 20)  # 晚高峰 17-20点
                log_progress("时间数据处理完成")
            except Exception as e:
                log_progress(f"时间数据处理错误: {e}")
        
        final_percent = len(bike_data) / initial_count * 100
        log_progress(f"数据清洗完成: {len(bike_data)} 行 ({final_percent:.1f}% 保留)")
        
        return bike_data
        
    except Exception as e:
        log_progress(f"数据加载错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def identify_commuting_corridors(bike_data):
    """
    识别主要通勤走廊
    """
    try:
        log_progress("正在识别通勤走廊...")
        
        # 按起点-终点对分组
        od_flows = bike_data.groupby([
            'start_station_id', 'start_station_name', 
            'end_station_id', 'end_station_name',
            'start_lat', 'start_lng', 'end_lat', 'end_lng'
        ]).size().reset_index(name='trip_count')
        
        # 过滤重要走廊（至少2次出行）
        major_corridors = od_flows[od_flows['trip_count'] >= 2].copy()
        
        # 如果有距离信息，计算流量强度
        if 'distance_km' in bike_data.columns:
            distance_info = bike_data.groupby(['start_station_id', 'end_station_id'])['distance_km'].mean().reset_index()
            major_corridors = pd.merge(major_corridors, distance_info, 
                                     on=['start_station_id', 'end_station_id'], how='left')
            major_corridors['flow_intensity'] = major_corridors['trip_count'] * major_corridors['distance_km']
        else:
            major_corridors['flow_intensity'] = major_corridors['trip_count']
        
        # 按重要性排序
        major_corridors = major_corridors.sort_values('flow_intensity', ascending=False)
        
        log_progress(f"识别出 {len(major_corridors)} 个主要通勤走廊")
        
        return major_corridors
        
    except Exception as e:
        log_progress(f"通勤走廊识别错误: {str(e)}")
        return None

def get_station_importance(bike_data, major_corridors):
    """
    计算站点重要性
    """
    try:
        log_progress("正在计算站点重要性...")
        
        # 计算每个站点的出发和到达次数
        start_stations = bike_data.groupby(['start_station_id', 'start_station_name', 'start_lat', 'start_lng']).size().reset_index(name='departures')
        end_stations = bike_data.groupby(['end_station_id', 'end_station_name', 'end_lat', 'end_lng']).size().reset_index(name='arrivals')
        
        # 合并出发和到达数据
        station_importance = pd.merge(
            start_stations, end_stations,
            left_on=['start_station_id', 'start_station_name', 'start_lat', 'start_lng'],
            right_on=['end_station_id', 'end_station_name', 'end_lat', 'end_lng'],
            how='outer'
        )
        
        # 清理列名
        station_importance = station_importance.rename(columns={
            'start_station_id': 'station_id',
            'start_station_name': 'station_name',
            'start_lat': 'lat',
            'start_lng': 'lng'
        })
        
        # 移除不需要的列
        if 'end_station_id' in station_importance.columns:
            station_importance = station_importance.drop(['end_station_id', 'end_station_name', 'end_lat', 'end_lng'], axis=1)
        
        # 处理NaN值
        station_importance['departures'] = station_importance['departures'].fillna(0)
        station_importance['arrivals'] = station_importance['arrivals'].fillna(0)
        
        # 计算总流量
        station_importance['total_flow'] = station_importance['departures'] + station_importance['arrivals']
        
        # 按总流量排序
        station_importance = station_importance.sort_values('total_flow', ascending=False)
        
        log_progress(f"计算了 {len(station_importance)} 个站点的重要性")
        
        return station_importance
        
    except Exception as e:
        log_progress(f"站点重要性计算错误: {str(e)}")
        return None

def identify_residential_work_areas(bike_data):
    """
    识别居住区域和工作区域
    基于早高峰和晚高峰的出行模式
    """
    try:
        log_progress("正在识别居住区域和工作区域...")
        
        # 检查是否有时间数据
        if 'is_morning_peak' not in bike_data.columns or 'is_evening_peak' not in bike_data.columns:
            log_progress("警告：缺少时间数据，无法准确识别居住和工作区域")
            return None, None
        
        # 早高峰：起点可能是居住区，终点可能是工作区
        morning_trips = bike_data[bike_data['is_morning_peak']]
        
        # 晚高峰：起点可能是工作区，终点可能是居住区
        evening_trips = bike_data[bike_data['is_evening_peak']]
        
        # 居住区域：早高峰出发地 + 晚高峰到达地
        residential_starts = morning_trips.groupby(['start_station_id', 'start_station_name', 'start_lat', 'start_lng']).size().reset_index(name='morning_departures')
        residential_ends = evening_trips.groupby(['end_station_id', 'end_station_name', 'end_lat', 'end_lng']).size().reset_index(name='evening_arrivals')
        
        # 合并居住区域数据
        residential_areas = pd.merge(
            residential_starts, residential_ends,
            left_on=['start_station_id', 'start_station_name', 'start_lat', 'start_lng'],
            right_on=['end_station_id', 'end_station_name', 'end_lat', 'end_lng'],
            how='outer'
        )
        
        # 清理列名
        residential_areas = residential_areas.rename(columns={
            'start_station_id': 'station_id',
            'start_station_name': 'station_name',
            'start_lat': 'lat',
            'start_lng': 'lng'
        })
        
        if 'end_station_id' in residential_areas.columns:
            residential_areas = residential_areas.drop(['end_station_id', 'end_station_name', 'end_lat', 'end_lng'], axis=1)
        
        # 处理NaN值
        residential_areas['morning_departures'] = residential_areas['morning_departures'].fillna(0)
        residential_areas['evening_arrivals'] = residential_areas['evening_arrivals'].fillna(0)
        
        # 计算居住指数
        residential_areas['residential_index'] = residential_areas['morning_departures'] + residential_areas['evening_arrivals']
        
        # 工作区域：早高峰到达地 + 晚高峰出发地
        work_starts = evening_trips.groupby(['start_station_id', 'start_station_name', 'start_lat', 'start_lng']).size().reset_index(name='evening_departures')
        work_ends = morning_trips.groupby(['end_station_id', 'end_station_name', 'end_lat', 'end_lng']).size().reset_index(name='morning_arrivals')
        
        # 合并工作区域数据
        work_areas = pd.merge(
            work_starts, work_ends,
            left_on=['start_station_id', 'start_station_name', 'start_lat', 'start_lng'],
            right_on=['end_station_id', 'end_station_name', 'end_lat', 'end_lng'],
            how='outer'
        )
        
        # 清理列名
        work_areas = work_areas.rename(columns={
            'start_station_id': 'station_id',
            'start_station_name': 'station_name',
            'start_lat': 'lat',
            'start_lng': 'lng'
        })
        
        if 'end_station_id' in work_areas.columns:
            work_areas = work_areas.drop(['end_station_id', 'end_station_name', 'end_lat', 'end_lng'], axis=1)
        
        # 处理NaN值
        work_areas['evening_departures'] = work_areas['evening_departures'].fillna(0)
        work_areas['morning_arrivals'] = work_areas['morning_arrivals'].fillna(0)
        
        # 计算工作指数
        work_areas['work_index'] = work_areas['evening_departures'] + work_areas['morning_arrivals']
        
        # 按指数排序
        residential_areas = residential_areas.sort_values('residential_index', ascending=False)
        work_areas = work_areas.sort_values('work_index', ascending=False)
        
        log_progress(f"识别出 {len(residential_areas)} 个居住区域和 {len(work_areas)} 个工作区域")
        
        return residential_areas, work_areas
        
    except Exception as e:
        log_progress(f"居住工作区域识别错误: {str(e)}")
        return None, None

def create_cluster_areas(bike_data, area_type="residential"):
    """
    使用聚类算法识别区域集群
    """
    try:
        log_progress(f"正在使用聚类算法识别{area_type}区域...")
        
        if area_type == "residential":
            # 居住区域：早高峰出发 + 晚高峰到达
            morning_departures = bike_data[bike_data['is_morning_peak']].groupby(
                ['start_lat', 'start_lng']).size().reset_index(name='count')
            evening_arrivals = bike_data[bike_data['is_evening_peak']].groupby(
                ['end_lat', 'end_lng']).size().reset_index(name='count')
            
            # 合并数据
            residential_points = pd.merge(
                morning_departures, evening_arrivals,
                left_on=['start_lat', 'start_lng'],
                right_on=['end_lat', 'end_lng'],
                how='outer'
            )
            
            # 清理数据
            residential_points['lat'] = residential_points['start_lat'].combine_first(residential_points['end_lat'])
            residential_points['lng'] = residential_points['start_lng'].combine_first(residential_points['end_lng'])
            residential_points['total_count'] = residential_points['count_x'].fillna(0) + residential_points['count_y'].fillna(0)
            
            points_data = residential_points[['lat', 'lng', 'total_count']].dropna()
            
        else:  # work areas
            # 工作区域：早高峰到达 + 晚高峰出发
            morning_arrivals = bike_data[bike_data['is_morning_peak']].groupby(
                ['end_lat', 'end_lng']).size().reset_index(name='count')
            evening_departures = bike_data[bike_data['is_evening_peak']].groupby(
                ['start_lat', 'start_lng']).size().reset_index(name='count')
            
            # 合并数据
            work_points = pd.merge(
                morning_arrivals, evening_departures,
                left_on=['end_lat', 'end_lng'],
                right_on=['start_lat', 'start_lng'],
                how='outer'
            )
            
            # 清理数据
            work_points['lat'] = work_points['end_lat'].combine_first(work_points['start_lat'])
            work_points['lng'] = work_points['end_lng'].combine_first(work_points['start_lng'])
            work_points['total_count'] = work_points['count_x'].fillna(0) + work_points['count_y'].fillna(0)
            
            points_data = work_points[['lat', 'lng', 'total_count']].dropna()
        
        if len(points_data) < 10:
            log_progress(f"数据点不足，无法进行{area_type}区域聚类")
            return None
        
        # 使用DBSCAN聚类
        coords = points_data[['lat', 'lng']].values
        
        # 标准化坐标
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coords)
        
        # DBSCAN聚类
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        clusters = dbscan.fit_predict(coords_scaled)
        
        points_data['cluster'] = clusters
        
        # 计算每个集群的中心和大小
        cluster_info = []
        for cluster_id in set(clusters):
            if cluster_id == -1:  # 噪声点
                continue
                
            cluster_points = points_data[points_data['cluster'] == cluster_id]
            center_lat = cluster_points['lat'].mean()
            center_lng = cluster_points['lng'].mean()
            total_flow = cluster_points['total_count'].sum()
            cluster_size = len(cluster_points)
            
            cluster_info.append({
                'cluster_id': cluster_id,
                'center_lat': center_lat,
                'center_lng': center_lng,
                'total_flow': total_flow,
                'cluster_size': cluster_size
            })
        
        cluster_df = pd.DataFrame(cluster_info)
        cluster_df = cluster_df.sort_values('total_flow', ascending=False)
        
        log_progress(f"识别出 {len(cluster_df)} 个{area_type}集群")
        
        return cluster_df
        
    except Exception as e:
        log_progress(f"聚类分析错误: {str(e)}")
        return None

def create_enhanced_commuting_map(bike_data, major_corridors, station_importance, residential_areas, work_areas):
    """
    创建增强版通勤地图，包含通勤走廊、站点、居住和工作区域
    """
    try:
        log_progress("正在创建增强版通勤地图...")
        
        # 计算NYC中心点
        center_lat = bike_data['start_lat'].mean()
        center_lng = bike_data['start_lng'].mean()
        
        # 创建基础地图
        m = folium.Map(
            location=[center_lat, center_lng],
            zoom_start=11,
            tiles='CartoDB positron'
        )
        
        # === 修复数据类型问题 ===
        # 确保坐标是数值类型
        major_corridors['start_lat'] = pd.to_numeric(major_corridors['start_lat'], errors='coerce')
        major_corridors['start_lng'] = pd.to_numeric(major_corridors['start_lng'], errors='coerce')
        major_corridors['end_lat'] = pd.to_numeric(major_corridors['end_lat'], errors='coerce')
        major_corridors['end_lng'] = pd.to_numeric(major_corridors['end_lng'], errors='coerce')
        
        # 移除坐标无效的行
        major_corridors = major_corridors.dropna(subset=['start_lat', 'start_lng', 'end_lat', 'end_lng'])
        
        if station_importance is not None:
            station_importance['lat'] = pd.to_numeric(station_importance['lat'], errors='coerce')
            station_importance['lng'] = pd.to_numeric(station_importance['lng'], errors='coerce')
            station_importance = station_importance.dropna(subset=['lat', 'lng'])
        
        # === 添加居住区域 ===
        if residential_areas is not None and len(residential_areas) > 0:
            log_progress("正在添加居住区域...")
            
            # 选择前50个重要居住区域
            display_residential = residential_areas.head(30).copy()
            display_residential['lat'] = pd.to_numeric(display_residential['lat'], errors='coerce')
            display_residential['lng'] = pd.to_numeric(display_residential['lng'], errors='coerce')
            display_residential = display_residential.dropna(subset=['lat', 'lng'])
            
            # 分级显示
            if len(display_residential) >= 10:
                high_threshold = display_residential['residential_index'].quantile(0.80)
                medium_threshold = display_residential['residential_index'].quantile(0.50)
            else:
                max_index = display_residential['residential_index'].max()
                high_threshold = max_index * 0.8
                medium_threshold = max_index * 0.5
            
            for idx, area in display_residential.iterrows():
                try:
                    lat = float(area['lat'])
                    lng = float(area['lng'])
                    residential_index = float(area['residential_index'])
                    
                    # 根据居住指数设置大小和颜色
                    if residential_index > high_threshold:
                        radius = 300  # 米
                        color = 'darkblue'
                        area_level = "高密度居住区"
                    elif residential_index > medium_threshold:
                        radius = 200  # 米
                        color = 'blue'
                        area_level = "中密度居住区"
                    else:
                        radius = 100  # 米
                        color = 'lightblue'
                        area_level = "低密度居住区"
                    
                    folium.Circle(
                        location=[lat, lng],
                        radius=radius,
                        popup=folium.Popup(f"""
                        <b>{area_level}</b><br>
                        <b>站点:</b> {area['station_name']}<br>
                        <b>早高峰出发:</b> {int(area['morning_departures'])}<br>
                        <b>晚高峰到达:</b> {int(area['evening_arrivals'])}<br>
                        <b>居住指数:</b> {int(residential_index)}
                        """, max_width=300),
                        color=color,
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.3,
                        weight=1,
                        tooltip=f"{area_level}: {area['station_name']}"
                    ).add_to(m)
                    
                except Exception as e:
                    continue
        
        # === 添加工作区域 ===
        if work_areas is not None and len(work_areas) > 0:
            log_progress("正在添加工作区域...")
            
            # 选择前50个重要工作区域
            display_work = work_areas.head(30).copy()
            display_work['lat'] = pd.to_numeric(display_work['lat'], errors='coerce')
            display_work['lng'] = pd.to_numeric(display_work['lng'], errors='coerce')
            display_work = display_work.dropna(subset=['lat', 'lng'])
            
            # 分级显示
            if len(display_work) >= 10:
                high_threshold = display_work['work_index'].quantile(0.80)
                medium_threshold = display_work['work_index'].quantile(0.50)
            else:
                max_index = display_work['work_index'].max()
                high_threshold = max_index * 0.8
                medium_threshold = max_index * 0.5
            
            for idx, area in display_work.iterrows():
                try:
                    lat = float(area['lat'])
                    lng = float(area['lng'])
                    work_index = float(area['work_index'])
                    
                    # 根据工作指数设置大小和颜色
                    if work_index > high_threshold:
                        radius = 300  # 米
                        color = 'darkred'
                        area_level = "高密度工作区"
                    elif work_index > medium_threshold:
                        radius = 200  # 米
                        color = 'red'
                        area_level = "中密度工作区"
                    else:
                        radius = 100  # 米
                        color = 'pink'
                        area_level = "低密度工作区"
                    
                    folium.Circle(
                        location=[lat, lng],
                        radius=radius,
                        popup=folium.Popup(f"""
                        <b>{area_level}</b><br>
                        <b>站点:</b> {area['station_name']}<br>
                        <b>早高峰到达:</b> {int(area['morning_arrivals'])}<br>
                        <b>晚高峰出发:</b> {int(area['evening_departures'])}<br>
                        <b>工作指数:</b> {int(work_index)}
                        """, max_width=300),
                        color=color,
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.3,
                        weight=1,
                        tooltip=f"{area_level}: {area['station_name']}"
                    ).add_to(m)
                    
                except Exception as e:
                    continue
        
        # === 原有的通勤走廊和站点显示逻辑 ===
        log_progress("=== 走廊流量诊断信息 ===")
        log_progress(f"总走廊数量: {len(major_corridors)}")
        
        # 修复分级逻辑
        display_corridors = major_corridors.head(50).copy()
        
        if len(display_corridors) > 0:
            high_threshold = display_corridors['flow_intensity'].quantile(0.90)
            medium_threshold = display_corridors['flow_intensity'].quantile(0.60)
            
            log_progress(f"显示走廊流量范围: {display_corridors['flow_intensity'].min():.1f} - {display_corridors['flow_intensity'].max():.1f}")
            log_progress(f"分级阈值 - 高: {high_threshold:.1f}, 中: {medium_threshold:.1f}")
            
            # 添加通勤走廊
            corridors_to_show = min(50, len(display_corridors))
            corridors_added = {'high': 0, 'medium': 0, 'low': 0}
            
            for idx, corridor in display_corridors.iterrows():
                try:
                    intensity = corridor['flow_intensity']
                    
                    if intensity > high_threshold:
                        color = 'red'
                        weight = 5
                        flow_level = "High"
                        corridors_added['high'] += 1
                    elif intensity > medium_threshold:
                        color = 'orange'
                        weight = 3
                        flow_level = "Medium"
                        corridors_added['medium'] += 1
                    else:
                        color = 'blue'
                        weight = 2
                        flow_level = "Low"
                        corridors_added['low'] += 1
                    
                    start_lat = float(corridor['start_lat'])
                    start_lng = float(corridor['start_lng'])
                    end_lat = float(corridor['end_lat'])
                    end_lng = float(corridor['end_lng'])
                    
                    folium.PolyLine(
                        locations=[
                            [start_lat, start_lng],
                            [end_lat, end_lng]
                        ],
                        popup=folium.Popup(f"""
                        <b>Commuting Corridor - {flow_level} Flow</b><br>
                        <b>From:</b> {corridor['start_station_name']}<br>
                        <b>To:</b> {corridor['end_station_name']}<br>
                        <b>Trips:</b> {corridor['trip_count']}<br>
                        <b>Flow Intensity:</b> {intensity:.1f}
                        """, max_width=300),
                        color=color,
                        weight=weight,
                        opacity=0.7,
                        tooltip=f"{flow_level} Flow: {corridor['start_station_name']} to {corridor['end_station_name']}"
                    ).add_to(m)
                    
                except Exception as e:
                    continue
        
        # 添加重要站点
        if station_importance is not None and len(station_importance) > 0:
            log_progress("正在添加站点标记...")
            
            display_stations = station_importance.head(70).copy()
            
            if len(display_stations) >= 10:
                high_station_threshold = display_stations['total_flow'].quantile(0.90)
                medium_station_threshold = display_stations['total_flow'].quantile(0.60)
            else:
                max_flow = display_stations['total_flow'].max()
                high_station_threshold = max_flow * 0.9
                medium_station_threshold = max_flow * 0.5
            
            stations_to_show = min(100, len(display_stations))
            stations_added = {'major': 0, 'medium': 0, 'minor': 0}
            
            for idx, station in display_stations.iterrows():
                try:
                    total_flow = float(station['total_flow'])
                    lat = float(station['lat'])
                    lng = float(station['lng'])
                    
                    if total_flow > high_station_threshold:
                        radius = 8
                        color = 'darkpurple'
                        station_level = "Major"
                        stations_added['major'] += 1
                    elif total_flow > medium_station_threshold:
                        radius = 6
                        color = 'purple'
                        station_level = "Medium"
                        stations_added['medium'] += 1
                    else:
                        radius = 4
                        color = 'lavender'
                        station_level = "Minor"
                        stations_added['minor'] += 1
                    
                    folium.CircleMarker(
                        location=[lat, lng],
                        radius=radius,
                        popup=folium.Popup(f"""
                        <b>Station - {station_level}</b><br>
                        <b>Name:</b> {station['station_name']}<br>
                        <b>Departures:</b> {int(station['departures'])}<br>
                        <b>Arrivals:</b> {int(station['arrivals'])}<br>
                        <b>Total Flow:</b> {int(total_flow)}
                        """, max_width=300),
                        color=color,
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.7,
                        weight=1,
                        tooltip=f"{station_level} Station: {station['station_name']}"
                    ).add_to(m)
                    
                except Exception as e:
                    continue
        
        # 更新图例以包含居住和工作区域
        legend_html = '''
        <div style="position: fixed; 
                    top: 10px; left: 50px; width: 350px; height: 450px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 12px; border-radius: 5px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.2);">
        <p style="margin:0 0 10px 0; font-weight:bold; font-size:16px;">Urban Commuting Skeleton Map</p>
        
        <p style="margin:5px 0; font-weight:bold;">Residential Areas:</p>
        <p style="margin:2px 0;"><i style="background:darkblue; width:20px; height:20px; border-radius:50%; display:inline-block; margin-right:8px; vertical-align:middle"></i> High Density</p>
        <p style="margin:2px 0;"><i style="background:blue; width:16px; height:16px; border-radius:50%; display:inline-block; margin-right:8px; vertical-align:middle"></i> Medium Density</p>
        <p style="margin:2px 0 10px 0;"><i style="background:lightblue; width:12px; height:12px; border-radius:50%; display:inline-block; margin-right:8px; vertical-align:middle"></i> Low Density</p>
        
        <p style="margin:5px 0; font-weight:bold;">Work Areas:</p>
        <p style="margin:2px 0;"><i style="background:darkred; width:20px; height:20px; border-radius:50%; display:inline-block; margin-right:8px; vertical-align:middle"></i> High Density</p>
        <p style="margin:2px 0;"><i style="background:red; width:16px; height:16px; border-radius:50%; display:inline-block; margin-right:8px; vertical-align:middle"></i> Medium Density</p>
        <p style="margin:2px 0 10px 0;"><i style="background:pink; width:12px; height:12px; border-radius:50%; display:inline-block; margin-right:8px; vertical-align:middle"></i> Low Density</p>
        
        <p style="margin:5px 0; font-weight:bold;">Corridors:</p>
        <p style="margin:2px 0;"><i style="background:red; width:20px; height:4px; display:inline-block; margin-right:8px; vertical-align:middle"></i> High Flow</p>
        <p style="margin:2px 0;"><i style="background:orange; width:16px; height:3px; display:inline-block; margin-right:8px; vertical-align:middle"></i> Medium Flow</p>
        <p style="margin:2px 0;"><i style="background:blue; width:12px; height:2px; display:inline-block; margin-right:8px; vertical-align:middle"></i> Low Flow</p>
        
        <p style="margin:5px 0; font-weight:bold;">Stations:</p>
        <p style="margin:2px 0;"><i style="background:darkpurple; width:16px; height:16px; border-radius:50%; display:inline-block; margin-right:8px; vertical-align:middle"></i> Major Station</p>
        <p style="margin:2px 0;"><i style="background:purple; width:12px; height:12px; border-radius:50%; display:inline-block; margin-right:8px; vertical-align:middle"></i> Medium Station</p>
        <p style="margin:2px 0;"><i style="background:lavender; width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:8px; vertical-align:middle"></i> Minor Station</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        log_progress("✓ 增强版地图创建完成！")
        return m
        
    except Exception as e:
        log_progress(f"地图创建错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def analyze_spatial_patterns(bike_data, residential_areas, work_areas):
    """
    分析空间模式
    """
    try:
        log_progress("正在分析空间模式...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 居住区域分布
        if residential_areas is not None and len(residential_areas) > 0:
            top_residential = residential_areas.head(20)
            axes[0,0].barh(range(len(top_residential)), top_residential['residential_index'], color='blue', alpha=0.7)
            axes[0,0].set_yticks(range(len(top_residential)))
            axes[0,0].set_yticklabels(top_residential['station_name'], fontsize=8)
            axes[0,0].set_xlabel('Residential Index')
            axes[0,0].set_title('Top 20 Residential Areas')
            axes[0,0].grid(True, alpha=0.3)
        
        # 2. 工作区域分布
        if work_areas is not None and len(work_areas) > 0:
            top_work = work_areas.head(20)
            axes[0,1].barh(range(len(top_work)), top_work['work_index'], color='red', alpha=0.7)
            axes[0,1].set_yticks(range(len(top_work)))
            axes[0,1].set_yticklabels(top_work['station_name'], fontsize=8)
            axes[0,1].set_xlabel('Work Index')
            axes[0,1].set_title('Top 20 Work Areas')
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. 居住-工作关系散点图
        if residential_areas is not None and work_areas is not None:
            # 合并数据查看关系
            merged_areas = pd.merge(
                residential_areas[['station_id', 'station_name', 'residential_index']],
                work_areas[['station_id', 'work_index']],
                on='station_id',
                how='inner'
            )
            if len(merged_areas) > 0:
                axes[1,0].scatter(merged_areas['residential_index'], merged_areas['work_index'], alpha=0.6)
                axes[1,0].set_xlabel('Residential Index')
                axes[1,0].set_ylabel('Work Index')
                axes[1,0].set_title('Residential vs Work Index Relationship')
                axes[1,0].grid(True, alpha=0.3)
                
                # 添加趋势线
                z = np.polyfit(merged_areas['residential_index'], merged_areas['work_index'], 1)
                p = np.poly1d(z)
                axes[1,0].plot(merged_areas['residential_index'], p(merged_areas['residential_index']), "r--", alpha=0.8)
        
        # 4. 区域类型分布
        if residential_areas is not None and work_areas is not None:
            area_types = ['High Residential', 'Medium Residential', 'Low Residential',
                         'High Work', 'Medium Work', 'Low Work']
            
            # 简单分类
            if len(residential_areas) >= 10:
                res_high = len(residential_areas[residential_areas['residential_index'] > residential_areas['residential_index'].quantile(0.8)])
                res_medium = len(residential_areas[(residential_areas['residential_index'] > residential_areas['residential_index'].quantile(0.5)) & 
                                                 (residential_areas['residential_index'] <= residential_areas['residential_index'].quantile(0.8))])
                res_low = len(residential_areas[residential_areas['residential_index'] <= residential_areas['residential_index'].quantile(0.5)])
            else:
                res_high = res_medium = res_low = 0
                
            if len(work_areas) >= 10:
                work_high = len(work_areas[work_areas['work_index'] > work_areas['work_index'].quantile(0.8)])
                work_medium = len(work_areas[(work_areas['work_index'] > work_areas['work_index'].quantile(0.5)) & 
                                           (work_areas['work_index'] <= work_areas['work_index'].quantile(0.8))])
                work_low = len(work_areas[work_areas['work_index'] <= work_areas['work_index'].quantile(0.5)])
            else:
                work_high = work_medium = work_low = 0
            
            counts = [res_high, res_medium, res_low, work_high, work_medium, work_low]
            colors = ['darkblue', 'blue', 'lightblue', 'darkred', 'red', 'pink']
            
            axes[1,1].bar(area_types, counts, color=colors, alpha=0.7)
            axes[1,1].set_ylabel('Number of Areas')
            axes[1,1].set_title('Distribution of Area Types')
            axes[1,1].tick_params(axis='x', rotation=45)
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('spatial_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
        
    except Exception as e:
        log_progress(f"空间模式分析错误: {str(e)}")
        return None

def analyze_basic_patterns(bike_data):
    """
    分析基本模式
    """
    try:
        log_progress("正在分析基本模式...")
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 小时分布（如果存在start_hour列）
        if 'start_hour' in bike_data.columns:
            hourly_pattern = bike_data.groupby('start_hour').size()
            axes[0,0].plot(hourly_pattern.index, hourly_pattern.values, marker='o', linewidth=2, color='blue')
            axes[0,0].set_xlabel('Hour of Day')
            axes[0,0].set_ylabel('Number of Trips')
            axes[0,0].set_title('Hourly Trip Distribution')
            axes[0,0].grid(True, alpha=0.3)
        else:
            axes[0,0].text(0.5, 0.5, 'No start_hour data', ha='center', va='center')
            axes[0,0].set_title('Hourly Distribution (No Data)')
        
        # 星期分布（如果存在start_dayofweek列）
        if 'start_dayofweek' in bike_data.columns:
            daily_pattern = bike_data.groupby('start_dayofweek').size()
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            axes[0,1].bar(days, daily_pattern.values, color='skyblue', alpha=0.7)
            axes[0,1].set_xlabel('Day of Week')
            axes[0,1].set_ylabel('Number of Trips')
            axes[0,1].set_title('Weekly Trip Distribution')
        else:
            axes[0,1].text(0.5, 0.5, 'No start_dayofweek data', ha='center', va='center')
            axes[0,1].set_title('Weekly Distribution (No Data)')
        
        # 时长分布（如果存在duration_minutes列）
        if 'duration_minutes' in bike_data.columns:
            axes[1,0].hist(bike_data['duration_minutes'], bins=50, alpha=0.7, color='lightgreen')
            axes[1,0].set_xlabel('Trip Duration (minutes)')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].set_title('Trip Duration Distribution')
            axes[1,0].set_xlim(0, 120)
        else:
            axes[1,0].text(0.5, 0.5, 'No duration_minutes data', ha='center', va='center')
            axes[1,0].set_title('Duration Distribution (No Data)')
        
        # 距离分布（如果存在distance_km列）
        if 'distance_km' in bike_data.columns:
            axes[1,1].hist(bike_data['distance_km'], bins=50, alpha=0.7, color='salmon')
            axes[1,1].set_xlabel('Trip Distance (km)')
            axes[1,1].set_ylabel('Frequency')
            axes[1,1].set_title('Trip Distance Distribution')
            axes[1,1].set_xlim(0, 10)
        else:
            axes[1,1].text(0.5, 0.5, 'No distance_km data', ha='center', va='center')
            axes[1,1].set_title('Distance Distribution (No Data)')
        
        plt.tight_layout()
        plt.savefig('basic_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
        
    except Exception as e:
        log_progress(f"模式分析错误: {str(e)}")
        return None

def main():
    """
    主函数
    """
    start_time = time.time()
    log_progress("=== NYC城市公共交通韧性分析开始 ===")
    
    # 1. 加载数据
    bike_data = load_and_preprocess_bike_data()
    if bike_data is None:
        log_progress("数据加载失败，程序退出")
        return
    
    # 2. 基本模式分析
    analyze_basic_patterns(bike_data)
    
    # 3. 识别通勤走廊
    major_corridors = identify_commuting_corridors(bike_data)
    if major_corridors is None or len(major_corridors) == 0:
        log_progress("通勤走廊识别失败，程序退出")
        return
    
    # 4. 计算站点重要性
    station_importance = get_station_importance(bike_data, major_corridors)
    
    # 5. 识别居住和工作区域
    residential_areas, work_areas = identify_residential_work_areas(bike_data)
    
    # 6. 空间模式分析
    analyze_spatial_patterns(bike_data, residential_areas, work_areas)
    
    # 7. 使用聚类算法识别区域（可选）
    residential_clusters = create_cluster_areas(bike_data, "residential")
    work_clusters = create_cluster_areas(bike_data, "work")
    
    # 8. 创建增强版地图
    commuting_map = create_enhanced_commuting_map(bike_data, major_corridors, station_importance, residential_areas, work_areas)
    if commuting_map is None:
        log_progress("地图创建失败，程序退出")
        return
    
    # 9. 保存地图
    commuting_map.save('nyc_urban_commuting_comprehensive.html')
    log_progress("✓ 综合城市通勤地图已保存为 'nyc_urban_commuting_comprehensive.html'")
    try:
        save_map_as_image("nyc_urban_commuting_comprehensive.html", "nyc_commuting_map.png")
    except:
        print("自动截图失败，请手动截图")
    
    # 10. 输出摘要
    log_progress("\n" + "="*50)
    log_progress("分析摘要")
    log_progress("="*50)
    log_progress(f"分析出行总数: {len(bike_data):,}")
    log_progress(f"主要通勤走廊: {len(major_corridors)}")
    if station_importance is not None:
        log_progress(f"分析站点数量: {len(station_importance)}")
    if residential_areas is not None:
        log_progress(f"识别居住区域: {len(residential_areas)}")
    if work_areas is not None:
        log_progress(f"识别工作区域: {len(work_areas)}")
    
    elapsed_time = time.time() - start_time
    log_progress(f"\n✓ 分析完成! 耗时: {elapsed_time:.1f} 秒")
    log_progress("✓ 请在浏览器中打开 'nyc_urban_commuting_comprehensive.html' 查看综合地图")
    log_progress("✓ 基本分析图表已保存为 'basic_analysis.png'")
    log_progress("✓ 空间分析图表已保存为 'spatial_analysis.png'")



# 运行分析
if __name__ == "__main__":
    main()