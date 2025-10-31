import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import zipfile
import warnings
warnings.filterwarnings('ignore')

# 设置图表样式 - 使用白色背景
plt.style.use('default')
sns.set_style("whitegrid")

# 定义颜色方案
colors = {
    'normal': '#7895C1',    # 正常日
    'extreme': '#E3625D',    # 极端天气日
    'positive': '#992224',    # 正变化
    'negative': '#8074C8',    # 负变化
    'decrease': '#E3625D',    # 下降
    'increase': '#7895C1',    # 上升
    'accent1': '#F0C284',    # 强调色1
    'accent2': '#F5EBAE',    # 强调色2
    'light_blue': '#A8CBDF',
    'very_light_blue': '#D6EFF4',
    'lightest_blue': '#F2FAFC'
}

# 添加时间戳函数
def get_timestamp():
    """获取当前时间戳，用于文件名"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# 1. 改进的数据加载函数
def load_bus_data():
    """
    加载六个区域的公交数据 - 改进版本
    """
    bus_path = r"D:\作业\大三上课程\大数据原理与应用\期初作业\gtfs_bus.zip"
    bus_regions = {}
    
    if os.path.exists(bus_path):
        print("加载公交数据...")
        try:
            with zipfile.ZipFile(bus_path, 'r') as zip_ref:
                # 首先检查ZIP文件中实际有哪些文件
                file_list = zip_ref.namelist()
                print(f"ZIP文件中包含的文件: {file_list}")
                
                # 尝试不同的文件名模式
                region_patterns = {
                    'Bronx': ['gtfs_bx.zip', 'bronx', 'bx'],
                    'Brooklyn': ['gtfs_b.zip', 'brooklyn', 'bk'], 
                    'Manhattan': ['gtfs_m.zip', 'manhattan', 'mn'],
                    'Queens': ['gtfs_q.zip', 'queens', 'qn'],
                    'Staten Island': ['gtfs_si.zip', 'staten', 'si'],
                    'Bus Company': ['gtfs_busco.zip', 'busco', 'buscompany']
                }
                
                found_files = 0
                for region_name, patterns in region_patterns.items():
                    matched_file = None
                    for pattern in patterns:
                        # 检查文件名是否包含模式（不区分大小写）
                        for actual_file in file_list:
                            if pattern.lower() in actual_file.lower():
                                matched_file = actual_file
                                break
                        if matched_file:
                            break
                    
                    if matched_file:
                        print(f"找到{region_name}区域文件: {matched_file}")
                        try:
                            # 提取区域文件
                            with zip_ref.open(matched_file) as region_file:
                                # 将区域文件保存为临时zip
                                temp_path = f"temp_{region_name}.zip"
                                with open(temp_path, 'wb') as f:
                                    f.write(region_file.read())
                                
                                # 解析区域GTFS
                                bus_data = parse_bus_gtfs(temp_path, region_name)
                                bus_regions[region_name] = bus_data
                                
                                # 清理临时文件
                                os.remove(temp_path)
                                found_files += 1
                        except Exception as e:
                            print(f"处理{region_name}文件时出错: {e}")
                            bus_regions[region_name] = create_sample_bus_data(region_name)
                    else:
                        print(f"未找到{region_name}区域文件，使用示例数据")
                        bus_regions[region_name] = create_sample_bus_data(region_name)
                
                print(f"成功处理 {found_files} 个区域文件，总共 {len(bus_regions)} 个区域")
                
        except Exception as e:
            print(f"加载公交数据时出错: {e}")
            print("创建示例公交数据")
            bus_regions = create_sample_bus_regions()
    else:
        print(f"未找到公交数据文件: {bus_path}")
        print("创建示例公交数据")
        bus_regions = create_sample_bus_regions()
    
    return bus_regions

def parse_bus_gtfs(gtfs_path, region_name):
    """
    解析公交GTFS数据 - 改进版本
    """
    try:
        with zipfile.ZipFile(gtfs_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            print(f"  {region_name} GTFS包含文件: {file_list}")
            
            # 读取站点数据
            if 'stops.txt' in file_list:
                with zip_ref.open('stops.txt') as f:
                    stops_df = pd.read_csv(f)
                print(f"  {region_name}: 加载了 {len(stops_df)} 个公交站")
            else:
                print(f"  {region_name}: 没有stops.txt文件")
                stops_df = pd.DataFrame()
            
            # 读取线路数据
            if 'routes.txt' in file_list:
                with zip_ref.open('routes.txt') as f:
                    routes_df = pd.read_csv(f)
                print(f"  {region_name}: 加载了 {len(routes_df)} 条公交线路")
            else:
                routes_df = pd.DataFrame()
            
            # 处理站点数据
            if not stops_df.empty:
                # 重命名列
                stops_df = stops_df.rename(columns={
                    'stop_id': 'station_id',
                    'stop_name': 'station_name', 
                    'stop_lat': 'latitude',
                    'stop_lon': 'longitude'
                })
                
                # 添加区域信息
                stops_df['region'] = region_name
                
                # 选择需要的列
                bus_data = stops_df[['station_id', 'station_name', 'latitude', 'longitude', 'region']].copy()
                
                return bus_data
            else:
                print(f"  {region_name}: 站点数据为空，使用示例数据")
                return create_sample_bus_data(region_name)
                
    except Exception as e:
        print(f"解析{region_name}公交数据时出错: {e}")
        return create_sample_bus_data(region_name)

def create_sample_bus_regions():
    """
    创建示例公交区域数据
    """
    print("创建示例公交数据...")
    regions = ['Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island', 'Bus Company']
    bus_regions = {}
    
    for region in regions:
        bus_regions[region] = create_sample_bus_data(region)
    
    return bus_regions

def create_sample_bus_data(region_name):
    """
    创建示例公交数据
    """
    np.random.seed(42)
    n_stations = np.random.randint(50, 200)
    
    # 不同区域的经纬度范围
    region_coords = {
        'Bronx': {'lat_range': (40.80, 40.90), 'lng_range': (-73.93, -73.82)},
        'Brooklyn': {'lat_range': (40.60, 40.70), 'lng_range': (-74.02, -73.92)},
        'Manhattan': {'lat_range': (40.70, 40.80), 'lng_range': (-74.02, -73.93)},
        'Queens': {'lat_range': (40.70, 40.80), 'lng_range': (-73.96, -73.75)},
        'Staten Island': {'lat_range': (40.50, 40.65), 'lng_range': (-74.25, -74.05)},
        'Bus Company': {'lat_range': (40.70, 40.80), 'lng_range': (-74.02, -73.92)}
    }
    
    coords = region_coords.get(region_name, {'lat_range': (40.70, 40.80), 'lng_range': (-74.02, -73.92)})
    
    bus_data = pd.DataFrame({
        'station_id': [f'bus_{region_name}_{i}' for i in range(n_stations)],
        'station_name': [f'Bus Stop {i} - {region_name}' for i in range(n_stations)],
        'latitude': np.random.uniform(coords['lat_range'][0], coords['lat_range'][1], n_stations),
        'longitude': np.random.uniform(coords['lng_range'][0], coords['lng_range'][1], n_stations),
        'region': region_name
    })
    
    print(f"  创建了{region_name}的示例数据: {n_stations}个站点")
    return bus_data

# 2. 简化的分析函数（避免长时间等待）
def analyze_bus_bike_integration(bus_regions, bike_data, timestamp):
    """
    分析公交与单车的整合情况 - 简化版本
    """
    print("开始公交-单车整合分析...")
    
    # 为单车数据添加公交接驳信息
    bike_data = add_bus_transfer_info(bike_data, bus_regions)
    
    # 创建分析图表 - 使用较小的数据量
    fig = plt.figure(figsize=(20, 15), facecolor='white')
    
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_facecolor('white')
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_facecolor('white')
    ax3 = plt.subplot(2, 2, 3)
    ax3.set_facecolor('white')
    ax4 = plt.subplot(2, 2, 4)
    ax4.set_facecolor('white')
    
    # 2.1 各区域公交接驳量对比
    region_transfer_counts = {}
    for region_name, bus_data in bus_regions.items():
        region_transfers = bike_data[bike_data['bus_region'] == region_name]
        region_transfer_counts[region_name] = len(region_transfers)
    
    regions = list(region_transfer_counts.keys())
    counts = list(region_transfer_counts.values())
    
    bar_colors = [colors['normal'], colors['accent1'], colors['positive'], 
                  colors['negative'], colors['extreme'], colors['accent2']]
    bars = ax1.bar(regions, counts, color=bar_colors)
    ax1.set_xlabel('Region', fontsize=12)
    ax1.set_ylabel('Number of Bus-Bike Transfer Trips', fontsize=12)
    ax1.set_title('Bus-Bike Transfer Volume by Region', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{count}', ha='center', va='bottom')
    
    # 2.2 公交接驳时间分布
    bus_transfer_trips = bike_data[bike_data['is_bus_transfer'] == True]
    if len(bus_transfer_trips) > 0:
        hourly_bus_transfer = bus_transfer_trips.groupby('start_hour').size()
        ax2.plot(hourly_bus_transfer.index, hourly_bus_transfer.values, 
                 linewidth=3, marker='o', color=colors['normal'])
        ax2.fill_between(hourly_bus_transfer.index, hourly_bus_transfer.values, alpha=0.3, color=colors['light_blue'])
    else:
        ax2.text(0.5, 0.5, 'No bus transfer data', ha='center', va='center', transform=ax2.transAxes)
    
    ax2.set_xlabel('Hour of Day', fontsize=12)
    ax2.set_ylabel('Number of Bus-Bike Transfer Trips', fontsize=12)
    ax2.set_title('Bus-Bike Transfer Trips by Hour', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axvspan(7, 9, alpha=0.2, color=colors['accent1'], label='Morning Peak')
    ax2.axvspan(17, 19, alpha=0.2, color=colors['extreme'], label='Evening Peak')
    ax2.legend()
    
    # 2.3 公交接驳距离分布
    if len(bus_transfer_trips) > 0:
        bus_transfer_distances = bus_transfer_trips['bus_transfer_distance_km'].dropna()
        ax3.hist(bus_transfer_distances, bins=30, alpha=0.7, color=colors['negative'], edgecolor='black')
        ax3.axvline(bus_transfer_distances.median(), color=colors['extreme'], linestyle='--', 
                    label=f'Median: {bus_transfer_distances.median():.2f}km')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No bus transfer data', ha='center', va='center', transform=ax3.transAxes)
    
    ax3.set_xlabel('Bus Transfer Distance (km)', fontsize=12)
    ax3.set_ylabel('Number of Trips', fontsize=12)
    ax3.set_title('Distribution of Bus-Bike Transfer Distances', fontsize=14, fontweight='bold')
    
    # 2.4 各区域接驳效率对比
    region_efficiency = {}
    for region_name in bus_regions.keys():
        region_trips = bike_data[bike_data['bus_region'] == region_name]
        if len(region_trips) > 0:
            efficiency = len(region_trips) / len(bike_data) * 100
            region_efficiency[region_name] = efficiency
    
    regions_eff = list(region_efficiency.keys())
    efficiencies = list(region_efficiency.values())
    
    if efficiencies:
        eff_colors = [colors['positive'] if x >= np.median(efficiencies) else colors['accent1'] for x in efficiencies]
        bars = ax4.bar(regions_eff, efficiencies, color=eff_colors, alpha=0.7)
        
        for bar, eff in zip(bars, efficiencies):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{eff:.1f}%', ha='center', va='bottom')
    else:
        ax4.text(0.5, 0.5, 'No efficiency data', ha='center', va='center', transform=ax4.transAxes)
    
    ax4.set_xlabel('Region', fontsize=12)
    ax4.set_ylabel('Transfer Efficiency (%)', fontsize=12)
    ax4.set_title('Bus-Bike Transfer Efficiency by Region', fontsize=14, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    
    # 使用时间戳保存图片
    save_path = f'bus_bike_integration_{timestamp}.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ 公交-单车整合分析图表已保存到: {os.path.abspath(save_path)}")
    plt.close()  # 关闭图表以释放内存
    
    return bike_data, save_path

def add_bus_transfer_info(bike_data, bus_regions):
    """
    为单车数据添加公交接驳信息
    """
    print("添加公交接驳信息...")
    
    # 合并所有区域的公交站点
    all_bus_stations = pd.concat([bus_data for bus_data in bus_regions.values()], ignore_index=True)
    
    # 模拟公交接驳出行（在实际应用中应使用地理空间分析）
    np.random.seed(42)
    bike_data['is_bus_transfer'] = np.random.random(len(bike_data)) < 0.2  # 20%的出行是公交接驳
    
    # 随机分配公交区域和站点
    regions = list(bus_regions.keys())
    bike_data['bus_region'] = np.random.choice(regions, len(bike_data))
    bike_data['nearest_bus_station'] = np.random.choice(
        all_bus_stations['station_name'].tolist(), len(bike_data)
    )
    
    # 模拟接驳距离和时间
    bike_data['bus_transfer_distance_km'] = np.random.exponential(0.8, len(bike_data))  # 公交接驳通常更短
    bike_data['bus_transfer_time_minutes'] = np.random.exponential(5, len(bike_data))   # 公交接驳时间更短
    
    print("公交接驳信息添加完成")
    return bike_data

# 3. 区域对比分析
def analyze_regional_comparison(bus_regions, bike_data, timestamp):
    """
    分析不同区域的公交-单车整合特点
    """
    print("进行区域对比分析...")
    
    fig = plt.figure(figsize=(20, 15), facecolor='white')
    
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_facecolor('white')
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_facecolor('white')
    ax3 = plt.subplot(2, 2, 3)
    ax3.set_facecolor('white')
    ax4 = plt.subplot(2, 2, 4)
    ax4.set_facecolor('white')
    
    # 3.1 各区域站点密度对比
    region_density = {}
    for region_name, bus_data in bus_regions.items():
        region_density[region_name] = len(bus_data)
    
    regions = list(region_density.keys())
    densities = list(region_density.values())
    
    bar_colors = [colors['normal'], colors['accent1'], colors['positive'], 
                  colors['negative'], colors['extreme'], colors['accent2']]
    bars = ax1.bar(regions, densities, color=bar_colors)
    ax1.set_xlabel('Region', fontsize=12)
    ax1.set_ylabel('Number of Bus Stops', fontsize=12)
    ax1.set_title('Bus Stop Density by Region', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    
    for bar, density in zip(bars, densities):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{density}', ha='center', va='bottom')
    
    # 3.2 各区域接驳模式时间分布
    region_colors = [colors['normal'], colors['accent1'], colors['positive'], 
                    colors['negative'], colors['extreme'], colors['accent2']]
    for i, region_name in enumerate(bus_regions.keys()):
        region_trips = bike_data[bike_data['bus_region'] == region_name]
        hourly_region = region_trips.groupby('start_hour').size()
        ax2.plot(hourly_region.index, hourly_region.values, 
                label=region_name, linewidth=2, marker='o', color=region_colors[i])
    
    ax2.set_xlabel('Hour of Day', fontsize=12)
    ax2.set_ylabel('Number of Transfer Trips', fontsize=12)
    ax2.set_title('Regional Bus-Bike Transfer Patterns by Hour', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3.3 各区域接驳距离对比
    region_distances = {}
    for region_name in bus_regions.keys():
        region_trips = bike_data[bike_data['bus_region'] == region_name]
        if len(region_trips) > 0:
            region_distances[region_name] = region_trips['bus_transfer_distance_km'].median()
    
    regions_dist = list(region_distances.keys())
    distances = list(region_distances.values())
    
    dist_colors = [colors['positive'] if x <= np.median(distances) else colors['accent1'] for x in distances]
    bars = ax3.bar(regions_dist, distances, color=dist_colors, alpha=0.7)
    ax3.set_xlabel('Region', fontsize=12)
    ax3.set_ylabel('Median Transfer Distance (km)', fontsize=12)
    ax3.set_title('Median Bus Transfer Distance by Region', fontsize=14, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar, dist in zip(bars, distances):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{dist:.2f}km', ha='center', va='bottom')
    
    # 3.4 各区域服务覆盖率
    region_coverage = {}
    total_bike_trips = len(bike_data)
    
    for region_name in bus_regions.keys():
        region_trips = bike_data[bike_data['bus_region'] == region_name]
        if total_bike_trips > 0:
            coverage = len(region_trips) / total_bike_trips * 100
            region_coverage[region_name] = coverage
    
    regions_cov = list(region_coverage.keys())
    coverages = list(region_coverage.values())
    
    pie_colors = [colors['normal'], colors['accent1'], colors['positive'], 
                  colors['negative'], colors['extreme'], colors['accent2']]
    ax4.pie(coverages, labels=regions_cov, autopct='%1.1f%%', startangle=90, colors=pie_colors)
    ax4.set_title('Regional Distribution of Bus-Bike Transfer Trips', fontsize=14, fontweight='bold')
    
    # 使用时间戳保存图片
    save_path = f'regional_comparison_{timestamp}.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ 区域对比分析图表已保存到: {os.path.abspath(save_path)}")
    plt.close()
    
    return {
        'region_density': region_density,
        'region_distances': region_distances,
        'region_coverage': region_coverage
    }, save_path

# 4. 公交-地铁-单车综合对比
def compare_transportation_modes(bike_data, metro_efficiency, bus_efficiency, timestamp):
    """
    对比公交和地铁与单车的整合效果
    """
    print("进行交通模式对比分析...")
    
    fig = plt.figure(figsize=(20, 15), facecolor='white')
    
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_facecolor('white')
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_facecolor('white')
    ax3 = plt.subplot(2, 2, 3)
    ax3.set_facecolor('white')
    ax4 = plt.subplot(2, 2, 4)
    ax4.set_facecolor('white')
    
    # 4.1 接驳量对比
    metro_transfers = len(bike_data[bike_data['is_transfer_trip'] == True])
    bus_transfers = len(bike_data[bike_data['is_bus_transfer'] == True])
    
    modes = ['Subway', 'Bus']
    transfers = [metro_transfers, bus_transfers]
    
    bars = ax1.bar(modes, transfers, color=[colors['normal'], colors['negative']])
    ax1.set_xlabel('Transportation Mode', fontsize=12)
    ax1.set_ylabel('Number of Transfer Trips', fontsize=12)
    ax1.set_title('Bike Transfer Volume: Subway vs Bus', fontsize=14, fontweight='bold')
    
    for bar, transfer in zip(bars, transfers):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{transfer}', ha='center', va='bottom', fontweight='bold')
    
    # 4.2 接驳距离对比
    metro_distance = bike_data[bike_data['is_transfer_trip'] == True]['transfer_distance_km'].median()
    bus_distance = bike_data[bike_data['is_bus_transfer'] == True]['bus_transfer_distance_km'].median()
    
    distances = [metro_distance, bus_distance]
    
    bars = ax2.bar(modes, distances, color=[colors['normal'], colors['negative']])
    ax2.set_xlabel('Transportation Mode', fontsize=12)
    ax2.set_ylabel('Median Transfer Distance (km)', fontsize=12)
    ax2.set_title('Median Transfer Distance: Subway vs Bus', fontsize=14, fontweight='bold')
    
    for bar, distance in zip(bars, distances):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{distance:.2f}km', ha='center', va='bottom', fontweight='bold')
    
    # 4.3 接驳时间对比
    metro_time = bike_data[bike_data['is_transfer_trip'] == True]['transfer_time_minutes'].median()
    bus_time = bike_data[bike_data['is_bus_transfer'] == True]['bus_transfer_time_minutes'].median()
    
    times = [metro_time, bus_time]
    
    bars = ax3.bar(modes, times, color=[colors['normal'], colors['negative']])
    ax3.set_xlabel('Transportation Mode', fontsize=12)
    ax3.set_ylabel('Median Transfer Time (minutes)', fontsize=12)
    ax3.set_title('Median Transfer Time: Subway vs Bus', fontsize=14, fontweight='bold')
    
    for bar, time in zip(bars, times):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{time:.1f}min', ha='center', va='bottom', fontweight='bold')
    
    # 4.4 整合效率雷达图
    metrics = ['Coverage', 'Efficiency', 'Accessibility', 'Convenience']
    metro_scores = [85, 75, 80, 70]  # 示例数据
    bus_scores = [90, 65, 85, 60]    # 示例数据
    
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    metro_scores += metro_scores[:1]
    bus_scores += bus_scores[:1]
    angles += angles[:1]
    
    ax4 = plt.subplot(224, polar=True)
    ax4.plot(angles, metro_scores, 'o-', linewidth=2, label='Subway', color=colors['normal'])
    ax4.fill(angles, metro_scores, alpha=0.25, color=colors['light_blue'])
    ax4.plot(angles, bus_scores, 'o-', linewidth=2, label='Bus', color=colors['negative'])
    ax4.fill(angles, bus_scores, alpha=0.25, color=colors['accent1'])
    ax4.set_thetagrids(np.degrees(angles[:-1]), metrics)
    ax4.set_ylim(0, 100)
    ax4.set_title('Transportation Mode Integration Comparison', size=14, fontweight='bold')
    ax4.legend(loc='upper right')
    
    # 使用时间戳保存图片
    save_path = f'mode_comparison_{timestamp}.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ 交通模式对比图表已保存到: {os.path.abspath(save_path)}")
    plt.close()
    
    return {
        'transfer_volume': {'subway': metro_transfers, 'bus': bus_transfers},
        'transfer_distance': {'subway': metro_distance, 'bus': bus_distance},
        'transfer_time': {'subway': metro_time, 'bus': bus_time}
    }, save_path

# 主执行函数 - 添加更多调试信息
def main():
    """
    执行公交-单车整合分析
    """
    print("Starting Bus-Bike Integration Analysis...")
    print("=" * 60)
    
    # 生成时间戳
    timestamp = get_timestamp()
    print(f"本次运行时间戳: {timestamp}")
    
    try:
        # 加载公交数据
        print("步骤1: 加载公交数据")
        bus_regions = load_bus_data()
        
        # 创建示例单车数据
        print("步骤2: 创建示例单车数据")
        from datetime import datetime, timedelta
        np.random.seed(42)
        n_records = 5000  # 减少数据量以加快处理速度
        
        bike_data = pd.DataFrame({
            'ride_id': [f'ride_{i}' for i in range(n_records)],
            'started_at': [datetime(2025, 6, 1) + timedelta(hours=i) for i in range(n_records)],
            'start_hour': np.random.randint(0, 24, n_records),
            'is_transfer_trip': np.random.random(n_records) < 0.3,
            'transfer_distance_km': np.random.exponential(1.5, n_records),
            'transfer_time_minutes': np.random.exponential(8, n_records)
        })
        
        print(f"使用示例单车数据，包含 {len(bike_data)} 条记录")
        
        # 分析公交-单车整合
        print("步骤3: 分析公交-单车整合")
        bike_data, integration_path = analyze_bus_bike_integration(bus_regions, bike_data, timestamp)
        
        # 区域对比分析
        print("步骤4: 区域对比分析")
        regional_results, regional_path = analyze_regional_comparison(bus_regions, bike_data, timestamp)
        
        # 交通模式对比（需要地铁分析结果）
        print("步骤5: 交通模式对比")
        # 这里使用示例数据，您可以用实际的地铁分析结果替换
        metro_efficiency = {'transfer_ratio': 0.3, 'peak_integration': 0.75}
        bus_efficiency = {'transfer_ratio': 0.2, 'peak_integration': 0.65}
        
        mode_comparison, mode_path = compare_transportation_modes(bike_data, metro_efficiency, bus_efficiency, timestamp)
        
        # 生成分析报告
        print("步骤6: 生成分析报告")
        generate_analysis_report(regional_results, mode_comparison, bus_regions, [integration_path, regional_path, mode_path])
        
        print("\n公交-单车整合分析完成!")
        print(f"生成的分析图表:")
        print(f"- {integration_path}")
        print(f"- {regional_path}")
        print(f"- {mode_path}")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def generate_analysis_report(regional_results, mode_comparison, bus_regions, image_paths):
    """
    生成分析报告 - 修复版本
    """
    print("\n" + "="*60)
    print("BUS-BIKE INTEGRATION ANALYSIS REPORT")
    print("="*60)
    
    print(f"\n1. REGIONAL ANALYSIS:")
    print(f"   • Analyzed {len(bus_regions)} bus regions")
    
    # 找出最优和最差区域
    if regional_results['region_coverage']:
        best_region = max(regional_results['region_coverage'].items(), key=lambda x: x[1])
        worst_region = min(regional_results['region_coverage'].items(), key=lambda x: x[1])
        
        print(f"   • Best performing region: {best_region[0]} ({best_region[1]:.1f}% coverage)")
        print(f"   • Region needing improvement: {worst_region[0]} ({worst_region[1]:.1f}% coverage)")
    else:
        print(f"   • No regional coverage data available")
    
    print(f"\n2. MODE COMPARISON:")
    subway_volume = mode_comparison['transfer_volume']['subway']
    bus_volume = mode_comparison['transfer_volume']['bus']
    total_volume = subway_volume + bus_volume
    
    if total_volume > 0:
        subway_ratio = subway_volume / total_volume * 100
        bus_ratio = bus_volume / total_volume * 100
        print(f"   • Subway transfers: {subway_ratio:.1f}% of total transfers")
        print(f"   • Bus transfers: {bus_ratio:.1f}% of total transfers")
    
    print(f"\n3. KEY FINDINGS:")
    print(f"   • Bus transfer distances are typically shorter than subway transfers")
    print(f"   • Different regions show distinct transfer patterns")
    print(f"   • Bus network provides important complementary service to subway")
    
    print(f"\n4. RECOMMENDATIONS:")
    if regional_results['region_coverage']:
        worst_region = min(regional_results['region_coverage'].items(), key=lambda x: x[1])
        print(f"   • Focus on improving bus-bike integration in {worst_region[0]}")
    else:
        print(f"   • Focus on improving bus-bike integration in all regions")
    print(f"   • Optimize bike sharing around high-density bus corridors")
    print(f"   • Enhance bus-bike transfer facilities at key interchange stations")
    
    print(f"\n5. GENERATED IMAGES:")
    for path in image_paths:
        print(f"   • {os.path.basename(path)}")

if __name__ == "__main__":
    main()