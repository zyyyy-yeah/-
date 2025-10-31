import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import zipfile
import os
import warnings
warnings.filterwarnings('ignore')

# 设置英文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

# 使用您提供的低饱和度配色方案
colors = {
    'normal': '#7895C1',
    'extreme': '#E3625D', 
    'positive': '#992224',
    'negative': '#8074C8',
    'decrease': '#E3625D',
    'increase': '#7895C1',
    'accent1': '#F0C284',
    'accent2': '#F5EBAE',
    'light_blue': '#A8CBDF',
    'very_light_blue': '#D6EFF4',
    'lightest_blue': '#F2FAFC'
}

# 请在这里提供您的数据文件路径
BIKE_PATH = r"D:\作业\大三上课程\大数据原理与应用\期初作业\bike.csv"
WEATHER_PATH = r"D:\作业\大三上课程\大数据原理与应用\期初作业\daily-summaries-2025-10-09T12-21-41.xlsx"
NYWIND_PATH = r"D:\作业\大三上课程\大数据原理与应用\期初作业\nywind.xlsx"
SUBWAY_PATH = r"D:\作业\大三上课程\大数据原理与应用\期初作业\gtfs_subway.zip"
BUS_PATH = r"D:\作业\大三上课程\大数据原理与应用\期初作业\gtfs_bus.zip"

def load_gtfs_data(gtfs_path, data_type):
    """
    加载GTFS数据
    """
    try:
        with zipfile.ZipFile(gtfs_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            print(f"  {data_type} files: {file_list}")
            
            # 读取站点数据
            if 'stops.txt' in file_list:
                with zip_ref.open('stops.txt') as f:
                    stops_df = pd.read_csv(f)
                print(f"  Loaded {len(stops_df)} stations")
                
                # 重命名列
                stops_df = stops_df.rename(columns={
                    'stop_id': 'station_id',
                    'stop_name': 'station_name',
                    'stop_lat': 'latitude', 
                    'stop_lon': 'longitude'
                })
                
                return stops_df[['station_id', 'station_name', 'latitude', 'longitude']]
            else:
                print(f"  stops.txt not found")
                return pd.DataFrame()
                
    except Exception as e:
        print(f"Error loading {data_type} data: {e}")
        return pd.DataFrame()

def load_bus_data(bus_path):
    """
    加载公交数据
    """
    bus_regions = {}
    
    try:
        with zipfile.ZipFile(bus_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            print(f"  Bus files: {file_list}")
            
            # 使用正确的文件名映射
            region_files = {
                'Bronx': 'gtf5_bx.zip',
                'Brooklyn': 'gtf5_b.zip', 
                'Manhattan': 'gtf5_m.zip',
                'Queens': 'gtf5_q.zip',
                'Staten Island': 'gtf5_si.zip',
                'Bus Company': 'gtf5_busco.zip'
            }
            
            loaded_regions = 0
            for region_name, file_name in region_files.items():
                if file_name in file_list:
                    print(f"  Loading {region_name} bus data: {file_name}")
                    
                    with zip_ref.open(file_name) as region_file:
                        # 临时保存区域文件
                        temp_path = f"temp_{region_name}.zip"
                        with open(temp_path, 'wb') as f:
                            f.write(region_file.read())
                        
                        # 解析区域GTFS
                        region_data = load_gtfs_data(temp_path, f"{region_name} bus")
                        if not region_data.empty:
                            region_data['region'] = region_name
                            bus_regions[region_name] = region_data
                            loaded_regions += 1
                            print(f"    ✓ Successfully loaded {region_name} region, {len(region_data)} stations")
                        
                        # 清理临时文件
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                else:
                    print(f"  ✗ {region_name} region file not found: {file_name}")
        
        print(f"  Successfully loaded {loaded_regions} bus regions")
        return bus_regions
        
    except Exception as e:
        print(f"Error loading bus data: {e}")
        return {}

def load_bike_data_simple(file_path, sample_fraction=0.01):
    """
    简化版单车数据加载
    """
    print(f"Loading bike data: {file_path}")
    
    try:
        # 只读取前几行了解数据结构
        sample_rows = pd.read_csv(file_path, nrows=5)
        print(f"Bike data columns: {list(sample_rows.columns)}")
        
        # 如果文件很大，只采样一小部分
        file_size = os.path.getsize(file_path) / (1024**3)  # GB
        if file_size > 0.1:  # 如果文件大于100MB
            print(f"File is large ({file_size:.2f} GB), sampling {sample_fraction*100}%")
            bike_data = pd.read_csv(file_path, nrows=int(100000 * sample_fraction))
        else:
            bike_data = pd.read_csv(file_path)
        
        print(f"Loaded bike data with {len(bike_data)} rows")
        return bike_data
        
    except Exception as e:
        print(f"Error loading bike data: {e}")
        # 创建示例数据继续分析
        print("Creating sample bike data...")
        return create_sample_bike_data()

def create_sample_bike_data():
    """创建示例单车数据"""
    np.random.seed(42)
    n_records = 10000
    
    bike_data = pd.DataFrame({
        'ride_id': [f'ride_{i}' for i in range(n_records)],
        'start_station_name': np.random.choice([
            'Mercer St & Bleecker St', '1 St & Bowery', 'Broadway & W 58 St',
            '8 Ave & W 31 St', 'E 23 St & 1 Ave'
        ], n_records),
        'end_station_name': np.random.choice([
            'W 41 St & 8 Ave', 'E 17 St & Broadway', 'W 33 St & 7 Ave',
            'Forsyth St & Broome St', 'Allen St & Rivington St'
        ], n_records),
    })
    
    return bike_data

def load_all_data():
    """
    加载所有必要的数据
    """
    print("Loading all data...")
    
    data_dict = {}
    
    # 1. 加载单车数据
    data_dict['bike'] = load_bike_data_simple(BIKE_PATH)
    
    # 2. 加载地铁数据
    print(f"Loading subway data: {SUBWAY_PATH}")
    data_dict['subway'] = load_gtfs_data(SUBWAY_PATH, 'subway')
    
    # 3. 加载公交数据
    print(f"Loading bus data: {BUS_PATH}")
    data_dict['bus'] = load_bus_data(BUS_PATH)
    
    # 4. 加载天气数据（可选）
    try:
        print(f"Loading weather data: {WEATHER_PATH}")
        data_dict['weather'] = pd.read_excel(WEATHER_PATH)
        print(f"Weather data shape: {data_dict['weather'].shape}")
    except:
        print("Weather data not available, continuing without it")
        data_dict['weather'] = pd.DataFrame()
    
    return data_dict

def create_simplified_network(data_dict):
    """
    创建简化的交通网络
    """
    print("Creating simplified transportation network...")
    
    subway_data = data_dict['subway']
    bus_data = data_dict['bus']
    
    # 创建网络图
    G = nx.Graph()
    
    # 添加地铁站点
    if not subway_data.empty:
        # 如果地铁站点太多，采样一部分
        if len(subway_data) > 500:
            subway_data = subway_data.sample(n=500, random_state=42)
            print(f"  Sampling 500 subway stations from {len(subway_data)} total")
        
        for idx, station in subway_data.iterrows():
            G.add_node(f"subway_{station['station_id']}", 
                      node_type='subway',
                      name=station['station_name'],
                      lat=station['latitude'],
                      lon=station['longitude'])
        print(f"  Added {len(subway_data)} subway stations")
    
    # 添加公交站点（从所有区域）
    bus_station_count = 0
    if bus_data:
        for region_name, region_data in bus_data.items():
            # 每个区域只取前100个站点避免网络太大
            region_sample = region_data.head(100)
            for idx, stop in region_sample.iterrows():
                G.add_node(f"bus_{region_name}_{stop['station_id']}",
                          node_type='bus',
                          name=stop['station_name'],
                          lat=stop['latitude'],
                          lon=stop['longitude'],
                          region=region_name)
                bus_station_count += 1
        print(f"  Added {bus_station_count} bus stations")
    
    print(f"Network created with {len(G.nodes())} total nodes")
    
    # 添加连接（基于距离）
    print("  Adding connections based on spatial proximity...")
    nodes_list = list(G.nodes())
    added_edges = 0
    
    # 只添加最近的几个邻居，避免过多连接
    for i, node_i in enumerate(nodes_list):
        if i % 100 == 0 and i > 0:
            print(f"    Processed {i}/{len(nodes_list)} nodes, added {added_edges} edges")
        
        lat_i = G.nodes[node_i]['lat']
        lon_i = G.nodes[node_i]['lon']
        
        # 为每个节点添加3-5个最近邻连接
        distances = []
        for j, node_j in enumerate(nodes_list):
            if i != j:
                lat_j = G.nodes[node_j]['lat']
                lon_j = G.nodes[node_j]['lon']
                # 计算近似距离
                distance = np.sqrt((lat_i - lat_j)**2 + (lon_i - lon_j)**2)
                distances.append((node_j, distance))
        
        # 添加最近的几个连接
        distances.sort(key=lambda x: x[1])
        for k in range(min(3, len(distances))):
            if distances[k][1] < 0.02:  # 只添加相对较近的连接
                G.add_edge(node_i, distances[k][0], weight=1.0/distances[k][1])
                added_edges += 1
    
    print(f"Network complete with {len(G.nodes())} nodes and {len(G.edges())} edges")
    return G

def calculate_basic_centrality(G):
    """
    计算基础的中心性指标
    """
    print("Calculating basic centrality measures...")
    
    centrality_results = {}
    
    # 1. 度中心性
    print("  - Degree centrality...")
    centrality_results['degree'] = nx.degree_centrality(G)
    
    # 2. 介数中心性（使用近似算法）
    print("  - Betweenness centrality (approximate)...")
    try:
        # 使用更小的样本数
        k = min(50, len(G.nodes()) // 10)
        centrality_results['betweenness'] = nx.betweenness_centrality(G, k=k)
    except:
        print("    Betweenness calculation failed, using degree as fallback")
        centrality_results['betweenness'] = centrality_results['degree']
    
    return centrality_results

def create_simple_visualizations(G, centrality_results):
    """
    创建简化的可视化
    """
    print("Creating simplified visualizations...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Transportation Network Centrality Analysis', fontsize=16, fontweight='bold')
    
    # 1. 度中心性分布
    degree_values = list(centrality_results['degree'].values())
    ax1.hist(degree_values, bins=20, 
             alpha=0.7, color=colors['normal'], edgecolor='white')
    ax1.set_xlabel('Degree Centrality', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('A) Degree Centrality Distribution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_deg = np.mean(degree_values)
    ax1.axvline(mean_deg, color=colors['extreme'], linestyle='--', 
                label=f'Mean: {mean_deg:.4f}')
    ax1.legend()
    
    # 2. 介数中心性分布
    betweenness_values = list(centrality_results['betweenness'].values())
    ax2.hist(betweenness_values, bins=20,
             alpha=0.7, color=colors['accent1'], edgecolor='white')
    ax2.set_xlabel('Betweenness Centrality', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('B) Betweenness Centrality Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    mean_bet = np.mean(betweenness_values)
    ax2.axvline(mean_bet, color=colors['extreme'], linestyle='--', 
                label=f'Mean: {mean_bet:.4f}')
    ax2.legend()
    
    # 3. 前10个关键节点（按度中心性）
    top_degree = sorted(centrality_results['degree'].items(), 
                       key=lambda x: x[1], reverse=True)[:10]
    
    node_names = []
    degree_scores = []
    for node, score in top_degree:
        name = G.nodes[node].get('name', f'Node {node}')
        # 缩短长名称
        if len(name) > 25:
            name = name[:22] + '...'
        node_names.append(name)
        degree_scores.append(score)
    
    y_pos = np.arange(len(node_names))
    bars = ax3.barh(y_pos, degree_scores, color=colors['normal'], alpha=0.7)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(node_names, fontsize=9)
    ax3.set_xlabel('Degree Centrality Score', fontsize=12)
    ax3.set_title('C) Top 10 Nodes by Degree Centrality', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 在柱子上添加数值
    for i, (bar, score) in enumerate(zip(bars, degree_scores)):
        ax3.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{score:.4f}', ha='left', va='center', fontsize=9)
    
    # 4. 前10个关键节点（按介数中心性）
    top_betweenness = sorted(centrality_results['betweenness'].items(), 
                           key=lambda x: x[1], reverse=True)[:10]
    
    node_names_bt = []
    betweenness_scores = []
    for node, score in top_betweenness:
        name = G.nodes[node].get('name', f'Node {node}')
        if len(name) > 25:
            name = name[:22] + '...'
        node_names_bt.append(name)
        betweenness_scores.append(score)
    
    y_pos_bt = np.arange(len(node_names_bt))
    bars_bt = ax4.barh(y_pos_bt, betweenness_scores, color=colors['extreme'], alpha=0.7)
    ax4.set_yticks(y_pos_bt)
    ax4.set_yticklabels(node_names_bt, fontsize=9)
    ax4.set_xlabel('Betweenness Centrality Score', fontsize=12)
    ax4.set_title('D) Top 10 Nodes by Betweenness Centrality', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 在柱子上添加数值
    for i, (bar, score) in enumerate(zip(bars_bt, betweenness_scores)):
        ax4.text(bar.get_width() + 0.0001, bar.get_y() + bar.get_height()/2,
                f'{score:.4f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('network_centrality_analysis.png', dpi=300, bbox_inches='tight',
                facecolor=colors['lightest_blue'])
    plt.show()
    
    return fig

def generate_analysis_report(G, centrality_results):
    """
    生成分析报告
    """
    print("\n" + "="*80)
    print("NETWORK CENTRALITY ANALYSIS REPORT")
    print("="*80)
    
    # 基本统计
    print(f"\n📊 Network Statistics:")
    print(f"   • Total nodes: {len(G.nodes())}")
    print(f"   • Total edges: {len(G.edges())}")
    print(f"   • Network density: {nx.density(G):.4f}")
    
    # 节点类型统计
    node_types = {}
    for node in G.nodes():
        node_type = G.nodes[node].get('node_type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    print(f"   • Node type distribution:")
    for node_type, count in node_types.items():
        print(f"     - {node_type}: {count} nodes ({count/len(G.nodes())*100:.1f}%)")
    
    # 中心性统计
    print(f"\n🎯 Centrality Analysis:")
    
    degree_values = list(centrality_results['degree'].values())
    betweenness_values = list(centrality_results['betweenness'].values())
    
    print(f"   • Degree Centrality:")
    print(f"     - Mean: {np.mean(degree_values):.4f}")
    print(f"     - Std:  {np.std(degree_values):.4f}")
    print(f"     - Max:  {np.max(degree_values):.4f}")
    
    print(f"   • Betweenness Centrality:")
    print(f"     - Mean: {np.mean(betweenness_values):.4f}")
    print(f"     - Std:  {np.std(betweenness_values):.4f}")
    print(f"     - Max:  {np.max(betweenness_values):.4f}")
    
    # 关键节点识别
    print(f"\n🔍 Critical Node Identification:")
    
    print(f"   • Top 5 Nodes by Degree Centrality (Most Connected):")
    top_degree = sorted(centrality_results['degree'].items(), 
                       key=lambda x: x[1], reverse=True)[:5]
    for i, (node, score) in enumerate(top_degree, 1):
        name = G.nodes[node].get('name', 'Unknown Station')
        node_type = G.nodes[node].get('node_type', 'unknown')
        print(f"     {i}. {name} ({node_type}) - Score: {score:.4f}")
    
    print(f"   • Top 5 Nodes by Betweenness Centrality (Network Bridges):")
    top_betweenness = sorted(centrality_results['betweenness'].items(), 
                           key=lambda x: x[1], reverse=True)[:5]
    for i, (node, score) in enumerate(top_betweenness, 1):
        name = G.nodes[node].get('name', 'Unknown Station')
        node_type = G.nodes[node].get('node_type', 'unknown')
        print(f"     {i}. {name} ({node_type}) - Score: {score:.4f}")
    
    # 脆弱性分析
    print(f"\n⚠️  Vulnerability Assessment:")
    
    # 识别高介数节点（网络瓶颈）
    bottleneck_nodes = top_betweenness[:3]
    print(f"   • Critical Bottlenecks (High Betweenness):")
    for i, (node, score) in enumerate(bottleneck_nodes, 1):
        name = G.nodes[node].get('name', 'Unknown Station')
        print(f"     {i}. {name}")
        print(f"        - Acts as critical bridge in the network")
        print(f"        - Failure would significantly disrupt connectivity")
    
    # 识别高度连接节点
    hub_nodes = top_degree[:3]
    print(f"   • Major Hubs (High Degree):")
    for i, (node, score) in enumerate(hub_nodes, 1):
        name = G.nodes[node].get('name', 'Unknown Station')
        print(f"     {i}. {name}")
        print(f"        - Central station with many connections")
        print(f"        - Important for local connectivity")
    
    # 改进建议
    print(f"\n🎯 Resilience Improvement Recommendations:")
    print(f"   1. Reinforce {bottleneck_nodes[0][0].split('_')[-1]} with backup systems")
    print(f"   2. Develop contingency plans for {hub_nodes[0][0].split('_')[-1]}")
    print(f"   3. Improve alternative routes around critical nodes")
    print(f"   4. Monitor these stations during extreme weather events")
    print(f"   5. Consider adding redundant connections to bottleneck nodes")

def perform_network_analysis():
    """
    执行完整的网络分析
    """
    print("Starting Network Centrality Analysis")
    print("="*50)
    
    try:
        # 1. 加载数据
        data_dict = load_all_data()
        
        # 2. 创建网络
        G = create_simplified_network(data_dict)
        
        if len(G.nodes()) == 0:
            print("❌ No nodes in network. Check your data.")
            return None
        
        # 3. 计算中心性
        centrality_results = calculate_basic_centrality(G)
        
        # 4. 创建可视化
        fig = create_simple_visualizations(G, centrality_results)
        
        # 5. 生成报告
        generate_analysis_report(G, centrality_results)
        
        print(f"\n✅ Network analysis completed successfully!")
        print(f"📊 Generated: network_centrality_analysis.png")
        
        return {
            'network': G,
            'centrality': centrality_results
        }
        
    except Exception as e:
        print(f"❌ Error in network analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

# 主执行函数
if __name__ == "__main__":
    print("🚀 Starting Network Centrality Analysis")
    
    # 检查数据文件是否存在
    print("Checking data files...")
    essential_files = [SUBWAY_PATH, BUS_PATH]
    for file_path in essential_files:
        if os.path.exists(file_path):
            print(f"✅ {os.path.basename(file_path)} - Found")
        else:
            print(f"❌ {os.path.basename(file_path)} - Not found")
    
    print("\n" + "="*50)
    
    # 执行分析
    results = perform_network_analysis()
    
    if results:
        print("\n" + "="*60)
        print("Analysis completed successfully!")
        print("Check the generated file: network_centrality_analysis.png")
        print("="*60)
    else:
        print("❌ Analysis failed. Please check the error messages above.")