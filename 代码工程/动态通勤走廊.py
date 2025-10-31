import pandas as pd
import folium
import os

def load_data():
    file_path = r"E:/10-大三上/1-学习/1-信管/3-大数据系统原理与应用/期中作业/bike.csv"
    if not os.path.exists(file_path):
        print("文件不存在，请检查路径")
        return None
    
    try:
        bike_data = pd.read_csv(file_path, low_memory=False)
        print(f"数据加载成功: {bike_data.shape}")
        return bike_data
    except:
        print("读取文件失败")
        return None

def create_simple_map(bike_data, corridors_count=30, stations_count=20, map_name="simple"):
    bike_data_clean = bike_data.dropna(subset=['start_lat', 'start_lng', 'end_lat', 'end_lng'])
    bike_data_clean = bike_data_clean[
        (bike_data_clean['start_lat'].between(40.4, 41.0)) & 
        (bike_data_clean['start_lng'].between(-74.3, -73.6))
    ]
    
    center_lat = bike_data_clean['start_lat'].mean()
    center_lng = bike_data_clean['start_lng'].mean()
    
    m = folium.Map(location=[center_lat, center_lng], zoom_start=12)
    
    corridors = bike_data_clean.groupby([
        'start_station_name', 'end_station_name', 'start_lat', 'start_lng', 'end_lat', 'end_lng'
    ]).size().reset_index(name='trip_count')
    corridors = corridors.sort_values('trip_count', ascending=False).head(corridors_count)
    
    for idx, row in corridors.iterrows():
        folium.PolyLine(
            locations=[[row['start_lat'], row['start_lng']], [row['end_lat'], row['end_lng']]],
            popup=f"{row['start_station_name']} → {row['end_station_name']} ({row['trip_count']}次)",
            color='red',
            weight=3,
            opacity=0.7
        ).add_to(m)
    
    start_stations = bike_data_clean.groupby(['start_station_name', 'start_lat', 'start_lng']).size().reset_index(name='count')
    end_stations = bike_data_clean.groupby(['end_station_name', 'end_lat', 'end_lng']).size().reset_index(name='count')
    
    stations = pd.concat([
        start_stations.rename(columns={'start_station_name': 'name', 'start_lat': 'lat', 'start_lng': 'lng'}),
        end_stations.rename(columns={'end_station_name': 'name', 'end_lat': 'lat', 'end_lng': 'lng'})
    ])
    stations = stations.groupby(['name', 'lat', 'lng'])['count'].sum().reset_index()
    stations = stations.sort_values('count', ascending=False).head(stations_count)
    
    for idx, row in stations.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lng']],
            radius=6,
            popup=f"{row['name']} ({row['count']}次)",
            color='blue',
            fill=True,
            fillOpacity=0.7
        ).add_to(m)
    
    m.save(f'nyc_commuting_{map_name}.html')
    print(f"地图已保存: nyc_commuting_{map_name}.html")

def main():
    print("开始分析NYC通勤数据...")
    
    bike_data = load_data()
    if bike_data is None:
        return
    
    print("生成简洁版地图...")
    create_simple_map(bike_data, 20, 15, "simple")
    
    print("生成标准版地图...")
    create_simple_map(bike_data, 40, 25, "standard")
    
    print("生成详细版地图...")
    create_simple_map(bike_data, 60, 35, "detailed")
    
    print("生成完整版地图...")
    create_simple_map(bike_data, 80, 50, "full")
    
    selector_html = '''
    <!DOCTYPE html>
    <html>
    <head><title>NYC通勤地图选择</title></head>
    <body>
        <h2>🗽 NYC通勤地图选择</h2>
        <p><a href="nyc_commuting_simple.html" target="_blank">简洁版 (20走廊, 15站点)</a></p>
        <p><a href="nyc_commuting_standard.html" target="_blank">标准版 (40走廊, 25站点)</a></p>
        <p><a href="nyc_commuting_detailed.html" target="_blank">详细版 (60走廊, 35站点)</a></p>
        <p><a href="nyc_commuting_full.html" target="_blank">完整版 (80走廊, 50站点)</a></p>
    </body>
    </html>
    '''
    
    with open('nyc_commuting_selector.html', 'w', encoding='utf-8') as f:
        f.write(selector_html)
    
    print("选择页面已保存: nyc_commuting_selector.html")
    print("分析完成！请打开 nyc_commuting_selector.html 选择查看地图")

if __name__ == "__main__":
    main()