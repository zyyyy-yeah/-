import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class EmergencyResponseAnalyzer:
    def __init__(self, bike_df, weather_df):
        """
        初始化应急响应分析器
        """
        # 选择需要的列以减少内存使用
        self.bike_df = bike_df[['started_at', 'member_casual', 'start_station_name', 
                               'end_station_name', 'duration_minutes', 'distance_km']].copy()
        self.weather_df = weather_df.copy()
        self.merged_df = None
        self.extreme_days = None
        
        # 定义统一颜色方案
        self.colors = {
            'normal': '#7895C1',      # 正常日
            'extreme': '#E3625D',     # 极端天气日
            'positive': '#992224',    # 正变化
            'negative': '#8074C8',    # 负变化
            'decrease': '#E3625D',    # 下降
            'increase': '#7895C1',    # 上升
            'accent1': '#F0C284',     # 强调色1
            'accent2': '#F5EBAE',     # 强调色2
            'light_blue': '#A8CBDF',
            'very_light_blue': '#D6EFF4',
            'lightest_blue': '#F2FAFC',
            'light_red': '#EF8B67',
            'medium_red': '#B54764',
            'dark_red': '#992224'
        }
        
    def prepare_data(self):
        """
        准备和合并数据
        """
        print("Preparing data...")
        
        # 从单车数据提取日期
        self.bike_df['date'] = self.bike_df['started_at'].dt.date
        
        # 计算每日骑行统计
        daily_stats = self.bike_df.groupby('date').agg({
            'duration_minutes': ['count', 'mean', 'median'],
            'distance_km': ['mean', 'median'],
            'member_casual': lambda x: (x == 'member').mean()  # 会员比例
        }).round(2)
        
        # 扁平化列名
        daily_stats.columns = ['daily_rides', 'avg_duration', 'median_duration', 
                              'avg_distance', 'median_distance', 'member_ratio']
        daily_stats = daily_stats.reset_index()
        
        # 准备天气数据
        self.weather_df['date'] = self.weather_df['DATE'].dt.date
        weather_cols = ['date', 'TMAX', 'PRCP']
        if 'pm25' in self.weather_df.columns:
            weather_cols.append('pm25')
        
        # 合并数据
        self.merged_df = daily_stats.merge(
            self.weather_df[weather_cols], 
            on='date', 
            how='inner'
        )
        
        print(f"Merged data: {len(self.merged_df)} days")
        return self.merged_df
    
    def identify_extreme_weather(self):
        """
        识别极端天气日
        """
        print("\n=== Identifying Extreme Weather Days ===")
        
        extreme_conditions = {}
        
        # 高温日 (TMAX > 95°F)
        if 'TMAX' in self.merged_df.columns:
            heat_days = self.merged_df[self.merged_df['TMAX'] > 95]
            extreme_conditions['heat_wave'] = heat_days['date'].tolist()
            print(f"Heat wave days (TMAX>95°F): {len(heat_days)} days")
        
        # 暴雨日 (PRCP > 10mm)
        if 'PRCP' in self.merged_df.columns:
            storm_days = self.merged_df[self.merged_df['PRCP'] > 10]
            extreme_conditions['heavy_rain'] = storm_days['date'].tolist()
            print(f"Heavy rain days (PRCP>10mm): {len(storm_days)} days")
        
        # 污染日 (pm25 > 150)
        if 'pm25' in self.merged_df.columns:
            pollution_days = self.merged_df[self.merged_df['pm25'] > 150]
            extreme_conditions['pollution'] = pollution_days['date'].tolist()
            print(f"Pollution days (PM2.5>150): {len(pollution_days)} days")
        
        # 合并所有极端天气日
        all_extreme_dates = []
        for condition, dates in extreme_conditions.items():
            all_extreme_dates.extend(dates)
        
        self.extreme_days = list(set(all_extreme_dates))
        print(f"Total extreme weather days: {len(self.extreme_days)} days")
        
        return extreme_conditions
    
    def compare_extreme_vs_normal(self):
        """
        对比极端天气日与正常日的骑行数据
        """
        print("\n=== Extreme Days vs Normal Days Comparison ===")
        
        if not self.extreme_days:
            print("No extreme weather days identified")
            return None
        
        # 标记极端天气日
        self.merged_df['is_extreme'] = self.merged_df['date'].isin(self.extreme_days)
        
        # 分组统计
        extreme_stats = self.merged_df[self.merged_df['is_extreme']].describe()
        normal_stats = self.merged_df[~self.merged_df['is_extreme']].describe()
        
        # 计算变化百分比
        comparison = {}
        metrics = ['daily_rides', 'avg_duration', 'median_duration', 
                  'avg_distance', 'median_distance', 'member_ratio']
        
        for metric in metrics:
            if metric in extreme_stats.columns and metric in normal_stats.columns:
                extreme_mean = extreme_stats.loc['mean', metric]
                normal_mean = normal_stats.loc['mean', metric]
                change_pct = ((extreme_mean - normal_mean) / normal_mean * 100) if normal_mean > 0 else 0
                
                comparison[metric] = {
                    'extreme': extreme_mean,
                    'normal': normal_mean,
                    'change_pct': change_pct
                }
                
                print(f"{metric}: Extreme {extreme_mean:.1f} vs Normal {normal_mean:.1f} | Change: {change_pct:+.1f}%")
        
        # 可视化对比
        self._plot_comparison(comparison)
        
        return comparison
    
    def _plot_comparison(self, comparison):
        """
        绘制对比图表
        """
        if not comparison:
            return
            
        metrics = list(comparison.keys())
        extreme_vals = [comparison[m]['extreme'] for m in metrics]
        normal_vals = [comparison[m]['normal'] for m in metrics]
        changes = [comparison[m]['change_pct'] for m in metrics]
        
        # 创建对比图表
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # 柱状图对比 - 使用统一配色
        x_pos = np.arange(len(metrics))
        width = 0.35
        
        axes[0].bar(x_pos - width/2, normal_vals, width, label='Normal Days', 
                   alpha=0.8, color=self.colors['normal'])
        axes[0].bar(x_pos + width/2, extreme_vals, width, label='Extreme Weather Days', 
                   alpha=0.8, color=self.colors['extreme'])
        axes[0].set_xlabel('Metrics')
        axes[0].set_ylabel('Values')
        axes[0].set_title('Riding Metrics: Extreme Weather vs Normal Days')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(['Ride Count', 'Avg Duration', 'Median Duration', 
                               'Avg Distance', 'Median Distance', 'Member Ratio'], rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (normal, extreme) in enumerate(zip(normal_vals, extreme_vals)):
            axes[0].text(i - width/2, normal + max(normal_vals)*0.01, f'{normal:.1f}', 
                        ha='center', va='bottom', fontsize=8)
            axes[0].text(i + width/2, extreme + max(extreme_vals)*0.01, f'{extreme:.1f}', 
                        ha='center', va='bottom', fontsize=8)
        
        # 变化百分比图表 - 使用统一配色
        bar_colors = [self.colors['positive'] if x > 0 else self.colors['negative'] for x in changes]
        bars = axes[1].bar(metrics, changes, color=bar_colors, alpha=0.8)
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1].set_xlabel('Metrics')
        axes[1].set_ylabel('Change Percentage (%)')
        axes[1].set_title('Percentage Change: Extreme Weather vs Normal Days')
        axes[1].set_xticklabels(['Ride Count', 'Avg Duration', 'Median Duration', 
                               'Avg Distance', 'Median Distance', 'Member Ratio'], rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        # 添加百分比标签
        for bar, change in zip(bars, changes):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (1 if bar.get_height() >=0 else -3), 
                        f'{change:+.1f}%', ha='center', va='bottom' if bar.get_height() >=0 else 'top', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_station_impact(self):
        """
        分析极端天气下站点使用率变化
        """
        print("\n=== Station Usage Impact Analysis ===")
        
        if not self.extreme_days:
            print("No extreme weather days data")
            return None
        
        # 标记极端天气日
        self.bike_df['is_extreme'] = self.bike_df['date'].isin(self.extreme_days)
        
        # 计算各站点在极端天气和正常日的使用量
        station_analysis = {}
        
        for station_type in ['start_station_name', 'end_station_name']:
            print(f"\nAnalyzing {station_type}...")
            
            # 极端天气日的站点使用量
            extreme_station_usage = self.bike_df[self.bike_df['is_extreme']].groupby(station_type).size()
            
            # 正常日的站点使用量
            normal_station_usage = self.bike_df[~self.bike_df['is_extreme']].groupby(station_type).size()
            
            # 合并计算变化率
            all_stations = set(extreme_station_usage.index) | set(normal_station_usage.index)
            station_changes = []
            
            for station in all_stations:
                extreme_count = extreme_station_usage.get(station, 0)
                normal_count = normal_station_usage.get(station, 0)
                
                if normal_count > 0:  # 避免除零
                    change_pct = ((extreme_count - normal_count) / normal_count * 100)
                    station_changes.append({
                        'station': station,
                        'extreme_usage': extreme_count,
                        'normal_usage': normal_count,
                        'change_pct': change_pct,
                        'total_usage': extreme_count + normal_count
                    })
            
            # 按变化率排序
            station_changes.sort(key=lambda x: x['change_pct'])
            
            # 找出受影响最大的站点（下降最多）
            most_decreased = station_changes[:10]  # 下降最多的10个站点
            most_increased = station_changes[-10:] # 上升最多的10个站点
            
            station_analysis[station_type] = {
                'most_decreased': most_decreased,
                'most_increased': most_increased,
                'all_changes': station_changes
            }
            
            # 打印结果
            print(f"Stations with largest decrease ({station_type}):")
            for i, station in enumerate(most_decreased):
                print(f"  {i+1:2d}. {station['station']}: {station['change_pct']:+.1f}%")
            
            print(f"Stations with largest increase ({station_type}):")
            for i, station in enumerate(reversed(most_increased)):
                print(f"  {i+1:2d}. {station['station']}: {station['change_pct']:+.1f}%")
        
        # 可视化站点影响
        self._plot_station_impact(station_analysis)
        
        return station_analysis
    
    def _plot_station_impact(self, station_analysis):
        """
        绘制站点影响图表 - 使用统一配色
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        for idx, station_type in enumerate(['start_station_name', 'end_station_name']):
            data = station_analysis[station_type]
            
            # 下降最多的站点 - 使用红色系
            decreased_stations = [s['station'] for s in data['most_decreased']]
            decreased_changes = [s['change_pct'] for s in data['most_decreased']]
            
            axes[0, idx].barh(decreased_stations, decreased_changes, 
                             color=self.colors['decrease'], alpha=0.8)
            axes[0, idx].set_title(f'{station_type.replace("_", " ").title()} - Stations with Largest Decrease')
            axes[0, idx].set_xlabel('Change Percentage (%)')
            axes[0, idx].grid(True, alpha=0.3)
            
            # 上升最多的站点 - 使用蓝色系
            increased_stations = [s['station'] for s in data['most_increased']]
            increased_changes = [s['change_pct'] for s in data['most_increased']]
            
            axes[1, idx].barh(increased_stations, increased_changes, 
                             color=self.colors['increase'], alpha=0.8)
            axes[1, idx].set_title(f'{station_type.replace("_", " ").title()} - Stations with Largest Increase')
            axes[1, idx].set_xlabel('Change Percentage (%)')
            axes[1, idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_analysis(self):
        """
        运行完整应急响应分析
        """
        print("Starting Emergency Response Analysis...")
        print("=" * 50)
        
        # 准备数据
        self.prepare_data()
        
        if self.merged_df is None or len(self.merged_df) == 0:
            print("Error: Data merge failed")
            return
        
        # 识别极端天气
        extreme_conditions = self.identify_extreme_weather()
        
        if not self.extreme_days:
            print("No qualified extreme weather days found")
            return
        
        # 对比分析
        comparison_results = self.compare_extreme_vs_normal()
        
        # 站点影响分析
        station_results = self.analyze_station_impact()
        
        print("\n🎉 Emergency Response Analysis Completed!")
        
        return {
            'extreme_conditions': extreme_conditions,
            'comparison': comparison_results,
            'station_impact': station_results
        }


def main():
    """
    主函数 - 使用示例
    """
    
    bike_df = pd.read_csv("E:/10-大三上/1-学习/1-信管/3-大数据系统原理与应用/期中作业/bike.csv")
    weather_df = pd.read_csv("E:/10-大三上/1-学习/1-信管/3-大数据系统原理与应用/期中作业/weather.csv")

    bike_df['started_at'] = pd.to_datetime(bike_df['started_at'])
    weather_df['DATE'] = pd.to_datetime(weather_df['DATE'])

    analyzer = EmergencyResponseAnalyzer(bike_df, weather_df)
    results = analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()