import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class WeatherImpactAnalyzer:
    def __init__(self, bike_df, weather_df):
        """
        初始化分析器 - 内存优化版本
        """
        # 只复制需要的列以减少内存使用
        self.bike_df = bike_df[['started_at', 'member_casual']].copy()
        self.weather_df = weather_df.copy()
        self.merged_df = None
        
        # 定义统一颜色方案
        self.colors = {
            'normal': '#7895C1',      # 蓝色系 - 正常/正面
            'extreme': '#E3625D',     # 红色系 - 极端/负面
            'positive': '#7895C1',    # 正相关/增长 - 蓝色
            'negative': '#E3625D',    # 负相关/下降 - 红色
            'decrease': '#E3625D',    # 下降
            'increase': '#7895C1',    # 上升
            'accent1': '#F0C284',     # 强调色1 - 金色
            'accent2': '#F5EBAE',     # 强调色2 - 浅金色
            'light_blue': '#A8CBDF',  # 浅蓝
            'very_light_blue': '#D6EFF4',  # 很浅的蓝
            'lightest_blue': '#F2FAFC',    # 最浅的蓝
            'light_red': '#EF8B67',   # 浅红
            'medium_red': '#B54764',  # 中红
            'dark_red': '#992224',    # 深红
            'purple': '#8074C8',      # 紫色
            'light_purple': '#A8A2D8' # 浅紫色
        }
        
    def prepare_data(self):
        """
        准备合并数据 - 内存优化
        """
        print("准备数据中...")
        
        # 从单车数据计算每日骑行量 - 使用更高效的方法
        self.bike_df['date'] = self.bike_df['started_at'].dt.date
        
        # 计算总骑行量
        daily_rides = self.bike_df.groupby('date').size().reset_index(name='daily_rides')
        
        # 计算用户类型骑行量 - 使用更高效的方法
        member_daily = self.bike_df[self.bike_df['member_casual'] == 'member'].groupby('date').size()
        casual_daily = self.bike_df[self.bike_df['member_casual'] == 'casual'].groupby('date').size()
        
        daily_rides = daily_rides.set_index('date')
        daily_rides['rides_member'] = member_daily
        daily_rides['rides_casual'] = casual_daily
        daily_rides = daily_rides.fillna(0).reset_index()
        
        # 清理内存
        del member_daily, casual_daily
        
        # 准备天气数据 - 只选择需要的列
        weather_cols = ['DATE']
        numeric_cols = ['TAVG', 'TMAX', 'TMIN', 'PRCP', 'AWND']
        
        # 只选择存在的列
        available_cols = [col for col in numeric_cols + ['pm25', 'SNOW'] if col in self.weather_df.columns]
        weather_cols.extend(available_cols)
        
        self.weather_df = self.weather_df[weather_cols].copy()
        self.weather_df['date'] = self.weather_df['DATE'].dt.date
        
        # 合并数据
        self.merged_df = daily_rides.merge(
            self.weather_df[['date'] + available_cols], 
            on='date', 
            how='inner'
        )
        
        print(f"合并后数据量: {len(self.merged_df)} 天")
        print(f"可用气象指标: {available_cols}")
        
        # 清理中间数据
        del daily_rides
        return self.merged_df
    
    def analyze_temperature_impact(self):
        """
        分析温度对骑行量的影响 - 华氏度版本
        """
        print("\n=== 温度影响分析 ===")
    
        if 'TAVG' not in self.merged_df.columns:
            print("缺少温度数据")
            return None
        
        # 华氏度温度分段 (基于常见的舒适度范围)
        temp_bins = [0, 32, 50, 60, 70, 80, 90, 100, 120]  # 华氏度
        temp_labels = ['极冷(<32°)', '寒冷(32-50°)', '凉爽(50-60°)', '舒适(60-70°)', 
                      '温暖(70-80°)', '较热(80-90°)', '炎热(90-100°)', '酷热(>100°)']
    
        self.merged_df['temp_category'] = pd.cut(
            self.merged_df['TAVG'], bins=temp_bins, labels=temp_labels
        )
    
        # 温度与骑行量的相关性
        corr_temp = self.merged_df['TAVG'].corr(self.merged_df['daily_rides'])
    
        print(f"温度与骑行量相关系数: {corr_temp:.3f}")
    
        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
        # 散点图 - 使用统一配色
        axes[0].scatter(self.merged_df['TAVG'], self.merged_df['daily_rides'], 
                   alpha=0.8, s=80, color=self.colors['normal'], 
                   edgecolors=self.colors['dark_red'], linewidth=0.5)
        axes[0].set_xlabel('Average Temperature (°F)')
        axes[0].set_ylabel('Daily Rides')
        axes[0].set_title(f'Temperature vs Rides (r={corr_temp:.3f})')
        axes[0].grid(True, alpha=0.3)
    
        # 分段柱状图 - 使用统一配色
        temp_avg_rides = self.merged_df.groupby('temp_category')['daily_rides'].mean()
        temp_avg_rides = temp_avg_rides.fillna(0)
        
        # 为不同温度段设置颜色
        temp_colors = [
            self.colors['light_blue'],    # 极冷
            self.colors['normal'],        # 寒冷
            self.colors['accent1'],       # 凉爽
            self.colors['increase'],      # 舒适
            self.colors['accent2'],       # 温暖
            self.colors['light_red'],     # 较热
            self.colors['extreme'],       # 炎热
            self.colors['dark_red']       # 酷热
        ]
    
        axes[1].bar(temp_avg_rides.index, temp_avg_rides.values, 
                   color=temp_colors[:len(temp_avg_rides)], alpha=0.8)
        axes[1].set_xlabel('Temperature Range (°F)')
        axes[1].set_ylabel('Average Daily Rides')
        axes[1].set_title('Average Rides by Temperature Range')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
    
        # 在柱状图上添加数值标签
        for i, v in enumerate(temp_avg_rides.values):
            axes[1].text(i, v + max(temp_avg_rides.values)*0.01, f'{v:.0f}', 
                        ha='center', va='bottom', fontsize=9)
    
        plt.tight_layout()
        plt.show()
    
        # 打印数据统计信息 - 显示华氏度范围
        print(f"温度范围: {self.merged_df['TAVG'].min():.1f}°F - {self.merged_df['TAVG'].max():.1f}°F")
        print(f"骑行量范围: {self.merged_df['daily_rides'].min():.0f} - {self.merged_df['daily_rides'].max():.0f}")
        print(f"有效数据天数: {len(self.merged_df)}")
    
        return temp_avg_rides
    
    def analyze_precipitation_impact(self):
        """
        分析降水影响
        """
        print("\n=== 降水影响分析 ===")
        
        if 'PRCP' not in self.merged_df.columns:
            print("缺少降水数据")
            return None
            
        # 降水分类
        precip_bins = [-1, 0, 1, 5, 10, 50]
        precip_labels = ['No Rain(0mm)', 'Light(0-1mm)', 'Moderate(1-5mm)', 'Heavy(5-10mm)', 'Storm(>10mm)']
        
        self.merged_df['precip_category'] = pd.cut(
            self.merged_df['PRCP'], bins=precip_bins, labels=precip_labels
        )
        
        # 雨天vs非雨天对比
        rainy_days = self.merged_df[self.merged_df['PRCP'] > 0]
        dry_days = self.merged_df[self.merged_df['PRCP'] == 0]
        
        rainy_avg = rainy_days['daily_rides'].mean() if len(rainy_days) > 0 else 0
        dry_avg = dry_days['daily_rides'].mean() if len(dry_days) > 0 else 0
        reduction_pct = ((dry_avg - rainy_avg) / dry_avg * 100) if dry_avg > 0 else 0
        
        print(f"Rainy days average rides: {rainy_avg:.0f}")
        print(f"Dry days average rides: {dry_avg:.0f}")
        print(f"Reduction on rainy days: {reduction_pct:.1f}%")
        
        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 降水分类对比 - 使用统一配色
        precip_avg = self.merged_df.groupby('precip_category')['daily_rides'].mean()
        precip_colors = [
            self.colors['normal'],        # 无雨
            self.colors['light_blue'],    # 小雨
            self.colors['accent1'],       # 中雨
            self.colors['light_red'],     # 大雨
            self.colors['extreme']        # 暴雨
        ]
        axes[0].bar(precip_avg.index, precip_avg.values, 
                   color=precip_colors[:len(precip_avg)], alpha=0.8)
        axes[0].set_xlabel('Precipitation Level')
        axes[0].set_ylabel('Average Daily Rides')
        axes[0].set_title('Impact of Precipitation on Rides')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # 雨天vs晴天分布 - 使用统一配色
        weather_types = ['Dry Days', 'Rainy Days']
        ride_means = [dry_avg, rainy_avg]
        weather_colors = [self.colors['normal'], self.colors['extreme']]
        axes[1].bar(weather_types, ride_means, color=weather_colors, alpha=0.8)
        axes[1].set_ylabel('Average Daily Rides')
        axes[1].set_title(f'Rainy vs Dry Days (Reduction: {reduction_pct:.1f}%)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return {
            'rainy_avg': rainy_avg,
            'dry_avg': dry_avg,
            'reduction_pct': reduction_pct
        }
    
    def analyze_snow_impact(self):
        """
        分析降雪影响
        """
        print("\n=== 降雪影响分析 ===")
        
        if 'SNOW' not in self.merged_df.columns:
            print("缺少降雪数据")
            return None
            
        # 降雪分类
        snow_days = self.merged_df[self.merged_df['SNOW'] > 0]
        no_snow_days = self.merged_df[self.merged_df['SNOW'] == 0]
        
        if len(snow_days) > 0:
            snow_avg = snow_days['daily_rides'].mean()
            no_snow_avg = no_snow_days['daily_rides'].mean()
            reduction_pct = ((no_snow_avg - snow_avg) / no_snow_avg * 100) if no_snow_avg > 0 else 0
            
            print(f"Snow days average rides: {snow_avg:.0f}")
            print(f"No snow days average rides: {no_snow_avg:.0f}")
            print(f"Reduction on snow days: {reduction_pct:.1f}%")
            
            # 可视化 - 使用统一配色
            plt.figure(figsize=(8, 6))
            conditions = ['No Snow', 'Snow']
            rides = [no_snow_avg, snow_avg]
            snow_colors = [self.colors['normal'], self.colors['extreme']]
            plt.bar(conditions, rides, color=snow_colors, alpha=0.8, edgecolor='black')
            plt.ylabel('Average Daily Rides')
            plt.title(f'Snow Impact on Rides (Reduction: {reduction_pct:.1f}%)')
            plt.grid(True, alpha=0.3)
            plt.show()
            
            return {
                'snow_avg': snow_avg,
                'no_snow_avg': no_snow_avg,
                'reduction_pct': reduction_pct
            }
        else:
            print("No snow days in dataset")
            return None
    
    def analyze_air_quality_impact(self):
        """
        分析空气质量影响
        """
        print("\n=== 空气质量影响分析 ===")
        
        if 'pm25' not in self.merged_df.columns:
            print("缺少空气质量数据")
            return None
            
        # 空气质量分段 (基于PM2.5)
        aqi_bins = [0, 35, 75, 115, 150, 500]
        aqi_labels = ['Excellent(0-35)', 'Good(35-75)', 'Light Polluted(75-115)', 
                     'Moderate Polluted(115-150)', 'Heavy Polluted(>150)']
        
        self.merged_df['aqi_category'] = pd.cut(
            self.merged_df['pm25'], bins=aqi_bins, labels=aqi_labels
        )
        
        # 空气质量与骑行量的相关性
        corr_aqi = self.merged_df['pm25'].corr(self.merged_df['daily_rides'])
        
        print(f"PM2.5与骑行量相关系数: {corr_aqi:.3f}")
        
        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 散点图 - 使用统一配色
        scatter_color = self.colors['negative'] if corr_aqi < 0 else self.colors['positive']
        axes[0].scatter(self.merged_df['pm25'], self.merged_df['daily_rides'], 
                       alpha=0.8, s=60, color=scatter_color, 
                       edgecolors=self.colors['dark_red'], linewidth=0.5)
        axes[0].set_xlabel('PM2.5 Concentration (μg/m³)')
        axes[0].set_ylabel('Daily Rides')
        axes[0].set_title(f'PM2.5 vs Rides (r={corr_aqi:.3f})')
        axes[0].grid(True, alpha=0.3)
        
        # 空气质量等级对比 - 使用统一配色
        aqi_avg_rides = self.merged_df.groupby('aqi_category')['daily_rides'].mean()
        aqi_colors = [
            self.colors['normal'],        # 优秀
            self.colors['light_blue'],    # 良好
            self.colors['accent2'],       # 轻度污染
            self.colors['light_red'],     # 中度污染
            self.colors['extreme']        # 重度污染
        ]
        axes[1].bar(aqi_avg_rides.index, aqi_avg_rides.values, 
                   color=aqi_colors[:len(aqi_avg_rides)], alpha=0.8)
        axes[1].set_xlabel('Air Quality Level')
        axes[1].set_ylabel('Average Daily Rides')
        axes[1].set_title('Air Quality Impact on Rides')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return aqi_avg_rides
    
    def analyze_wind_impact(self):
        """
        分析风速影响
        """
        print("\n=== 风速影响分析 ===")
        
        if 'AWND' not in self.merged_df.columns:
            print("缺少风速数据")
            return None
            
        # 风速分段
        wind_bins = [0, 5, 10, 15, 20, 50]
        wind_labels = ['Calm(0-5)', 'Light(5-10)', 'Moderate(10-15)', 'Strong(15-20)', 'Storm(>20)']
        
        self.merged_df['wind_category'] = pd.cut(
            self.merged_df['AWND'], bins=wind_bins, labels=wind_labels
        )
        
        # 风速与骑行量的相关性
        corr_wind = self.merged_df['AWND'].corr(self.merged_df['daily_rides'])
        
        print(f"风速与骑行量相关系数: {corr_wind:.3f}")
        
        # 可视化 - 使用统一配色
        wind_avg_rides = self.merged_df.groupby('wind_category')['daily_rides'].mean()
        
        # 根据相关性选择颜色
        wind_colors = [
            self.colors['normal'] if corr_wind >= 0 else self.colors['extreme'],
            self.colors['light_blue'] if corr_wind >= 0 else self.colors['light_red'],
            self.colors['accent1'],
            self.colors['light_red'] if corr_wind >= 0 else self.colors['light_blue'],
            self.colors['extreme'] if corr_wind >= 0 else self.colors['normal']
        ]
        
        plt.figure(figsize=(10, 6))
        plt.bar(wind_avg_rides.index, wind_avg_rides.values, 
               color=wind_colors[:len(wind_avg_rides)], alpha=0.8)
        plt.xlabel('Wind Speed Level')
        plt.ylabel('Average Daily Rides')
        plt.title(f'Wind Speed Impact on Rides (r={corr_wind:.3f})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return wind_avg_rides
    
    def comprehensive_analysis(self):
        """
        综合影响分析
        """
        print("\n=== 综合影响分析 ===")
        
        # 计算各因素与骑行量的相关系数
        factors = []
        correlations = []
        
        weather_factors = ['TAVG', 'PRCP', 'AWND']
        if 'pm25' in self.merged_df.columns:
            weather_factors.append('pm25')
        if 'SNOW' in self.merged_df.columns:
            weather_factors.append('SNOW')
        
        for factor in weather_factors:
            if factor in self.merged_df.columns:
                corr = self.merged_df[factor].corr(self.merged_df['daily_rides'])
                factors.append(factor)
                correlations.append(corr)
        
        # 创建相关性图表 - 使用统一配色
        plt.figure(figsize=(10, 6))
        bar_colors = [self.colors['positive'] if x > 0 else self.colors['negative'] for x in correlations]
        bars = plt.bar(factors, correlations, color=bar_colors, alpha=0.8)
        
        # 添加数值标签
        for bar, corr in zip(bars, correlations):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + (0.01 if height >=0 else -0.03), 
                    f'{corr:.3f}', ha='center', va='bottom' if height >=0 else 'top', fontsize=10)
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.xlabel('Weather Factors')
        plt.ylabel('Correlation with Rides')
        plt.title('Correlation between Weather Factors and Daily Rides')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # 输出总结
        print("\n📊 Analysis Summary:")
        for factor, corr in zip(factors, correlations):
            direction = "positive" if corr > 0 else "negative"
            strength = "strong" if abs(corr) > 0.5 else "moderate" if abs(corr) > 0.3 else "weak"
            print(f"  {factor}: {corr:.3f} ({strength} {direction} correlation)")
    
    def run_complete_analysis(self):
        """
        运行完整分析
        """
        print("开始天气对骑行量的影响分析...")
        print("=" * 50)
        
        # 准备数据
        self.prepare_data()
        
        if self.merged_df is None or len(self.merged_df) == 0:
            print("错误: 数据合并失败或没有重叠的日期")
            return
        
        # 执行各项分析
        results = {}
        
        results['temperature'] = self.analyze_temperature_impact()
        results['precipitation'] = self.analyze_precipitation_impact()
        results['snow'] = self.analyze_snow_impact()
        results['air_quality'] = self.analyze_air_quality_impact()
        results['wind'] = self.analyze_wind_impact()
        
        # 综合分析
        self.comprehensive_analysis()
        
        print("\n🎉 天气影响分析完成!")
        return results

# 使用示例
def main():
    bike_df = pd.read_csv("E:/10-大三上/1-学习/1-信管/3-大数据系统原理与应用/期中作业/bike.csv")  # 包含 started_at, member_casual 列
    weather_df = pd.read_csv("E:/10-大三上/1-学习/1-信管/3-大数据系统原理与应用/期中作业/weather.csv")  # 天气数据
    
    bike_df['started_at'] = pd.to_datetime(bike_df['started_at'])
    weather_df['DATE'] = pd.to_datetime(weather_df['DATE'])
    
    analyzer = WeatherImpactAnalyzer(bike_df, weather_df)
    results = analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()