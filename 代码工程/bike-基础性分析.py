import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置图形样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class EfficientBikeDataAnalyzer:
    def __init__(self, data_path, sample_frac=None, random_state=42):
        """
        初始化分析器，支持抽样
        """
        print("Loading data...")
        
        # 只读取需要的列，减少内存占用
        usecols = ['started_at', 'member_casual', 'start_station_name', 'end_station_name']
        
        if sample_frac:
            # 抽样读取数据
            self.df = pd.read_csv(data_path, usecols=usecols).sample(frac=sample_frac, random_state=random_state)
            print(f"Sampling completed, total {len(self.df):,} records (sampling rate: {sample_frac*100}%)")
            self.analysis_mode = "sampling"
        else:
            # 全量读取，但只读取需要的列
            self.df = pd.read_csv(data_path, usecols=usecols, low_memory=False)
            print(f"Full data loading completed, total {len(self.df):,} records")
            self.analysis_mode = "full"
        
        # 转换时间列并提取时间特征
        print("Processing time features...")
        self.df['started_at'] = pd.to_datetime(self.df['started_at'])
        self.df['start_hour'] = self.df['started_at'].dt.hour
        self.df['start_dayofweek'] = self.df['started_at'].dt.dayofweek
        self.df['start_month'] = self.df['started_at'].dt.month
        self.df['is_weekend'] = self.df['start_dayofweek'].isin([5, 6]).astype(int)
        
        print(f"Time range: {self.df['started_at'].min()} to {self.df['started_at'].max()}")
        
        # 内存优化：删除原始时间列，只保留提取的特征
        self.df.drop('started_at', axis=1, inplace=True)
        
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    def _get_analysis_data(self, sample_size=50000):
        """
        根据分析模式获取数据
        """
        if self.analysis_mode == "sampling":
            # 抽样模式：使用抽样数据
            if len(self.df) > sample_size:
                return self.df.sample(n=sample_size, random_state=42)
            else:
                return self.df
        else:
            # 全量模式：直接使用所有数据
            return self.df
    
    def efficient_temporal_analysis(self):
        """
        高效的时间维度分析
        """
        print("\n=== Temporal Analysis ===")
        
        # 根据模式选择数据
        analysis_data = self._get_analysis_data(50000 if self.analysis_mode == "sampling" else None)
        
        # 对于全量数据，使用聚合来减少绘图数据量
        if self.analysis_mode == "full":
            print("Using aggregated data for visualization...")
            # 小时分布聚合
            hourly_data = analysis_data['start_hour'].value_counts().sort_index()
            # 按用户类型的小时分布聚合
            hourly_by_user = analysis_data.groupby(['start_hour', 'member_casual']).size().unstack()
            # 星期分布聚合
            daily_data = analysis_data['start_dayofweek'].value_counts().sort_index()
            # 月份分布聚合
            monthly_data = analysis_data['start_month'].value_counts().sort_index()
        else:
            # 抽样模式直接使用数据
            hourly_data = analysis_data['start_hour'].value_counts().sort_index()
            hourly_by_user = analysis_data.groupby(['start_hour', 'member_casual']).size().unstack()
            daily_data = analysis_data['start_dayofweek'].value_counts().sort_index()
            monthly_data = analysis_data['start_month'].value_counts().sort_index()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Temporal Analysis', fontsize=16)
        
        # 1. 小时分布
        axes[0,0].plot(hourly_data.index, hourly_data.values, marker='o', linewidth=2)
        axes[0,0].set_title('Hourly Ride Distribution')
        axes[0,0].set_xlabel('Hour')
        axes[0,0].set_ylabel('Ride Count')
        axes[0,0].grid(True)
        
        # 2. 按用户类型的小时分布
        hourly_by_user.plot(ax=axes[0,1], linewidth=2)
        axes[0,1].set_title('Hourly Distribution by User Type')
        axes[0,1].set_xlabel('Hour')
        axes[0,1].set_ylabel('Ride Count')
        axes[0,1].legend(title='User Type')
        
        # 3. 星期分布
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[1,0].bar(daily_data.index, daily_data.values, color='skyblue', alpha=0.7)
        axes[1,0].set_title('Weekly Distribution')
        axes[1,0].set_xlabel('Day of Week')
        axes[1,0].set_ylabel('Ride Count')
        axes[1,0].set_xticks(range(7))
        axes[1,0].set_xticklabels(day_names)
        
        # 4. 月份分布
        axes[1,1].bar(monthly_data.index, monthly_data.values, color='lightcoral', alpha=0.7)
        axes[1,1].set_title('Monthly Distribution')
        axes[1,1].set_xlabel('Month')
        axes[1,1].set_ylabel('Ride Count')
        
        plt.tight_layout()
        plt.show()
        
        # 时间分析洞察（使用全量数据计算）
        peak_hour = self.df['start_hour'].mode()[0] if len(self.df['start_hour'].mode()) > 0 else 0
        peak_day = self.df['start_dayofweek'].mode()[0] if len(self.df['start_dayofweek'].mode()) > 0 else 0
        
        print(f"Peak Hour: {peak_hour}:00")
        print(f"Peak Day: {day_names[peak_day]}")
        print(f"Weekend Ratio: {self.df['is_weekend'].mean()*100:.1f}%")
    
    def efficient_user_analysis(self):
        """
        高效的用户行为分析
        """
        print("\n=== User Behavior Analysis ===")
        
        # 用户类型分布（使用全量数据）
        user_dist = self.df['member_casual'].value_counts()
        weekend_by_user = self.df.groupby('member_casual')['is_weekend'].mean()
        
        # 对于可视化，使用抽样数据
        viz_data = self._get_analysis_data(30000 if self.analysis_mode == "sampling" else 50000)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('User Behavior Analysis', fontsize=16)
        
        # 1. 用户类型分布
        axes[0,0].bar(user_dist.index, user_dist.values, color=['blue', 'orange'], alpha=0.7)
        axes[0,0].set_title('User Type Distribution')
        axes[0,0].set_ylabel('User Count')
        
        # 2. 用户类型周末对比
        axes[0,1].bar(weekend_by_user.index, weekend_by_user.values * 100, color=['blue', 'orange'], alpha=0.7)
        axes[0,1].set_title('Weekend Usage Ratio')
        axes[0,1].set_ylabel('Weekend Usage (%)')
        
        # 3. 小时分布按用户类型（抽样可视化）
        hourly_by_user_viz = viz_data.groupby(['start_hour', 'member_casual']).size().unstack()
        hourly_by_user_viz.plot(ax=axes[1,0], linewidth=2)
        axes[1,0].set_title('Hourly Distribution by User Type (Sample)')
        axes[1,0].set_xlabel('Hour')
        axes[1,0].set_ylabel('Ride Count')
        axes[1,0].legend(title='User Type')
        
        # 4. 星期分布按用户类型（抽样可视化）
        daily_by_user = viz_data.groupby(['start_dayofweek', 'member_casual']).size().unstack()
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        x_pos = np.arange(7)
        bar_width = 0.35
        
        axes[1,1].bar(x_pos - bar_width/2, daily_by_user.iloc[:, 0], bar_width, 
                     label=daily_by_user.columns[0], alpha=0.7)
        axes[1,1].bar(x_pos + bar_width/2, daily_by_user.iloc[:, 1], bar_width, 
                     label=daily_by_user.columns[1], alpha=0.7)
        axes[1,1].set_title('Weekly Distribution by User Type (Sample)')
        axes[1,1].set_xlabel('Day of Week')
        axes[1,1].set_ylabel('Ride Count')
        axes[1,1].set_xticks(x_pos)
        axes[1,1].set_xticklabels(day_names)
        axes[1,1].legend(title='User Type')
        
        plt.tight_layout()
        plt.show()
        
        # 用户行为洞察（使用全量数据）
        print("User Type Distribution:")
        for user_type, count in user_dist.items():
            percentage = count / len(self.df) * 100
            print(f"  {user_type}: {count:,} ({percentage:.1f}%)")
        
        print(f"\nWeekend Usage by User Type:")
        for user_type, ratio in weekend_by_user.items():
            print(f"  {user_type}: {ratio*100:.1f}%")
    
    def efficient_spatial_analysis(self, top_n=15):
        """
        高效的空间分析（只分析TOP站点）
        """
        print("\n=== Spatial Analysis ===")
        
        # 使用全量数据计算热门站点
        print("Calculating popular stations...")
        top_start_stations = self.df['start_station_name'].value_counts().head(top_n)
        top_end_stations = self.df['end_station_name'].value_counts().head(top_n)
        
        print(f"Top {top_n} Start Stations:")
        for i, (station, count) in enumerate(top_start_stations.items(), 1):
            print(f"  {i:2d}. {station}: {count:,}")
        
        # 创建空间分析图表
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # 起始站TOP N
        top_start_stations.sort_values(ascending=True).plot(kind='barh', ax=axes[0], color='lightblue')
        axes[0].set_title(f'Top {top_n} Start Stations')
        axes[0].set_xlabel('Ride Count')
        
        # 终点站TOP N
        top_end_stations.sort_values(ascending=True).plot(kind='barh', ax=axes[1], color='lightcoral')
        axes[1].set_title(f'Top {top_n} End Stations')
        axes[1].set_xlabel('Ride Count')
        
        plt.tight_layout()
        plt.show()
    
    def basic_statistics(self):
        """
        基础统计信息
        """
        print("\n=== Basic Data Overview ===")
        print(f"Total Records: {len(self.df):,}")
        print(f"Analysis Mode: {self.analysis_mode.upper()}")
        
        print("\nUser Type Distribution:")
        user_dist = self.df['member_casual'].value_counts()
        for user_type, count in user_dist.items():
            percentage = count / len(self.df) * 100
            print(f"  {user_type}: {count:,} ({percentage:.1f}%)")
        
        print("\nTime Feature Statistics:")
        print(f"  Hours range: {self.df['start_hour'].min():02d}:00 - {self.df['start_hour'].max():02d}:00")
        print(f"  Months range: {self.df['start_month'].min()} - {self.df['start_month'].max()}")
        print(f"  Weekend rides: {self.df['is_weekend'].sum():,} ({self.df['is_weekend'].mean()*100:.1f}%)")
        
        print(f"\nMemory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    def run_complete_analysis(self, spatial_top_n=15):
        """
        运行完整分析
        """
        print("=" * 50)
        print("Bike Sharing Data Analysis Started")
        print(f"Mode: {self.analysis_mode.upper()}")
        print("=" * 50)
        
        self.basic_statistics()
        self.efficient_temporal_analysis()
        self.efficient_user_analysis()
        self.efficient_spatial_analysis(top_n=spatial_top_n)
        
        print("\n🎉 Analysis Completed!")

def analyze_with_sampling():
    """
    抽样分析模式 - 快速测试
    """
    print("🚀 Sampling Analysis Mode (Quick Test)")
    DATA_PATH = "E:/10-大三上/1-学习/1-信管/3-大数据系统原理与应用/期中作业/bike.csv"
    
    analyzer = EfficientBikeDataAnalyzer(DATA_PATH, sample_frac=0.1)  # 10%抽样
    analyzer.run_complete_analysis(spatial_top_n=10)

def analyze_full_data():
    """
    全量数据分析模式 - 真正使用所有数据
    """
    print("📊 Full Data Analysis Mode")
    DATA_PATH = "E:/10-大三上/1-学习/1-信管/3-大数据系统原理与应用/期中作业/bike.csv"
    
    # 内存使用警告
    print("⚠️  Warning: Full data analysis may use significant memory")
    print("   Optimizations applied to reduce memory usage")
    
    analyzer = EfficientBikeDataAnalyzer(DATA_PATH, sample_frac=None)  # 全量数据
    analyzer.run_complete_analysis(spatial_top_n=15)

def main():
    """
    主函数 - 选择分析模式
    """
    print("Select Analysis Mode:")
    print("1. Sampling Mode (Recommended for testing, 10% data)")
    print("2. Full Data Mode (Uses 100% data for analysis)")
    print("   Note: Full mode uses aggregated data for visualization")
    print("         but all calculations are based on complete dataset")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        analyze_with_sampling()
    elif choice == "2":
        analyze_full_data()
    else:
        print("Invalid choice, defaulting to sampling mode")
        analyze_with_sampling()

if __name__ == "__main__":
    main()