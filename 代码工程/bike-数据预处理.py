import pandas as pd
import numpy as np
from datetime import datetime
import os
import glob
import random

def process_data_with_sampling(csv_folder_path, output_path, sample_frac=None, chunksize=50000):
    """
    支持抽样和全量处理的数据处理函数
    """
    print("开始处理数据...")
    
    # 查找所有CSV文件
    csv_files = glob.glob(os.path.join(csv_folder_path, "*.csv"))
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    # 初始化计数器
    total_processed = 0
    total_cleaned = 0
    chunk_count = 0
    
    # 创建输出文件，先写入header
    first_chunk = True
    
    for file_idx, file in enumerate(csv_files):
        print(f"\n处理文件 {file_idx+1}/{len(csv_files)}: {os.path.basename(file)}")
        
        # 分块读取文件
        for chunk_idx, chunk in enumerate(pd.read_csv(file, chunksize=chunksize, low_memory=False)):
            # 如果启用抽样，在块级别进行抽样
            if sample_frac is not None:
                # 确保抽样后的块至少有一些数据
                if len(chunk) > 10:  # 只有块足够大时才抽样
                    sample_size = max(1, int(len(chunk) * sample_frac))
                    chunk = chunk.sample(n=sample_size, random_state=42)
            
            chunk_count += 1
            total_processed += len(chunk)
            
            print(f"  处理块 {chunk_idx+1}, 当前块大小: {len(chunk):,}")
            
            # 处理当前数据块
            cleaned_chunk = process_single_chunk(chunk)
            total_cleaned += len(cleaned_chunk)
            
            # 如果清洗后的块不为空，写入文件
            if len(cleaned_chunk) > 0:
                if first_chunk:
                    cleaned_chunk.to_csv(output_path, index=False, mode='w')
                    first_chunk = False
                else:
                    cleaned_chunk.to_csv(output_path, index=False, mode='a', header=False)
            
            # 每处理5个块输出一次进度
            if chunk_count % 5 == 0:
                print(f"    已处理 {chunk_count} 个块，总记录: {total_processed:,}，保留: {total_cleaned:,}")
                
            # 如果启用了抽样且已经处理了足够的数据，可以提前退出
            if sample_frac is not None and total_processed >= 100000:
                print(f"抽样数据量已达到 {total_processed:,}，提前结束处理")
                break
                
        # 抽样模式下，如果已经处理了足够数据，提前结束文件循环
        if sample_frac is not None and total_processed >= 100000:
            break
    
    print(f"\n=== 处理完成 ===")
    print(f"总处理记录: {total_processed:,}")
    print(f"清洗后记录: {total_cleaned:,}")
    print(f"数据保留率: {total_cleaned/total_processed*100:.2f}%")
    
    return total_processed, total_cleaned

def process_single_chunk(chunk):
    """
    处理单个数据块
    """
    # 1. 数据类型转换
    chunk['started_at'] = pd.to_datetime(chunk['started_at'], errors='coerce')
    chunk['ended_at'] = pd.to_datetime(chunk['ended_at'], errors='coerce')
    
    # 2. 计算骑行时长
    chunk['duration_minutes'] = (chunk['ended_at'] - chunk['started_at']).dt.total_seconds() / 60
    
    # 3. 删除关键字段缺失的记录
    critical_columns = ['started_at', 'ended_at', 'start_lat', 'start_lng', 'end_lat', 'end_lng']
    chunk_clean = chunk.dropna(subset=critical_columns)
    
    if len(chunk_clean) == 0:
        return pd.DataFrame()
    
    # 4. 过滤异常时长 (1分钟到24小时)
    chunk_clean = chunk_clean[
        (chunk_clean['duration_minutes'] >= 1) & 
        (chunk_clean['duration_minutes'] <= 24 * 60)
    ]
    
    if len(chunk_clean) == 0:
        return pd.DataFrame()
    
    # 5. 过滤异常坐标
    chunk_clean = chunk_clean[
        (chunk_clean['start_lat'].between(-90, 90)) &
        (chunk_clean['start_lng'].between(-180, 180)) &
        (chunk_clean['end_lat'].between(-90, 90)) &
        (chunk_clean['end_lng'].between(-180, 180))
    ]
    
    if len(chunk_clean) == 0:
        return pd.DataFrame()
    
    # 6. 提取时间特征
    chunk_clean['start_hour'] = chunk_clean['started_at'].dt.hour
    chunk_clean['start_dayofweek'] = chunk_clean['started_at'].dt.dayofweek
    chunk_clean['start_month'] = chunk_clean['started_at'].dt.month
    chunk_clean['start_date'] = chunk_clean['started_at'].dt.date
    chunk_clean['is_weekend'] = chunk_clean['start_dayofweek'].isin([5, 6])
    
    # 7. 计算简化距离（避免复杂计算节省内存）
    chunk_clean['distance_km'] = np.sqrt(
        (chunk_clean['end_lat'] - chunk_clean['start_lat'])**2 +
        (chunk_clean['end_lng'] - chunk_clean['start_lng'])**2
    ) * 111  # 大致转换为公里
    
    # 8. 过滤异常距离
    chunk_clean = chunk_clean[chunk_clean['distance_km'].between(0.01, 50)]
    
    # 9. 优化数据类型减少内存
    categorical_columns = ['rideable_type', 'member_casual']
    for col in categorical_columns:
        if col in chunk_clean.columns and chunk_clean[col].notna().any():
            chunk_clean[col] = chunk_clean[col].astype('category')
    
    return chunk_clean

def analyze_final_data(output_path, sample_size=100000):
    """
    分析最终清洗后的数据
    """
    print("\n正在分析最终数据...")
    
    try:
        # 读取数据进行分析
        if os.path.getsize(output_path) > 100 * 1024 * 1024:  # 如果文件大于100MB，只读取部分
            df_sample = pd.read_csv(output_path, nrows=sample_size)
            print(f"文件较大，仅读取前 {sample_size:,} 行进行分析")
        else:
            df_sample = pd.read_csv(output_path)
        
        print("\n=== 数据概览 ===")
        print(f"数据形状: {df_sample.shape}")
        print(f"列名: {list(df_sample.columns)}")
        
        print(f"\n用户类型分布:")
        print(df_sample['member_casual'].value_counts())
        
        print(f"\n车辆类型分布:")
        print(df_sample['rideable_type'].value_counts())
        
        if 'duration_minutes' in df_sample.columns:
            print(f"\n骑行时长统计:")
            print(f"  平均: {df_sample['duration_minutes'].mean():.2f} 分钟")
            print(f"  中位数: {df_sample['duration_minutes'].median():.2f} 分钟")
            print(f"  最大: {df_sample['duration_minutes'].max():.2f} 分钟")
            print(f"  最小: {df_sample['duration_minutes'].min():.2f} 分钟")
        
        if 'distance_km' in df_sample.columns:
            print(f"\n骑行距离统计:")
            print(f"  平均: {df_sample['distance_km'].mean():.2f} 公里")
            print(f"  中位数: {df_sample['distance_km'].median():.2f} 公里")
        
        print(f"\n时间范围:")
        if 'started_at' in df_sample.columns:
            # 转换回datetime用于分析
            df_sample['started_at'] = pd.to_datetime(df_sample['started_at'])
            print(f"  开始: {df_sample['started_at'].min()}")
            print(f"  结束: {df_sample['started_at'].max()}")
            
    except Exception as e:
        print(f"分析数据时出错: {e}")

def main():
    """
    主函数
    """
    # 配置参数 - 在这里切换模式！
    CSV_FOLDER_PATH = "E:/10-大三上/1-学习/1-信管/3-大数据系统原理与应用/期中作业/202506-citibike-tripdata"  # 修改为您的CSV文件所在文件夹路径
    OUTPUT_PATH = "E:/10-大三上/1-学习/1-信管/3-大数据系统原理与应用/期中作业/bike.csv"
    CHUNKSIZE = 50000  # 每个块的大小
    
    # === 选择运行模式 ===
    # MODE 1: 抽样测试 (推荐先运行这个)
    SAMPLE_FRAC = None  # 1% 的抽样率
    
    # MODE 2: 全量处理 (确认测试无误后使用)
    # SAMPLE_FRAC = None  # 处理全部数据
    
    print("=== 共享单车数据预处理 (抽样/全量可选版) ===")
    print(f"输入路径: {CSV_FOLDER_PATH}")
    print(f"输出路径: {OUTPUT_PATH}")
    print(f"块大小: {CHUNKSIZE:,}")
    
    if SAMPLE_FRAC is not None:
        print(f"运行模式: 抽样模式 ({SAMPLE_FRAC*100}% 数据)")
    else:
        print("运行模式: 全量模式")
    
    try:
        # 处理数据
        total_processed, total_cleaned = process_data_with_sampling(
            CSV_FOLDER_PATH, OUTPUT_PATH, SAMPLE_FRAC, CHUNKSIZE
        )
        
        # 分析结果
        analyze_final_data(OUTPUT_PATH)
        
        print(f"\n=== 处理完成 ===")
        print(f"输出文件: {OUTPUT_PATH}")
        file_size = os.path.getsize(OUTPUT_PATH) / (1024*1024)
        print(f"文件大小: {file_size:.2f} MB")
        
        # 使用建议
        if SAMPLE_FRAC is not None:
            print(f"\n💡 提示: 抽样测试成功！现在您可以修改 SAMPLE_FRAC = None 来运行全量数据")
        else:
            print(f"\n🎉 全量数据处理完成！")
        
    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()