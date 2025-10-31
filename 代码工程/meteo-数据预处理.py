import pandas as pd
import numpy as np
from datetime import datetime
import os

def process_weather_data():
    """
    处理天气数据，整合三个文件为一个包含2025年6月气象数据的CSV文件
    改进版：处理缺失的温度数据
    """
    # 设置文件保存路径
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    output_path = os.path.join(desktop_path, "weather_data_june_2025.csv")
    
    # 读取第一个文件（空气质量数据）
    print("正在读取空气质量数据...")
    df_air = pd.read_excel("E:/10-大三上/1-学习/1-信管/3-大数据系统原理与应用/期中作业/meteorologic/纽约-air-quality.xlsx")  # 请替换为实际文件名
    
    # 选择需要的列并重命名
    df_air = df_air[['date', 'pm25', 'o3', 'no2', 'co']].copy()
    df_air.rename(columns={'date': 'DATE'}, inplace=True)
    df_air['DATE'] = pd.to_datetime(df_air['DATE'])
    
    # 读取第二个文件（气象数据1）
    print("正在读取气象数据1...")
    df_weather1 = pd.read_excel("E:/10-大三上/1-学习/1-信管/3-大数据系统原理与应用/期中作业/meteorologic/nywind.xlsx")  # 请替换为实际文件名
    
    # 选择需要的列
    df_weather1 = df_weather1[['DATE', 'AWND', 'PGTM', 'TAVG', 'TMAX', 'TMIN', 
                              'WDF2', 'WSF2', 'WT01', 'WT02', 'WT03', 'WT08']].copy()
    df_weather1['DATE'] = pd.to_datetime(df_weather1['DATE'])
    
    # 读取第三个文件（气象数据2）
    print("正在读取气象数据2...")
    df_weather2 = pd.read_csv("E:/10-大三上/1-学习/1-信管/3-大数据系统原理与应用/期中作业/meteorologic/daily-summaries-2025-10-09T12-21-41.csv")  # 请替换为实际文件名
    
    # 选择需要的列
    df_weather2 = df_weather2[['DATE', 'PRCP', 'SNOW', 'TAVG', 'TMAX', 'TMIN']].copy()
    df_weather2['DATE'] = pd.to_datetime(df_weather2['DATE'])
    
    print("正在合并数据...")
    
    # 首先合并两个气象数据文件
    df_combined = pd.merge(df_weather1, df_weather2, on='DATE', how='outer', suffixes=('', '_dup'))
    
    # 处理重复的温度列 - 改进策略
    print("处理重复和缺失的温度数据...")
    
    # 对于每个温度列，使用最佳可用数据
    for temp_col in ['TAVG', 'TMAX', 'TMIN']:
        dup_col = f'{temp_col}_dup'
        
        if dup_col in df_combined.columns:
            # 策略1: 优先使用第一个文件的温度数据
            # 策略2: 如果第一个文件缺失，使用第二个文件的数据
            df_combined[temp_col] = df_combined[temp_col].fillna(df_combined[dup_col])
            
            # 策略3: 如果两个文件都缺失，尝试用其他温度数据推算
            if temp_col == 'TAVG' and df_combined[temp_col].isna().any():
                # 如果TAVG缺失，用(TMAX + TMIN)/2估算
                mask = df_combined[temp_col].isna() & df_combined['TMAX'].notna() & df_combined['TMIN'].notna()
                df_combined.loc[mask, temp_col] = (df_combined.loc[mask, 'TMAX'] + df_combined.loc[mask, 'TMIN']) / 2
            
            # 删除重复列
            df_combined.drop(dup_col, axis=1, inplace=True)
    
    # 处理PGTM列（阵风时间）的缺失值
    if 'PGTM' in df_combined.columns:
        pgtm_missing = df_combined['PGTM'].isna().sum()
        if pgtm_missing > 0:
            print(f"PGTM列有 {pgtm_missing} 个缺失值，已用0填充")
            df_combined['PGTM'] = df_combined['PGTM'].fillna(0)
    
    # 合并空气质量数据
    df_final = pd.merge(df_combined, df_air, on='DATE', how='outer')
    
    # 筛选2025年6月的数据
    print("正在筛选2025年6月数据...")
    df_june_2025 = df_final[
        (df_final['DATE'].dt.year == 2025) & 
        (df_final['DATE'].dt.month == 6)
    ].copy()
    
    # 按日期排序
    df_june_2025.sort_values('DATE', inplace=True)
    df_june_2025.reset_index(drop=True, inplace=True)
    
    # 数据质量检查和改进
    print("\n正在进行数据质量检查...")
    
    # 检查各列的缺失情况
    missing_info = df_june_2025.isnull().sum()
    print("各列缺失值统计:")
    for col in df_june_2025.columns:
        missing_count = missing_info[col]
        total_count = len(df_june_2025)
        if missing_count > 0:
            print(f"  {col}: {missing_count}/{total_count} ({missing_count/total_count*100:.1f}%)")
    
    # 改进：如果TAVG仍然有缺失，用(TMAX+TMIN)/2填充
    if 'TAVG' in df_june_2025.columns and 'TMAX' in df_june_2025.columns and 'TMIN' in df_june_2025.columns:
        tavg_missing = df_june_2025['TAVG'].isna().sum()
        if tavg_missing > 0:
            print(f"使用TMAX和TMIN计算缺失的TAVG值 ({tavg_missing} 个)")
            mask = df_june_2025['TAVG'].isna() & df_june_2025['TMAX'].notna() & df_june_2025['TMIN'].notna()
            df_june_2025.loc[mask, 'TAVG'] = (df_june_2025.loc[mask, 'TMAX'] + df_june_2025.loc[mask, 'TMIN']) / 2
    
    # 最终缺失值检查
    final_missing = df_june_2025.isnull().sum().sum()
    if final_missing > 0:
        print(f"\n⚠️  警告: 仍有 {final_missing} 个缺失值存在")
        print("缺失值分布:")
        for col in df_june_2025.columns:
            missing_count = df_june_2025[col].isna().sum()
            if missing_count > 0:
                print(f"  {col}: {missing_count}")
    else:
        print("✅ 所有缺失值已处理完成")
    
    # 保存结果
    df_june_2025.to_csv(output_path, index=False)
    
    # 显示最终数据概况
    print(f"\n✅ 处理完成！文件已保存到: {output_path}")
    print(f"\n📊 最终数据概况:")
    print(f"  记录数量: {len(df_june_2025)} 条")
    print(f"  日期范围: {df_june_2025['DATE'].min().strftime('%Y-%m-%d')} 到 {df_june_2025['DATE'].max().strftime('%Y-%m-%d')}")
    print(f"  气象指标数量: {len(df_june_2025.columns)} 个")
    
    # 显示温度数据的统计信息
    temp_cols = ['TAVG', 'TMAX', 'TMIN']
    available_temp_cols = [col for col in temp_cols if col in df_june_2025.columns]
    
    if available_temp_cols:
        print(f"\n🌡️  温度数据统计:")
        for col in available_temp_cols:
            if df_june_2025[col].notna().any():
                print(f"  {col}: {df_june_2025[col].min():.1f}°C ~ {df_june_2025[col].max():.1f}°C, 平均 {df_june_2025[col].mean():.1f}°C")
            else:
                print(f"  {col}: 全部缺失")
    
    # 显示其他重要指标的统计
    important_cols = ['PRCP', 'AWND', 'pm25']
    for col in important_cols:
        if col in df_june_2025.columns and df_june_2025[col].notna().any():
            if col == 'PRCP':
                print(f"  🌧️  {col}: 最大 {df_june_2025[col].max():.1f}mm, 有降水天数 {df_june_2025[col].gt(0).sum()}")
            elif col == 'AWND':
                print(f"  💨  {col}: 平均 {df_june_2025[col].mean():.1f} m/s, 最大 {df_june_2025[col].max():.1f} m/s")
            elif col == 'pm25':
                print(f"  😷  {col}: 平均 {df_june_2025[col].mean():.1f} μg/m³, 最大 {df_june_2025[col].max():.1f} μg/m³")
    
    return df_june_2025

def check_data_quality(df):
    """
    检查数据质量
    """
    print("\n🔍 数据质量详细检查:")
    
    # 检查每个列的数据情况
    for col in df.columns:
        non_null_count = df[col].notna().sum()
        null_count = df[col].isna().sum()
        total_count = len(df)
        
        if null_count > 0:
            print(f"  {col}: {non_null_count}/{total_count} 有效值 ({null_count} 个缺失)")
            
            # 对于数值列，显示统计信息
            if pd.api.types.is_numeric_dtype(df[col]):
                valid_data = df[col].dropna()
                if len(valid_data) > 0:
                    print(f"     范围: {valid_data.min():.2f} ~ {valid_data.max():.2f}, 平均: {valid_data.mean():.2f}")
        else:
            print(f"  {col}: ✅ 完整")

# 执行数据处理
if __name__ == "__main__":
    try:
        print("=== 天气数据预处理 (改进版) ===")
        print("专门处理温度数据缺失问题")
        print("=" * 40)
        
        final_weather_data = process_weather_data()
        
        # 运行详细数据质量检查
        check_data_quality(final_weather_data)
        
        print("\n🎉 天气数据预处理完成！")
        print(f"💡 提示: 检查桌面上的 weather_data_june_2025.csv 文件")
        
    except FileNotFoundError as e:
        print(f"❌ 错误: 找不到文件")
        print("请确保以下文件存在于当前目录:")
        print("1. 第一个文件.xlsx (空气质量数据)")
        print("2. 第二个文件.xlsx (气象数据1)") 
        print("3. 第三个文件.csv (气象数据2)")
        print(f"详细错误: {e}")
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()