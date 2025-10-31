import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from datetime import datetime
import warnings
import zipfile
import os
warnings.filterwarnings('ignore')

# 设置字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

# 尝试导入高级库，如果失败则使用基础功能
try:
    import geopandas as gpd
    from shapely.geometry import Point, LineString
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    print("geopandas not available, using basic analysis functions")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("networkx not available, skipping network analysis")

class SimpleSubwayAnalyzer:
    def __init__(self, gtfs_zip_path):
        self.gtfs_path = gtfs_zip_path
        self.data_loaded = False
        
        # 颜色配置
        self.colors = {
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
    
    def load_gtfs_data(self):
        """Load GTFS data from ZIP file"""
        print("Loading GTFS data from ZIP file...")
        
        try:
            # Check if file exists
            if not os.path.exists(self.gtfs_path):
                print(f"Error: File '{self.gtfs_path}' does not exist!")
                return False
            
            # Open ZIP file
            with zipfile.ZipFile(self.gtfs_path, 'r') as zip_ref:
                # Get file list in ZIP
                file_list = zip_ref.namelist()
                print(f"Files in ZIP: {file_list}")
                
                # Load required files
                required_files = ['routes.txt', 'trips.txt', 'stop_times.txt', 'stops.txt']
                missing_files = [f for f in required_files if f not in file_list]
                
                if missing_files:
                    print(f"Warning: Missing required files: {missing_files}")
                    # Try to load available files
                    available_files = [f for f in required_files if f in file_list]
                    print(f"Will load available files: {available_files}")
                
                # Load data
                if 'routes.txt' in file_list:
                    with zip_ref.open('routes.txt') as f:
                        self.routes = pd.read_csv(f)
                    print(f"Loaded routes.txt: {len(self.routes)} records")
                
                if 'trips.txt' in file_list:
                    with zip_ref.open('trips.txt') as f:
                        self.trips = pd.read_csv(f)
                    print(f"Loaded trips.txt: {len(self.trips)} records")
                
                if 'stop_times.txt' in file_list:
                    with zip_ref.open('stop_times.txt') as f:
                        self.stop_times = pd.read_csv(f)
                    print(f"Loaded stop_times.txt: {len(self.stop_times)} records")
                
                if 'stops.txt' in file_list:
                    with zip_ref.open('stops.txt') as f:
                        self.stops = pd.read_csv(f)
                    print(f"Loaded stops.txt: {len(self.stops)} records")
                
                # Try to load optional files
                optional_files = ['agency.txt', 'calendar.txt', 'shapes.txt', 'transfers.txt']
                for file in optional_files:
                    if file in file_list:
                        with zip_ref.open(file) as f:
                            setattr(self, file.replace('.txt', ''), pd.read_csv(f))
                        print(f"Loaded {file}")
            
            print("GTFS data loading completed!")
            self._print_data_summary()
            self.data_loaded = True
            
        except Exception as e:
            print(f"Data loading error: {e}")
            return False
        return True
    
    def _print_data_summary(self):
        """Print data overview"""
        print(f"\n=== Data Overview ===")
        if hasattr(self, 'routes'):
            print(f"Number of routes: {len(self.routes)}")
        if hasattr(self, 'stops'):
            print(f"Number of stations: {len(self.stops)}")
        if hasattr(self, 'trips'):
            print(f"Number of trips: {len(self.trips)}")
        if hasattr(self, 'stop_times'):
            print(f"Number of stop time records: {len(self.stop_times)}")
        
        # Show route information
        if hasattr(self, 'routes'):
            print(f"\nTop 5 routes:")
            route_cols = ['route_id', 'route_short_name', 'route_long_name']
            available_cols = [col for col in route_cols if col in self.routes.columns]
            print(self.routes[available_cols].head())
        
        # Show station information
        if hasattr(self, 'stops'):
            print(f"\nTop 5 stations:")
            stop_cols = ['stop_id', 'stop_name', 'stop_lat', 'stop_lon']
            available_cols = [col for col in stop_cols if col in self.stops.columns]
            print(self.stops[available_cols].head())
    
    def analyze_basic_stats(self):
        """Basic statistical analysis"""
        if not self.data_loaded:
            print("Please load data first!")
            return
        
        print("\n=== Basic Statistical Analysis ===")
        
        # 1. Station geographic distribution
        if hasattr(self, 'stops') and 'stop_lat' in self.stops.columns and 'stop_lon' in self.stops.columns:
            print(f"Station coordinate range:")
            print(f"  Longitude: {self.stops['stop_lon'].min():.3f} - {self.stops['stop_lon'].max():.3f}")
            print(f"  Latitude: {self.stops['stop_lat'].min():.3f} - {self.stops['stop_lat'].max():.3f}")
        
        # 2. Route type analysis
        if hasattr(self, 'routes') and 'route_type' in self.routes.columns:
            route_type_counts = self.routes['route_type'].value_counts()
            print(f"\nRoute type distribution:")
            for route_type, count in route_type_counts.items():
                print(f"  Type {route_type}: {count} routes")
        
        # 3. Service frequency analysis
        self._analyze_service_frequency()
        
        # 4. Popular stations analysis
        self._analyze_popular_stations()
    
    def _analyze_service_frequency(self):
        """Analyze service frequency"""
        print("\n--- Service Frequency Analysis ---")
        
        # Check if necessary data exists
        if not hasattr(self, 'stop_times') or not hasattr(self, 'trips') or not hasattr(self, 'routes'):
            print("Missing required data files, skipping service frequency analysis")
            return
        
        # Merge data
        try:
            merged_data = self.stop_times.merge(
                self.trips[['trip_id', 'route_id']], on='trip_id'
            ).merge(
                self.routes[['route_id', 'route_short_name']], on='route_id'
            )
            
            # Count trips per route
            route_frequency = merged_data.groupby(['route_id', 'route_short_name']).agg({
                'trip_id': 'nunique'
            }).reset_index()
            route_frequency = route_frequency.rename(columns={'trip_id': 'trip_count'})
            route_frequency = route_frequency.sort_values('trip_count', ascending=False)
            
            print("Route service frequency ranking (Top 10):")
            for i, row in route_frequency.head(10).iterrows():
                print(f"  {row['route_short_name']}: {row['trip_count']} trips")
            
            self.route_frequency = route_frequency
            
        except Exception as e:
            print(f"Service frequency analysis error: {e}")
    
    def _analyze_popular_stations(self):
        """Analyze popular stations"""
        print("\n--- Popular Stations Analysis ---")
        
        if not hasattr(self, 'stop_times') or not hasattr(self, 'stops'):
            print("Missing required data files, skipping popular stations analysis")
            return
        
        try:
            # Count departures per station
            departure_counts = self.stop_times.groupby('stop_id').size().reset_index(name='departure_count')
            departure_counts = departure_counts.merge(
                self.stops[['stop_id', 'stop_name']], on='stop_id'
            )
            departure_counts = departure_counts.sort_values('departure_count', ascending=False)
            
            print("Busiest stations ranking (Top 10):")
            for i, row in departure_counts.head(10).iterrows():
                print(f"  {row['stop_name']}: {row['departure_count']} departures")
            
            self.departure_counts = departure_counts
            
        except Exception as e:
            print(f"Popular stations analysis error: {e}")
    
    def create_simple_network(self):
        """Create simplified network analysis"""
        if not NETWORKX_AVAILABLE:
            print("networkx not available, skipping network analysis")
            return None
            
        if not hasattr(self, 'stop_times') or not hasattr(self, 'stops'):
            print("Missing required data files, skipping network analysis")
            return None
            
        print("\n=== Network Analysis ===")
        
        try:
            G = nx.Graph()
            
            # Add nodes
            for _, stop in self.stops.iterrows():
                G.add_node(stop['stop_id'], name=stop['stop_name'])
            
            # Add connections (simplified)
            trip_groups = self.stop_times.groupby('trip_id')
            connection_count = {}
            
            for trip_id, group in trip_groups:
                sorted_stops = group.sort_values('stop_sequence')
                stops_list = sorted_stops['stop_id'].tolist()
                
                for i in range(len(stops_list) - 1):
                    edge = (stops_list[i], stops_list[i + 1])
                    connection_count[edge] = connection_count.get(edge, 0) + 1
            
            # Add edges
            for edge, weight in connection_count.items():
                G.add_edge(edge[0], edge[1], weight=weight)
            
            print(f"Network built: {G.number_of_nodes()} stations, {G.number_of_edges()} connections")
            
            # Calculate basic network metrics
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G, k=min(100, len(G)))
            
            # Find most important hub stations
            hub_stations = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:15]
            
            self.key_hubs = []
            for stop_id, centrality in hub_stations:
                stop_info = self.stops[self.stops['stop_id'] == stop_id].iloc[0]
                self.key_hubs.append({
                    'stop_id': stop_id,
                    'stop_name': stop_info['stop_name'],
                    'betweenness': centrality,
                    'degree': degree_centrality.get(stop_id, 0)
                })
            
            print("\nKey hub stations (Top 5):")
            for hub in self.key_hubs[:5]:
                print(f"  {hub['stop_name']}: Betweenness centrality {hub['betweenness']:.4f}")
            
            self.network = G
            return G
            
        except Exception as e:
            print(f"Network analysis error: {e}")
            return None
    
    def create_improved_visualizations(self):
        """Create improved visualizations with English labels"""
        if not self.data_loaded:
            print("Please load data first!")
            return
        
        print("\n=== Generating Improved Visualizations ===")
        
        try:
            # Create clearer chart layout with white background
            fig = plt.figure(figsize=(16, 12), facecolor='white')
            
            # 1. Subway Network Geographic Distribution
            ax1 = plt.subplot(2, 2, 1)
            ax1.set_facecolor('white')
            if hasattr(self, 'stops') and 'stop_lon' in self.stops.columns and 'stop_lat' in self.stops.columns:
                scatter = ax1.scatter(self.stops['stop_lon'], self.stops['stop_lat'], 
                                    alpha=0.7, s=30, c=self.colors['extreme'], 
                                    edgecolors=self.colors['positive'], linewidth=0.5)
                ax1.set_title('NYC Subway Station Distribution', fontsize=14, fontweight='bold')
                ax1.set_xlabel('Longitude')
                ax1.set_ylabel('Latitude')
                ax1.grid(True, alpha=0.3)
                
                # Add station count annotation
                total_stations = len(self.stops)
                ax1.text(0.02, 0.98, f'Total Stations: {total_stations}', 
                        transform=ax1.transAxes, fontsize=12, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            else:
                ax1.text(0.5, 0.5, 'Geographic data unavailable', ha='center', va='center', fontsize=12)
                ax1.set_title('Subway Station Distribution')
            
            # 2. Service Frequency Ranking
            ax2 = plt.subplot(2, 2, 2)
            ax2.set_facecolor('white')
            if hasattr(self, 'route_frequency') and len(self.route_frequency) > 0:
                top_routes = self.route_frequency.head(10)
                y_pos = np.arange(len(top_routes))
                
                bars = ax2.barh(y_pos, top_routes['trip_count'], 
                               color=self.colors['normal'], 
                               edgecolor=self.colors['increase'], 
                               alpha=0.7)
                
                ax2.set_yticks(y_pos)
                
                # Shorten route names for better display
                def shorten_name(name, max_length=15):
                    if len(str(name)) > max_length:
                        return str(name)[:max_length-3] + '...'
                    return str(name)
                
                short_names = [shorten_name(name) for name in top_routes['route_short_name']]
                ax2.set_yticklabels(short_names, fontsize=10)
                
                ax2.set_xlabel('Number of Trips')
                ax2.set_title('Top 10 Subway Routes by Service Frequency', fontsize=14, fontweight='bold')
                ax2.grid(True, axis='x', alpha=0.3)
                
                # Add value labels on bars
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax2.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                            f'{int(width)}', ha='left', va='center', fontsize=9)
            else:
                ax2.text(0.5, 0.5, 'Service frequency data unavailable', ha='center', va='center', fontsize=12)
                ax2.set_title('Route Service Frequency')
            
            # 3. Popular Stations Analysis
            ax3 = plt.subplot(2, 2, 3)
            ax3.set_facecolor('white')
            if hasattr(self, 'departure_counts') and len(self.departure_counts) > 0:
                top_stations = self.departure_counts.head(10)
                y_pos = np.arange(len(top_stations))
                
                bars = ax3.barh(y_pos, top_stations['departure_count'], 
                               color=self.colors['negative'], 
                               edgecolor=self.colors['positive'], 
                               alpha=0.7)
                
                ax3.set_yticks(y_pos)
                
                # Shorten station names
                station_names = [shorten_name(name, 20) for name in top_stations['stop_name']]
                ax3.set_yticklabels(station_names, fontsize=9)
                
                ax3.set_xlabel('Departure Count')
                ax3.set_title('Top 10 Busiest Subway Stations', fontsize=14, fontweight='bold')
                ax3.grid(True, axis='x', alpha=0.3)
                
                # Add value labels
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax3.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                            f'{int(width)}', ha='left', va='center', fontsize=9)
            else:
                ax3.text(0.5, 0.5, 'Station usage data unavailable', ha='center', va='center', fontsize=12)
                ax3.set_title('Popular Stations')
            
            # 4. Network Centrality Analysis
            ax4 = plt.subplot(2, 2, 4)
            ax4.set_facecolor('white')
            if hasattr(self, 'network') and self.network and hasattr(self, 'key_hubs'):
                # Use key hub data
                hub_names = [hub['stop_name'] for hub in self.key_hubs[:8]]
                hub_scores = [hub['betweenness'] for hub in self.key_hubs[:8]]
                
                y_pos = np.arange(len(hub_names))
                
                bars = ax4.barh(y_pos, hub_scores, 
                               color=self.colors['accent1'], 
                               edgecolor=self.colors['decrease'], 
                               alpha=0.7)
                
                ax4.set_yticks(y_pos)
                short_hub_names = [shorten_name(name, 18) for name in hub_names]
                ax4.set_yticklabels(short_hub_names, fontsize=9)
                
                ax4.set_xlabel('Betweenness Centrality')
                ax4.set_title('Key Hub Stations in Network', fontsize=14, fontweight='bold')
                ax4.grid(True, axis='x', alpha=0.3)
                
                # Add value labels
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax4.text(width + width*0.001, bar.get_y() + bar.get_height()/2, 
                            f'{width:.4f}', ha='left', va='center', fontsize=8)
            else:
                ax4.text(0.5, 0.5, 'Network analysis data unavailable', ha='center', va='center', fontsize=12)
                ax4.set_title('Network Hub Analysis')
            
            plt.tight_layout()
            plt.savefig('subway_analysis_improved_en.png', dpi=300, bbox_inches='tight', facecolor='white')
            print("Improved visualizations saved as 'subway_analysis_improved_en.png'")
            plt.show()
            
            # Print chart explanations
            self._explain_visualizations_en()
            
        except Exception as e:
            print(f"Visualization generation error: {e}")
            import traceback
            traceback.print_exc()
    
    def _explain_visualizations_en(self):
        """Explain visualizations in English"""
        print("\n" + "="*60)
        print("VISUALIZATION EXPLANATION")
        print("="*60)
        
        print("\n1. NYC Subway Station Distribution")
        print("   - Shows geographic distribution of all subway stations")
        print("   - Each red dot represents a subway station")
        print("   - Helps identify spatial density and coverage gaps")
        
        print("\n2. Top Subway Routes by Service Frequency")
        print("   - Ranks top 10 subway routes by number of trips")
        print("   - Longer bars indicate higher service frequency")
        print("   - High-frequency routes are major urban transportation arteries")
        
        print("\n3. Top Busiest Subway Stations")
        print("   - Shows top 10 stations by departure count")
        print("   - Reflects station passenger volume and importance")
        print("   - Busy stations are typically transfer hubs or in core areas")
        
        print("\n4. Key Hub Stations in Network")
        print("   - Shows stations with highest betweenness centrality")
        print("   - Betweenness centrality measures connection importance in network")
        print("   - High-centrality stations are critical for network resilience")
        
        print("\n" + "="*60)
        
        # Provide additional insights if specific data exists
        if hasattr(self, 'key_hubs') and self.key_hubs:
            print("\nKEY FINDINGS:")
            print(f"   Most important hub: {self.key_hubs[0]['stop_name']}")
            print(f"   Total stations analyzed: {len(self.stops) if hasattr(self, 'stops') else 'N/A'}")
            
        if hasattr(self, 'route_frequency') and len(self.route_frequency) > 0:
            busiest_route = self.route_frequency.iloc[0]
            print(f"   Busiest route: {busiest_route['route_short_name']} ({busiest_route['trip_count']} trips)")
    
    def visualize_basic_results(self):
        """Basic visualization - using improved English version"""
        self.create_improved_visualizations()
    
    def export_analysis_results(self):
        """Export analysis results"""
        if not self.data_loaded:
            print("Please load data first!")
            return
        
        try:
            # Export key data
            results = {
                'summary': {
                    'total_routes': len(self.routes) if hasattr(self, 'routes') else 0,
                    'total_stations': len(self.stops) if hasattr(self, 'stops') else 0,
                    'total_trips': len(self.trips) if hasattr(self, 'trips') else 0
                }
            }
            
            if hasattr(self, 'route_frequency'):
                results['top_routes'] = self.route_frequency.head(10).to_dict('records')
            
            if hasattr(self, 'departure_counts'):
                results['top_stations'] = self.departure_counts.head(10).to_dict('records')
            
            # Save as JSON
            import json
            with open('subway_analysis_results.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print("Analysis results exported as 'subway_analysis_results.json'")
            
            # Save as CSV
            if hasattr(self, 'route_frequency'):
                self.route_frequency.to_csv('route_frequency.csv', index=False, encoding='utf-8')
            
            if hasattr(self, 'departure_counts'):
                self.departure_counts.to_csv('station_popularity.csv', index=False, encoding='utf-8')
            
            print("Detailed data exported as CSV files")
            
        except Exception as e:
            print(f"Export results error: {e}")
    
    def run_complete_analysis(self):
        """Run complete analysis workflow"""
        print("Starting subway system analysis...")
        print("=" * 50)
        
        # 1. Load data
        if not self.load_gtfs_data():
            return
        
        # 2. Basic statistical analysis
        self.analyze_basic_stats()
        
        # 3. Network analysis
        self.create_simple_network()
        
        # 4. Visualization
        self.visualize_basic_results()
        
        # 5. Export results
        self.export_analysis_results()
        
        print("\n" + "=" * 50)
        print("Analysis completed!")

# Usage example
if __name__ == "__main__":
    # Replace with your GTFS ZIP file path
    gtfs_zip_path = "D:\\作业\\大三上课程\\大数据原理与应用\\期初作业\\gtfs_subway.zip"
    
    # Create analyzer and run analysis
    analyzer = SimpleSubwayAnalyzer(gtfs_zip_path)
    analyzer.run_complete_analysis()