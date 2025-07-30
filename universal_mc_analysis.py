import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

def read_earthquake_catalog(filename, year_col=13, month_col=14, day_col=15, 
                          hour_col=16, minute_col=17, second_col=18, mag_col=2,
                          delimiter=None):
    """
    Read earthquake catalog with flexible format support
    """
    try:
        # Try different delimiters if not specified
        if delimiter is None:
            delimiters = ['\s+', ',', '\t', ';', '|']
        else:
            delimiters = [delimiter]
        
        data = None
        for delim in delimiters:
            try:
                data = pd.read_csv(filename, sep=delim, header=None, engine='python')
                if data is not None and len(data.columns) > max([year_col, month_col, day_col, 
                                                                  hour_col, minute_col, second_col, mag_col]):
                    break
            except:
                continue
        
        if data is None:
            raise ValueError("Could not parse file with any common delimiter")
        
        # Extract data
        years = data.iloc[:, year_col].values
        months = data.iloc[:, month_col].values
        days = data.iloc[:, day_col].values
        hours = data.iloc[:, hour_col].values
        minutes = data.iloc[:, minute_col].values
        seconds = data.iloc[:, second_col].values
        magnitudes = data.iloc[:, mag_col].values
        
        # Build timestamps with error handling
        timestamps = []
        valid_indices = []
        
        for i in range(len(years)):
            try:
                dt = datetime(int(years[i]), int(months[i]), int(days[i]), 
                            int(hours[i]), int(minutes[i]), int(float(seconds[i])))
                timestamps.append(dt)
                valid_indices.append(i)
            except:
                continue
        
        timestamps = np.array(timestamps)
        magnitudes = magnitudes[valid_indices]
        
        # Remove NaN values
        valid_mask = ~np.isnan(magnitudes)
        timestamps = timestamps[valid_mask]
        magnitudes = magnitudes[valid_mask]
        
        print(f"\nSuccessfully read {len(magnitudes)} events from {filename}")
        print(f"Magnitude range: {np.min(magnitudes):.2f} - {np.max(magnitudes):.2f}")
        print(f"Time range: {timestamps[0]} to {timestamps[-1]}")
        
        return timestamps, magnitudes
    
    except Exception as e:
        print(f"\nError reading file: {e}")
        return None, None

def calculate_FM(timestamps, magnitudes, m_th):
    """Calculate normalized quantity FM(t,m|m_th)"""
    valid_mask = magnitudes < m_th
    valid_times = timestamps[valid_mask]
    
    if len(valid_times) < 10:
        return None, None
    
    sorted_indices = np.argsort(valid_times)
    sorted_times = valid_times[sorted_indices]
    
    n_events = len(sorted_times)
    FM_values = np.arange(1, n_events + 1) / n_events
    
    return sorted_times, FM_values

def calculate_curve_similarity(times1, FM1, times2, FM2, time_range):
    """Calculate Kolmogorov-Smirnov statistic between two FM curves"""
    try:
        start_time = time_range[0]
        rel_times1 = np.array([(t - start_time).total_seconds() / 86400 for t in times1])
        rel_times2 = np.array([(t - start_time).total_seconds() / 86400 for t in times2])
        
        # Common time grid
        common_time = np.linspace(0, (time_range[1] - time_range[0]).total_seconds() / 86400, 1000)
        
        # Interpolate
        f1 = interp1d(rel_times1, FM1, bounds_error=False, fill_value=(0, 1))
        f2 = interp1d(rel_times2, FM2, bounds_error=False, fill_value=(0, 1))
        
        FM1_interp = f1(common_time)
        FM2_interp = f2(common_time)
        
        # KS statistic
        ks_stat = np.max(np.abs(FM1_interp - FM2_interp))
        
        return ks_stat
    except:
        return np.inf

def estimate_completeness_magnitude(timestamps, magnitudes, m_th_values, 
                                  ks_threshold=0.01, min_events=50):
    """Automatically estimate completeness magnitude"""
    m_th_values = sorted(m_th_values)
    
    # Calculate FM curves
    curves = {}
    for m_th in m_th_values:
        times, FM = calculate_FM(timestamps, magnitudes, m_th)
        if times is not None and len(times) >= min_events:
            curves[m_th] = (times, FM)
    
    if len(curves) < 2:
        return None, None
    
    # Get time range
    all_times = np.concatenate([curves[m_th][0] for m_th in curves])
    time_range = (np.min(all_times), np.max(all_times))
    
    # Calculate KS statistics
    ks_stats = {}
    m_th_list = sorted(curves.keys())
    
    for i in range(len(m_th_list) - 1):
        m_th1 = m_th_list[i]
        m_th2 = m_th_list[i + 1]
        
        times1, FM1 = curves[m_th1]
        times2, FM2 = curves[m_th2]
        
        ks_stat = calculate_curve_similarity(times1, FM1, times2, FM2, time_range)
        ks_stats[m_th2] = ks_stat
    
    # Find completeness magnitude
    mc = None
    for m_th in sorted(ks_stats.keys()):
        if ks_stats[m_th] < ks_threshold:
            mc = m_th
            break
    
    # Alternative method if no clear completeness found
    if mc is None and len(ks_stats) > 0:
        ks_values = list(ks_stats.values())
        if len(ks_values) > 2:
            gradients = np.gradient(ks_values)
            stable_idx = np.argmin(np.abs(gradients))
            mc = list(ks_stats.keys())[stable_idx]
    
    return mc, ks_stats

def plot_magnitude_completeness(filename, m_th_values=None, 
                               year_col=13, month_col=14, day_col=15,
                               hour_col=16, minute_col=17, second_col=18, 
                               mag_col=2, delimiter=None, auto_detect_mc=True,
                               ks_threshold=0.01, min_events=50):
    """
    Plot magnitude completeness with automatic Mc detection
    """
    # Read data
    timestamps, magnitudes = read_earthquake_catalog(filename, year_col, month_col, 
                                                    day_col, hour_col, minute_col, 
                                                    second_col, mag_col, delimiter)
    
    if timestamps is None:
        return None
    
    # Default m_th values if not specified
    if m_th_values is None:
        mag_min = np.floor(np.min(magnitudes) * 10) / 10
        mag_max = np.ceil(np.max(magnitudes) * 10) / 10
        m_th_values = np.arange(mag_min, min(mag_max, mag_min + 3), 0.1)
    
    # Convert to relative time (days)
    start_time = timestamps[0]
    relative_days = np.array([(t - start_time).total_seconds() / 86400 for t in timestamps])
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Left Y-axis: Magnitude scatter plot
    ax1.scatter(relative_days, magnitudes, s=10, c='black', alpha=0.6, rasterized=True)
    ax1.set_xlabel('Time (days)', fontsize=12)
    ax1.set_ylabel('Magnitude', fontsize=12)
    ax1.set_ylim(0, max(magnitudes) + 0.5)
    ax1.grid(True, alpha=0.3)
    
    # Right Y-axis: Normalized FM
    ax2 = ax1.twinx()
    
    # Color palette
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(m_th_values)))
    
    # Store curves for completeness analysis
    curves_data = {}
    
    # Plot FM curves
    for i, m_th in enumerate(m_th_values):
        sorted_times, FM_values = calculate_FM(timestamps, magnitudes, m_th)
        
        if sorted_times is not None:
            relative_times = np.array([(t - start_time).total_seconds() / 86400 
                                     for t in sorted_times])
            
            ax2.plot(relative_times, FM_values, '-', color=colors[i], 
                    linewidth=2, label=f'$m_{{th}}$ = {m_th:.1f}', alpha=0.8)
            
            curves_data[m_th] = (sorted_times, FM_values)
    
    ax2.set_ylabel('Normalized Quantity $F_M(t,m|m_{th})$', fontsize=12)
    ax2.set_ylim(0, 1.05)
    
    # Automatic completeness magnitude detection
    mc = None
    if auto_detect_mc and len(curves_data) > 1:
        mc, ks_stats = estimate_completeness_magnitude(timestamps, magnitudes, 
                                                       list(curves_data.keys()),
                                                       ks_threshold, min_events)
        
        if mc is not None:
            # Add vertical line at completeness magnitude
            ax1.axhline(y=mc, color='red', linestyle='--', linewidth=2.5, alpha=0.7)
            ax1.text(0.02, 0.95, f'$M_c$ = {mc:.1f}', transform=ax1.transAxes,
                    fontsize=14, color='red', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             edgecolor='red', alpha=0.8))
            
            # Print detailed results
            print("\n" + "="*60)
            print("MAGNITUDE COMPLETENESS ANALYSIS RESULTS")
            print("="*60)
            print(f"Estimated Completeness Magnitude (Mc): {mc:.2f}")
            print(f"Total number of events: {len(magnitudes)}")
            print(f"Events with M ≥ Mc: {np.sum(magnitudes >= mc)}")
            print(f"Completeness percentage: {100*np.sum(magnitudes >= mc)/len(magnitudes):.1f}%")
            print(f"Catalog time span: {(timestamps[-1] - timestamps[0]).days:.1f} days")
            
            if ks_stats:
                print("\nKolmogorov-Smirnov Statistics:")
                for m_th in sorted(ks_stats.keys()):
                    status = "COMPLETE" if ks_stats[m_th] < ks_threshold else "INCOMPLETE"
                    print(f"  m_th = {m_th:.1f}: KS = {ks_stats[m_th]:.4f} [{status}]")
            
            print("\nInterpretation:")
            print(f"The catalog appears to be complete for magnitudes ≥ {mc:.1f}")
        else:
            print("\nWarning: Could not automatically determine completeness magnitude.")
            print("Possible reasons:")
            print("- Insufficient data for some magnitude thresholds")
            print("- No clear convergence of FM curves")
            print("- Try adjusting ks_threshold or min_events parameters")
    
    # Legend
    ax2.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), 
              frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.show()
    
    return mc

def interactive_analysis():
    """Interactive analysis with user input"""
    print("\n" + "="*60)
    print("EARTHQUAKE CATALOG COMPLETENESS MAGNITUDE ANALYSIS")
    print("="*60)
    
    # Get filename
    filename = input("\nEnter earthquake catalog filename (default: tangshan.reloc): ").strip()
    if not filename:
        filename = "tangshan.reloc"
    
    # Get magnitude thresholds
    print("\nEnter magnitude thresholds (comma-separated, e.g., 1.1,1.2,1.3,1.4,1.5,2.0,2.5)")
    print("Press Enter to use automatic range based on data")
    m_th_input = input("Magnitude thresholds: ").strip()
    
    if m_th_input:
        try:
            m_th_values = [float(x.strip()) for x in m_th_input.split(',')]
        except:
            print("Invalid format, will use automatic range")
            m_th_values = None
    else:
        m_th_values = None
    
    # Ask for column customization
    custom_cols = input("\nCustomize column numbers? (y/n, default: n): ").strip().lower()
    
    if custom_cols == 'y':
        print("\nEnter column numbers (0-based indexing, press Enter for defaults):")
        try:
            year_col = int(input("Year column (default 13): ") or "13")
            month_col = int(input("Month column (default 14): ") or "14")
            day_col = int(input("Day column (default 15): ") or "15")
            hour_col = int(input("Hour column (default 16): ") or "16")
            minute_col = int(input("Minute column (default 17): ") or "17")
            second_col = int(input("Second column (default 18): ") or "18")
            mag_col = int(input("Magnitude column (default 2): ") or "2")
        except:
            print("Invalid input, using defaults")
            year_col, month_col, day_col = 13, 14, 15
            hour_col, minute_col, second_col = 16, 17, 18
            mag_col = 2
    else:
        year_col, month_col, day_col = 13, 14, 15
        hour_col, minute_col, second_col = 16, 17, 18
        mag_col = 2
    
    # Ask for delimiter
    delimiter_input = input("\nSpecify delimiter (space/comma/tab, default: auto-detect): ").strip().lower()
    delimiter_map = {'space': '\s+', 'comma': ',', 'tab': '\t'}
    delimiter = delimiter_map.get(delimiter_input, None)
    
    # Advanced options
    advanced = input("\nUse advanced options? (y/n, default: n): ").strip().lower()
    
    if advanced == 'y':
        try:
            ks_threshold = float(input("KS threshold for completeness (default 0.01): ") or "0.01")
            min_events = int(input("Minimum events per threshold (default 50): ") or "50")
        except:
            ks_threshold = 0.01
            min_events = 50
    else:
        ks_threshold = 0.01
        min_events = 50
    
    # Perform analysis
    print("\nAnalyzing catalog...")
    mc = plot_magnitude_completeness(filename, m_th_values, year_col, month_col, 
                                    day_col, hour_col, minute_col, second_col, 
                                    mag_col, delimiter, auto_detect_mc=True,
                                    ks_threshold=ks_threshold, min_events=min_events)
    
    # Save results option
    if mc is not None:
        save = input("\nSave results to file? (y/n): ").strip().lower()
        if save == 'y':
            output_file = f"{filename.split('.')[0]}_completeness_analysis.txt"
            with open(output_file, 'w') as f:
                f.write(f"Completeness Analysis Results for {filename}\n")
                f.write(f"Completeness Magnitude (Mc): {mc:.2f}\n")
                f.write(f"Analysis performed on: {datetime.now()}\n")
            print(f"Results saved to {output_file}")

# Main program
if __name__ == "__main__":
    # Interactive mode
    interactive_analysis()
    
    # Or direct usage example:
    # mc = plot_magnitude_completeness('tangshan.reloc', 
    #                                 m_th_values=[1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5],
    #                                 auto_detect_mc=True)