import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# Global configuration for scientific rigor
FLOAT_TOLERANCE = 1e-6  # Tolerance for floating point comparisons
DEFAULT_KS_THRESHOLD = 0.05
DEFAULT_MIN_EVENTS = 50

def read_earthquake_catalog(filename, year_col=13, month_col=14, day_col=15, 
                          hour_col=16, minute_col=17, second_col=18, mag_col=2,
                          delimiter=None, verbose=True):
    """
    Read earthquake catalog with flexible format support and robust error handling
    
    Parameters:
    -----------
    filename : str
        Path to earthquake catalog file
    year_col, month_col, ... : int
        Column indices (0-based) for time components
    mag_col : int
        Column index for magnitude
    delimiter : str or None
        File delimiter, auto-detected if None
    verbose : bool
        Print detailed information
    """
    try:
        # Try different delimiters if not specified
        if delimiter is None:
            delimiters = ['\s+', ',', '\t', ';', '|']
        else:
            delimiters = [delimiter]
        
        data = None
        used_delimiter = None
        for delim in delimiters:
            try:
                data = pd.read_csv(filename, sep=delim, header=None, engine='python')
                if data is not None and len(data.columns) > max([year_col, month_col, day_col, 
                                                                  hour_col, minute_col, second_col, mag_col]):
                    used_delimiter = delim
                    break
            except:
                continue
        
        if data is None:
            raise ValueError("Could not parse file with any common delimiter")
        
        # Extract data with consistent data types
        years = data.iloc[:, year_col].astype(float).values
        months = data.iloc[:, month_col].astype(float).values
        days = data.iloc[:, day_col].astype(float).values
        hours = data.iloc[:, hour_col].astype(float).values
        minutes = data.iloc[:, minute_col].astype(float).values
        seconds = data.iloc[:, second_col].astype(float).values
        magnitudes = data.iloc[:, mag_col].astype(float).values
        
        # Build timestamps with error handling
        timestamps = []
        valid_indices = []
        invalid_count = 0
        
        for i in range(len(years)):
            try:
                dt = datetime(int(years[i]), int(months[i]), int(days[i]), 
                            int(hours[i]), int(minutes[i]), int(float(seconds[i])))
                timestamps.append(dt)
                valid_indices.append(i)
            except:
                invalid_count += 1
                continue
        
        timestamps = np.array(timestamps)
        magnitudes = magnitudes[valid_indices]
        
        # Remove NaN values
        valid_mask = ~np.isnan(magnitudes)
        timestamps = timestamps[valid_mask]
        magnitudes = magnitudes[valid_mask]
        
        if verbose:
            print(f"\n{'='*60}")
            print("DATA READING SUMMARY")
            print(f"{'='*60}")
            print(f"File: {filename}")
            print(f"Delimiter detected: {used_delimiter if used_delimiter != '\s+' else 'space'}")
            print(f"Total rows in file: {len(data)}")
            print(f"Invalid time entries skipped: {invalid_count}")
            print(f"Valid events loaded: {len(magnitudes)}")
            print(f"Magnitude range: {np.min(magnitudes):.2f} - {np.max(magnitudes):.2f}")
            print(f"Time range: {timestamps[0]} to {timestamps[-1]}")
            print(f"Time span: {(timestamps[-1] - timestamps[0]).days:.1f} days")
        
        return timestamps, magnitudes
    
    except Exception as e:
        print(f"\nError reading file: {e}")
        return None, None

def calculate_FM(timestamps, magnitudes, m_th, verbose=False):
    """
    Calculate normalized quantity FM(t,m|m_th) with improved robustness
    """
    # Use strict inequality to avoid boundary issues
    valid_mask = magnitudes < m_th - FLOAT_TOLERANCE
    valid_times = timestamps[valid_mask]
    
    if verbose:
        print(f"  m_th = {m_th:.1f}: {len(valid_times)} events with M < {m_th}")
    
    if len(valid_times) < 10:
        return None, None
    
    sorted_indices = np.argsort(valid_times)
    sorted_times = valid_times[sorted_indices]
    
    n_events = len(sorted_times)
    FM_values = np.arange(1, n_events + 1) / n_events
    
    return sorted_times, FM_values

def calculate_curve_similarity(times1, FM1, times2, FM2, time_range):
    """
    Calculate Kolmogorov-Smirnov statistic between two FM curves
    with improved numerical stability
    """
    try:
        start_time = time_range[0]
        rel_times1 = np.array([(t - start_time).total_seconds() / 86400 for t in times1])
        rel_times2 = np.array([(t - start_time).total_seconds() / 86400 for t in times2])
        
        # Ensure sufficient overlap
        overlap_start = max(rel_times1[0], rel_times2[0])
        overlap_end = min(rel_times1[-1], rel_times2[-1])
        
        if overlap_end <= overlap_start:
            return np.inf
        
        # Common time grid with higher resolution for better accuracy
        n_points = max(1000, len(rel_times1) + len(rel_times2))
        common_time = np.linspace(overlap_start, overlap_end, n_points)
        
        # Interpolate with linear method for stability
        f1 = interp1d(rel_times1, FM1, kind='linear', bounds_error=False, fill_value=(0, 1))
        f2 = interp1d(rel_times2, FM2, kind='linear', bounds_error=False, fill_value=(0, 1))
        
        FM1_interp = f1(common_time)
        FM2_interp = f2(common_time)
        
        # KS statistic
        ks_stat = np.max(np.abs(FM1_interp - FM2_interp))
        
        return ks_stat
    except Exception as e:
        print(f"Warning: Error in curve similarity calculation: {e}")
        return np.inf

def estimate_completeness_magnitude(timestamps, magnitudes, m_th_values, 
                                  ks_threshold=DEFAULT_KS_THRESHOLD, 
                                  min_events=DEFAULT_MIN_EVENTS,
                                  verbose=True):
    """
    Automatically estimate completeness magnitude with improved robustness
    
    This method implements multiple improvements:
    1. Floating point tolerance for boundary cases
    2. Multiple candidate handling
    3. Detailed diagnostic output
    4. Fallback methods for edge cases
    """
    m_th_values = sorted(m_th_values)
    
    # Calculate FM curves
    curves = {}
    if verbose:
        print(f"\n{'='*60}")
        print("FM CURVE CALCULATION")
        print(f"{'='*60}")
    
    for m_th in m_th_values:
        times, FM = calculate_FM(timestamps, magnitudes, m_th, verbose=verbose)
        if times is not None and len(times) >= min_events:
            curves[m_th] = (times, FM)
        elif verbose:
            print(f"  m_th = {m_th:.1f}: Insufficient data (< {min_events} events)")
    
    if len(curves) < 2:
        if verbose:
            print("\nInsufficient data for completeness analysis")
        return None, None
    
    # Get time range
    all_times = np.concatenate([curves[m_th][0] for m_th in curves])
    time_range = (np.min(all_times), np.max(all_times))
    
    # Calculate KS statistics with improved logic
    ks_stats = {}
    m_th_list = sorted(curves.keys())
    
    if verbose:
        print(f"\n{'='*60}")
        print("KOLMOGOROV-SMIRNOV ANALYSIS")
        print(f"{'='*60}")
        print(f"KS threshold: {ks_threshold}")
        print("\nPairwise comparisons:")
    
    for i in range(len(m_th_list) - 1):
        m_th1 = m_th_list[i]
        m_th2 = m_th_list[i + 1]
        
        times1, FM1 = curves[m_th1]
        times2, FM2 = curves[m_th2]
        
        ks_stat = calculate_curve_similarity(times1, FM1, times2, FM2, time_range)
        ks_stats[m_th2] = ks_stat
        
        if verbose:
            status = "COMPLETE" if ks_stat < ks_threshold else "INCOMPLETE"
            print(f"  KS({m_th1:.1f}, {m_th2:.1f}) → m_th={m_th2:.1f}: KS = {ks_stat:.6f} [{status}]")
    
    # Find completeness magnitude with improved logic
    mc_candidates = []
    mc = None
    
    # Primary method: Find all candidates below threshold
    for m_th in sorted(ks_stats.keys()):
        # Use tolerance to avoid floating point issues
        if ks_stats[m_th] < ks_threshold - FLOAT_TOLERANCE:
            mc_candidates.append(m_th)
    
    if mc_candidates:
        # Choose the first (most conservative) candidate
        mc = mc_candidates[0]
        
        if verbose:
            print(f"\n{'='*60}")
            print("COMPLETENESS MAGNITUDE DETERMINATION")
            print(f"{'='*60}")
            print(f"Candidates with KS < {ks_threshold}: {mc_candidates}")
            print(f"Selected (most conservative): Mc = {mc:.2f}")
            
            # Check for boundary cases
            if len(mc_candidates) > 1:
                print(f"\nNote: Multiple candidates found. Consider:")
                print(f"  - Conservative estimate: {mc_candidates[0]:.2f}")
                print(f"  - Liberal estimate: {mc_candidates[-1]:.2f}")
                print(f"  - Median estimate: {np.median(mc_candidates):.2f}")
    else:
        # Fallback method: gradient analysis
        if verbose:
            print(f"\n{'='*60}")
            print("FALLBACK METHOD: GRADIENT ANALYSIS")
            print(f"{'='*60}")
            print("No clear completeness found using primary method")
        
        ks_values = list(ks_stats.values())
        if len(ks_values) > 2:
            # Find where the gradient stabilizes
            gradients = np.gradient(ks_values)
            second_gradients = np.gradient(gradients)
            
            # Look for minimum second derivative (inflection point)
            stable_idx = np.argmin(np.abs(second_gradients))
            mc = list(ks_stats.keys())[stable_idx]
            
            if verbose:
                print(f"Gradient analysis suggests: Mc = {mc:.2f}")
                print("Warning: This is less reliable than the primary method")
    
    # Additional diagnostics
    if verbose and mc is not None:
        n_complete = np.sum(magnitudes >= mc)
        completeness_ratio = n_complete / len(magnitudes)
        
        print(f"\n{'='*60}")
        print("COMPLETENESS ASSESSMENT")
        print(f"{'='*60}")
        print(f"Estimated Mc: {mc:.2f}")
        print(f"Complete events (M ≥ {mc:.1f}): {n_complete}")
        print(f"Completeness ratio: {completeness_ratio:.1%}")
        
        # Quality check
        if completeness_ratio < 0.1:
            print("\nWARNING: Very low completeness ratio (<10%)")
            print("Consider: Lower magnitude thresholds or larger dataset")
        elif completeness_ratio > 0.9:
            print("\nWARNING: Very high completeness ratio (>90%)")
            print("Consider: Higher magnitude thresholds to test")
    
    return mc, ks_stats

def plot_magnitude_completeness(filename, m_th_values=None, 
                               year_col=13, month_col=14, day_col=15,
                               hour_col=16, minute_col=17, second_col=18, 
                               mag_col=2, delimiter=None, auto_detect_mc=True,
                               ks_threshold=DEFAULT_KS_THRESHOLD, 
                               min_events=DEFAULT_MIN_EVENTS,
                               verbose=True, save_figure=False):
    """
    Plot magnitude completeness with automatic Mc detection and enhanced diagnostics
    """
    # Read data
    timestamps, magnitudes = read_earthquake_catalog(filename, year_col, month_col, 
                                                    day_col, hour_col, minute_col, 
                                                    second_col, mag_col, delimiter, 
                                                    verbose)
    
    if timestamps is None:
        return None
    
    # Default m_th values if not specified
    if m_th_values is None:
        mag_min = np.floor(np.min(magnitudes) * 10) / 10
        mag_max = np.ceil(np.max(magnitudes) * 10) / 10
        # Ensure reasonable range
        m_th_values = np.arange(max(0, mag_min), 
                               min(mag_max, mag_min + 3), 
                               0.1)
        if verbose:
            print(f"\nAuto-generated magnitude thresholds: {m_th_values[0]:.1f} to {m_th_values[-1]:.1f}")
    
    # Convert to relative time (days)
    start_time = timestamps[0]
    relative_days = np.array([(t - start_time).total_seconds() / 86400 for t in timestamps])
    
    # Create figure with improved layout
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Left Y-axis: Magnitude scatter plot
    scatter = ax1.scatter(relative_days, magnitudes, s=10, c='black', 
                         alpha=0.6, edgecolors='none', rasterized=True)
    ax1.set_xlabel('Time (days)', fontsize=12)
    ax1.set_ylabel('Magnitude', fontsize=12)
    ax1.set_ylim(0, max(magnitudes) + 0.5)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_axisbelow(True)
    
    # Right Y-axis: Normalized FM
    ax2 = ax1.twinx()
    
    # Color palette - use distinguishable colors
    n_curves = len(m_th_values)
    if n_curves <= 10:
        colors = plt.cm.tab10(np.linspace(0, 0.9, n_curves))
    else:
        colors = plt.cm.viridis(np.linspace(0, 0.9, n_curves))
    
    # Store curves for completeness analysis
    curves_data = {}
    
    # Plot FM curves
    for i, m_th in enumerate(m_th_values):
        sorted_times, FM_values = calculate_FM(timestamps, magnitudes, m_th)
        
        if sorted_times is not None:
            relative_times = np.array([(t - start_time).total_seconds() / 86400 
                                     for t in sorted_times])
            
            # Use thicker lines for better visibility
            ax2.plot(relative_times, FM_values, '-', color=colors[i], 
                    linewidth=2.5, label=f'$m_{{th}}$ = {m_th:.1f}', 
                    alpha=0.8)
            
            curves_data[m_th] = (sorted_times, FM_values)
    
    ax2.set_ylabel('Normalized Quantity $F_M(t,m|m_{th})$', fontsize=12)
    ax2.set_ylim(0, 1.05)
    
    # Automatic completeness magnitude detection
    mc = None
    if auto_detect_mc and len(curves_data) > 1:
        mc, ks_stats = estimate_completeness_magnitude(timestamps, magnitudes, 
                                                       list(curves_data.keys()),
                                                       ks_threshold, min_events,
                                                       verbose)
        
        if mc is not None:
            # Add horizontal line at completeness magnitude
            ax1.axhline(y=mc, color='red', linestyle='--', linewidth=3, alpha=0.8)
            
            # Add text box with results
            textstr = f'$M_c$ = {mc:.2f}\n'
            textstr += f'N = {len(magnitudes)}\n'
            textstr += f'N($M \geq M_c$) = {np.sum(magnitudes >= mc)}'
            
            props = dict(boxstyle='round,pad=0.5', facecolor='white', 
                        edgecolor='red', linewidth=2, alpha=0.9)
            ax1.text(0.02, 0.95, textstr, transform=ax1.transAxes, fontsize=12,
                    verticalalignment='top', bbox=props, fontweight='bold')
            
            # Final summary
            if not verbose:
                print(f"\n{'='*60}")
                print("MAGNITUDE COMPLETENESS ANALYSIS RESULTS")
                print(f"{'='*60}")
                print(f"Estimated Completeness Magnitude (Mc): {mc:.2f}")
                print(f"Total number of events: {len(magnitudes)}")
                print(f"Events with M ≥ Mc: {np.sum(magnitudes >= mc)}")
                print(f"Completeness percentage: {100*np.sum(magnitudes >= mc)/len(magnitudes):.1f}%")
        else:
            print("\nWarning: Could not automatically determine completeness magnitude.")
            print("Please inspect the plot visually for curve convergence.")
    
    # Legend with optimized placement
    if len(m_th_values) <= 15:
        ax2.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), 
                  frameon=True, fancybox=True, shadow=True, 
                  title='Magnitude\nThresholds')
    
    # Improve overall appearance
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_figure:
        figure_file = f"{filename.split('.')[0]}_completeness.png"
        plt.savefig(figure_file, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved as: {figure_file}")
    
    plt.show()
    
    return mc

def interactive_analysis():
    """Interactive analysis with user input and enhanced options"""
    print("\n" + "="*60)
    print("EARTHQUAKE CATALOG COMPLETENESS MAGNITUDE ANALYSIS")
    print("Universal Tool v2.0 - Enhanced for Robustness")
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
            m_th_values = sorted(m_th_values)  # Ensure sorted
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
            ks_threshold = float(input(f"KS threshold for completeness (default {DEFAULT_KS_THRESHOLD}): ") 
                               or str(DEFAULT_KS_THRESHOLD))
            min_events = int(input(f"Minimum events per threshold (default {DEFAULT_MIN_EVENTS}): ") 
                           or str(DEFAULT_MIN_EVENTS))
            verbose = input("Enable verbose output? (y/n, default: y): ").strip().lower() != 'n'
            save_figure = input("Save figure? (y/n, default: n): ").strip().lower() == 'y'
        except:
            ks_threshold = DEFAULT_KS_THRESHOLD
            min_events = DEFAULT_MIN_EVENTS
            verbose = True
            save_figure = False
    else:
        ks_threshold = DEFAULT_KS_THRESHOLD
        min_events = DEFAULT_MIN_EVENTS
        verbose = True
        save_figure = False
    
    # Perform analysis
    print("\nAnalyzing catalog...")
    mc = plot_magnitude_completeness(filename, m_th_values, year_col, month_col, 
                                    day_col, hour_col, minute_col, second_col, 
                                    mag_col, delimiter, auto_detect_mc=True,
                                    ks_threshold=ks_threshold, min_events=min_events,
                                    verbose=verbose, save_figure=save_figure)
    
    # Save results option
    if mc is not None:
        save = input("\nSave detailed results to file? (y/n): ").strip().lower()
        if save == 'y':
            output_file = f"{filename.split('.')[0]}_completeness_analysis.txt"
            with open(output_file, 'w') as f:
                f.write("="*60 + "\n")
                f.write("EARTHQUAKE CATALOG COMPLETENESS ANALYSIS REPORT\n")
                f.write("="*60 + "\n\n")
                f.write(f"Analysis Date: {datetime.now()}\n")
                f.write(f"Catalog File: {filename}\n")
                f.write(f"Completeness Magnitude (Mc): {mc:.2f}\n")
                f.write(f"KS Threshold Used: {ks_threshold}\n")
                f.write(f"Minimum Events Required: {min_events}\n")
                f.write(f"\nMagnitude Thresholds Tested: {m_th_values}\n")
                f.write("\nThis analysis used the normalized cumulative distribution method\n")
                f.write("(Petrillo & Zhuang, 2023) with Kolmogorov-Smirnov statistics.\n")
            print(f"Results saved to {output_file}")
    
    # Ask if user wants to perform another analysis
    another = input("\nPerform another analysis? (y/n): ").strip().lower()
    if another == 'y':
        interactive_analysis()

# Main program
if __name__ == "__main__":
    # Interactive mode
    interactive_analysis()
    
    # Example of direct usage with all parameters:
    # mc = plot_magnitude_completeness(
    #     'tangshan.reloc',
    #     m_th_values=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5, 3.0],
    #     auto_detect_mc=True,
    #     ks_threshold=0.05,
    #     min_events=50,
    #     verbose=True,
    #     save_figure=True
    # )