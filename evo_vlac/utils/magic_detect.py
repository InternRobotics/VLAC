import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from typing import Tuple, List, Optional, Dict
import warnings
from scipy import stats
from enum import Enum

def adaptive_peak_valley_detection(
    data: np.ndarray,
    assist_peaks: np.ndarray=None,
    window_size: Optional[int] = None,
    min_distance: Optional[int] = None,
    prominence_threshold: float = 0.1,
    max_iterations: int = 10,
    verbose: bool = False,
    outlier_sensitivity=1.5,
    min_amplitude_ratio=0.1
) -> Dict[str, np.ndarray]:
    """
    自适应峰谷检测函数，使用滚动窗口方法并通过多轮迭代优化
    
    Parameters:
    -----------
    data : np.ndarray
        输入时间序列数据
    window_size : int, optional
        滚动窗口大小，如果为None则自动计算
    min_distance : int, optional
        峰谷间最小距离，如果为None则自动计算
    prominence_threshold : float
        突出度阈值比例
    max_iterations : int
        最大迭代次数
    verbose : bool
        是否输出详细信息
        
    Returns:
    --------
    dict : 包含peaks, valleys, segments, window_size, iterations等信息
    """
    
    if len(data) < 3:
        raise ValueError("Data length must be >= 3")
    
    data = np.array(data)
    n = len(data)
    
    # 自适应窗口大小计算
    if window_size is None:
        window_size = _calculate_adaptive_window_size(data)
    
    # 自适应最小距离计算
    if min_distance is None:
        min_distance = max(1, window_size // 3)
    
    if verbose:
        print(f"Data length: {n}")
        print(f"Adaptive window size: {window_size}")
        print(f"Min distance: {min_distance}")
    
    # 第一轮：基于滚动窗口的初始检测
    if assist_peaks:
        initial_peaks=assist_peaks
        initial_valleys=[int((assist_peaks[k]+assist_peaks[k-1])/2)for k in range(1,len(assist_peaks))]
    else:
        initial_peaks, initial_valleys = _rolling_window_detection(
            data, window_size, min_distance, prominence_threshold
        )

    
    if verbose:
        print(f"Initial detection - Peaks: {len(initial_peaks)}, Valleys: {len(initial_valleys)}")
    
    # 多轮迭代优化
    peaks, valleys, iterations = _iterative_peak_valley_optimization(
        data, initial_peaks, initial_valleys, max_iterations, verbose,outlier_method='iqr',outlier_sensitivity=outlier_sensitivity,min_amplitude_ratio=min_amplitude_ratio
    )
    
    # 生成序列分段
    segments = _generate_segments(peaks, valleys, n)
    
    # 计算统计信息
    stats = _calculate_statistics(data, peaks, valleys)
    
    return {
        'peaks': peaks,
        'valleys': valleys,
        'segments': segments,
        'window_size': window_size,
        'min_distance': min_distance,
        'iterations': iterations,
        'stats': stats
    }


def _calculate_adaptive_window_size(data: np.ndarray) -> int:
    """Adaptive window size calculation based on first-order difference periodicity"""
    n = len(data)
    
    if n < 10:
        return 3
    
    # Calculate first-order difference
    diff_data = np.diff(data)
    diff_n = len(diff_data)
    
    try:
        # Method 1: Autocorrelation of first-order difference
        diff_centered = diff_data - np.mean(diff_data)
        autocorr = np.correlate(diff_centered, diff_centered, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        # Normalize autocorrelation
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]
        
        # Find first significant peak (excluding lag 0)
        # Look for peaks with minimum height and distance
        min_lag = max(2, diff_n // 50)  # Minimum lag to consider
        max_lag = min(diff_n // 3, 100)  # Maximum lag to consider
        
        if max_lag > min_lag:
            search_autocorr = autocorr[min_lag:max_lag]
            peaks_auto, properties = signal.find_peaks(
                search_autocorr, 
                height=0.1,  # Minimum correlation
                distance=max(1, min_lag // 2)
            )
            
            if len(peaks_auto) > 0:
                # First significant peak indicates period
                period = peaks_auto[0] + min_lag
                window_size = min(max(period // 2, 5), n // 4)
            else:
                # Fallback: find first local maximum
                for i in range(1, min(50, len(search_autocorr) - 1)):
                    if (search_autocorr[i] > search_autocorr[i-1] and 
                        search_autocorr[i] > search_autocorr[i+1] and
                        search_autocorr[i] > 0.05):
                        period = i + min_lag
                        window_size = min(max(period // 2, 5), n // 4)
                        break
                else:
                    window_size = min(max(int(np.sqrt(n)), 5), n // 4)
        else:
            window_size = min(max(int(np.sqrt(n)), 5), n // 4)
            
    except Exception as e:
        # Fallback method: FFT-based period detection on difference
        try:
            # Remove DC component
            diff_fft = np.fft.fft(diff_data - np.mean(diff_data))
            freqs = np.fft.fftfreq(diff_n)
            
            # Find dominant frequency (excluding DC)
            power_spectrum = np.abs(diff_fft[1:diff_n//2])
            if len(power_spectrum) > 0:
                dominant_freq_idx = np.argmax(power_spectrum) + 1
                dominant_freq = freqs[dominant_freq_idx]
                
                if dominant_freq > 0:
                    period = int(1 / dominant_freq)
                    window_size = min(max(period // 2, 5), n // 4)
                else:
                    window_size = min(max(int(np.sqrt(n)), 5), n // 4)
            else:
                window_size = min(max(int(np.sqrt(n)), 5), n // 4)
        except:
            window_size = min(max(int(np.sqrt(n)), 5), n // 4)
    
    # Method 2: Statistical approach on difference
    try:
        # Find significant changes in first-order difference
        diff_abs = np.abs(diff_data)
        threshold = np.mean(diff_abs) + 0.5 * np.std(diff_abs)
        change_points = np.where(diff_abs > threshold)[0]
        
        if len(change_points) > 2:
            # Calculate average distance between change points
            distances = np.diff(change_points)
            if len(distances) > 0:
                avg_distance = np.median(distances)  # Use median for robustness
                window_size_v2 = min(max(int(avg_distance), 5), n // 4)
                # Combine with autocorrelation result
                window_size = int((window_size + window_size_v2) / 2)
    except:
        pass
    
    # Ensure window size is odd and within reasonable bounds
    window_size = max(3, min(window_size, n // 3))
    if window_size % 2 == 0:
        window_size += 1
    
    return window_size


def _rolling_window_detection(
    data: np.ndarray, 
    window_size: int, 
    min_distance: int, 
    prominence_threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """基于滚动窗口的初始峰谷检测"""
    n = len(data)
    half_window = window_size // 2
    peaks = []
    valleys = []
    
    # 计算全局统计用于突出度判断
    global_std = np.std(data)
    threshold = global_std * prominence_threshold
    
    for i in range(half_window, n - half_window):
        # 提取窗口数据
        window_start = max(0, i - half_window)
        window_end = min(n, i + half_window + 1)
        window_data = data[window_start:window_end]
        window_indices = np.arange(window_start, window_end)
        
        current_value = data[i]
        window_max = np.max(window_data)
        window_min = np.min(window_data)
        
        # 检测峰值
        if (current_value == window_max and 
            current_value - window_min > threshold and
            (len(peaks) == 0 or i - peaks[-1] >= min_distance)):
            peaks.append(i)
        
        # 检测谷值
        elif (current_value == window_min and 
              window_max - current_value > threshold and
              (len(valleys) == 0 or i - valleys[-1] >= min_distance)):
            valleys.append(i)
    
    return np.array(peaks), np.array(valleys)


def _iterative_peak_valley_optimization(
    data: np.ndarray, 
    initial_peaks: np.ndarray, 
    initial_valleys: np.ndarray, 
    max_iterations: int, 
    verbose: bool,
    outlier_method: str = 'iqr',
    outlier_sensitivity: float = 1.5,
    min_amplitude_ratio: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    多轮迭代优化峰谷检测结果，并在结束后进行基于差值分布的后处理
    
    Parameters:
    -----------
    data : np.ndarray
        输入时间序列数据
    initial_peaks : np.ndarray
        初始峰值索引
    initial_valleys : np.ndarray
        初始谷值索引
    max_iterations : int
        最大迭代次数
    verbose : bool
        是否输出详细信息
    outlier_method : str
        离群点检测方法 ['iqr', 'zscore', 'percentile', 'mad', 'dbscan']
    outlier_sensitivity : float
        离群点检测灵敏度参数
    min_amplitude_ratio : float
        最小振幅比例（相对于数据标准差）
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, int] : 峰值索引, 谷值索引, 迭代次数
    """
    
    peaks = initial_peaks.copy()
    valleys = initial_valleys.copy()
    
    # 原有的迭代优化过程
    for iteration in range(max_iterations):
        old_peaks = peaks.copy()
        old_valleys = valleys.copy()
        
        # 合并所有关键点并排序
        all_points = []
        for p in peaks:
            all_points.append((p, 'peak', data[p]))
        for v in valleys:
            all_points.append((v, 'valley', data[v]))
        
        all_points.sort(key=lambda x: x[0])
        
        if len(all_points) < 2:
            break
        
        # 优化规则1: 确保峰谷交替
        optimized_points = _enforce_alternating_pattern(all_points, data)
        
        # 优化规则2: 确保峰是两谷间最高点，谷是两峰间最低点
        optimized_points = _optimize_local_extrema(optimized_points, data)
        
        # 分离峰谷
        new_peaks = []
        new_valleys = []
        for point in optimized_points:
            if point[1] == 'peak':
                new_peaks.append(point[0])
            else:
                new_valleys.append(point[0])
        
        peaks = np.array(new_peaks)
        valleys = np.array(new_valleys)
        
        if verbose:
            print(f"Iteration {iteration + 1}: Peaks {len(peaks)}, Valleys {len(valleys)}")
        
        # Check convergence
        if (np.array_equal(peaks, old_peaks) and 
            np.array_equal(valleys, old_valleys)):
            if verbose:
                print(f"Converged after {iteration + 1} iterations")
            break
    
    # 新增：基于峰谷差值分布的后处理
    if verbose:
        print("Starting post-processing based on peak-valley amplitude distribution...")
    
    peaks, valleys = _postprocess_amplitude_filtering(
        data, peaks, valleys, outlier_method, outlier_sensitivity, 
        min_amplitude_ratio, verbose
    )
    
    return peaks, valleys, iteration + 1


def _postprocess_amplitude_filtering(
    data: np.ndarray,
    peaks: np.ndarray,
    valleys: np.ndarray,
    outlier_method: str,
    outlier_sensitivity: float,
    min_amplitude_ratio: float,
    verbose: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于峰谷差值分布进行后处理，过滤掉差值过小的峰谷对
    """
    
    if len(peaks) == 0 or len(valleys) == 0:
        return peaks, valleys
    
    # 1. 计算相邻峰谷之间的差值
    peak_valley_pairs, amplitudes = _calculate_peak_valley_amplitudes(data, peaks, valleys)
    
    if len(amplitudes) == 0:
        return peaks, valleys
    
    if verbose:
        print(f"Found {len(amplitudes)} peak-valley pairs")
        print(f"Amplitude statistics: mean={np.mean(amplitudes):.3f}, std={np.std(amplitudes):.3f}")
        print(f"Amplitude range: [{np.min(amplitudes):.3f}, {np.max(amplitudes):.3f}]")
    
    # 2. 检测差值过小的离群点
    outlier_indices = _detect_amplitude_outliers(
        amplitudes, outlier_method, outlier_sensitivity, verbose
    )
    
    # 3. 应用最小振幅比例过滤
    data_std = np.std(data)
    min_amplitude = min_amplitude_ratio * data_std
    small_amplitude_indices = np.where(amplitudes < min_amplitude)[0]
    
    # 合并两种过滤方法的结果
    all_outlier_indices = np.unique(np.concatenate([outlier_indices, small_amplitude_indices]))
    
    if verbose and len(all_outlier_indices) > 0:
        print(f"Detected {len(outlier_indices)} statistical outliers")
        print(f"Detected {len(small_amplitude_indices)} small amplitude pairs (< {min_amplitude:.3f})")
        print(f"Total pairs to filter: {len(all_outlier_indices)}")
    
    # 4. 根据离群点删除相应的峰谷点
    if len(all_outlier_indices) > 0:
        peaks, valleys = _remove_outlier_peak_valley_pairs(
            data, peaks, valleys, peak_valley_pairs, all_outlier_indices, verbose
        )
    
    # 5. 确保最终结果满足峰谷相间且谷是峰间最低值的要求
    peaks, valleys = _final_peak_valley_validation(data, peaks, valleys, verbose)
    
    return peaks, valleys


def _calculate_peak_valley_amplitudes(
    data: np.ndarray, 
    peaks: np.ndarray, 
    valleys: np.ndarray
) -> Tuple[List[Tuple], np.ndarray]:
    """
    计算相邻峰谷之间的差值（振幅）
    
    Returns:
    --------
    peak_valley_pairs : List[Tuple]
        每个元素为 (peak_idx, valley_idx, amplitude, pair_type)
        pair_type: 'peak_to_valley' 或 'valley_to_peak'
    amplitudes : np.ndarray
        所有振幅值的数组
    """
    
    # 合并峰谷点并排序
    all_extrema = []
    for p in peaks:
        all_extrema.append((p, 'peak', data[p]))
    for v in valleys:
        all_extrema.append((v, 'valley', data[v]))
    
    all_extrema.sort(key=lambda x: x[0])
    
    peak_valley_pairs = []
    amplitudes = []
    
    # 计算相邻极值点之间的振幅
    for i in range(len(all_extrema) - 1):
        current = all_extrema[i]
        next_point = all_extrema[i + 1]
        
        # 只计算峰谷相邻的情况
        if current[1] != next_point[1]:
            amplitude = abs(current[2] - next_point[2])
            pair_type = f"{current[1]}_to_{next_point[1]}"
            
            peak_valley_pairs.append((
                current[0] if current[1] == 'peak' else next_point[0],  # peak_idx
                current[0] if current[1] == 'valley' else next_point[0],  # valley_idx
                amplitude,
                pair_type
            ))
            amplitudes.append(amplitude)
    
    return peak_valley_pairs, np.array(amplitudes)


def _detect_amplitude_outliers(
    amplitudes: np.ndarray,
    method: str,
    sensitivity: float,
    verbose: bool
) -> np.ndarray:
    """
    检测振幅中的离群点（差值过小的点）
    
    Parameters:
    -----------
    amplitudes : np.ndarray
        振幅数组
    method : str
        检测方法 ['iqr', 'zscore', 'percentile', 'mad', 'dbscan']
    sensitivity : float
        灵敏度参数
    verbose : bool
        是否输出详细信息
        
    Returns:
    --------
    np.ndarray : 离群点的索引
    """
    
    if len(amplitudes) < 3:
        return np.array([])
    
    outlier_indices = []
    
    if method == 'iqr':
        # IQR方法：检测下四分位数以下的异常小值
        q1 = np.percentile(amplitudes, 25)
        q3 = np.percentile(amplitudes, 75)
        iqr = q3 - q1
        lower_bound = q1 - sensitivity * iqr
        
        outlier_indices = np.where(amplitudes < lower_bound)[0]
        
        if verbose:
            print(f"IQR method: Q1={q1:.3f}, Q3={q3:.3f}, IQR={iqr:.3f}")
            print(f"Lower bound: {lower_bound:.3f}")
    
    elif method == 'zscore':
        # Z-score方法：检测标准化后绝对值过大的点（但这里我们关注小值）
        z_scores = np.abs(stats.zscore(amplitudes))
        mean_amp = np.mean(amplitudes)
        
        # 找出既是统计离群点又是小于均值的点
        small_values = amplitudes < mean_amp
        statistical_outliers = z_scores > sensitivity
        outlier_indices = np.where(small_values & statistical_outliers)[0]
        
        if verbose:
            print(f"Z-score method: mean={mean_amp:.3f}, threshold={sensitivity}")
    
    elif method == 'percentile':
        # 百分位数方法：直接取最小的sensitivity比例的点
        threshold_percentile = sensitivity * 100 if sensitivity <= 1 else sensitivity
        threshold_value = np.percentile(amplitudes, threshold_percentile)
        outlier_indices = np.where(amplitudes <= threshold_value)[0]
        
        if verbose:
            print(f"Percentile method: {threshold_percentile:.1f}th percentile = {threshold_value:.3f}")
    
    elif method == 'mad':
        # MAD (Median Absolute Deviation) 方法
        median_amp = np.median(amplitudes)
        mad = np.median(np.abs(amplitudes - median_amp))
        
        if mad > 0:
            modified_z_scores = 0.6745 * (amplitudes - median_amp) / mad
            # 找出负的modified z-scores中绝对值较大的（即明显小于中位数的）
            outlier_indices = np.where(modified_z_scores < -sensitivity)[0]
        else:
            outlier_indices = np.array([])
        
        if verbose:
            print(f"MAD method: median={median_amp:.3f}, MAD={mad:.3f}")
    
    elif method == 'dbscan':
        # DBSCAN聚类方法（需要sklearn）
        try:
            from sklearn.cluster import DBSCAN
            
            # 将振幅作为一维特征进行聚类
            X = amplitudes.reshape(-1, 1)
            
            # 调整eps参数基于sensitivity
            eps = sensitivity * np.std(amplitudes)
            min_samples = max(2, len(amplitudes) // 10)
            
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
            
            # 找出被标记为噪声的点（label = -1）和小振幅的聚类
            noise_points = np.where(labels == -1)[0]
            
            # 在非噪声点中，找出平均值最小的聚类
            unique_labels = np.unique(labels[labels != -1])
            if len(unique_labels) > 1:
                cluster_means = []
                for label in unique_labels:
                    cluster_indices = np.where(labels == label)[0]
                    cluster_mean = np.mean(amplitudes[cluster_indices])
                    cluster_means.append((label, cluster_mean))
                
                # 找出平均振幅最小的聚类
                min_cluster_label = min(cluster_means, key=lambda x: x[1])[0]
                min_cluster_indices = np.where(labels == min_cluster_label)[0]
                
                outlier_indices = np.concatenate([noise_points, min_cluster_indices])
            else:
                outlier_indices = noise_points
                
            if verbose:
                print(f"DBSCAN method: eps={eps:.3f}, min_samples={min_samples}")
                print(f"Found {len(unique_labels)} clusters and {len(noise_points)} noise points")
                
        except ImportError:
            if verbose:
                print("DBSCAN method requires sklearn, falling back to IQR method")
            return _detect_amplitude_outliers(amplitudes, 'iqr', sensitivity, verbose)
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
    
    if verbose and len(outlier_indices) > 0:
        outlier_amplitudes = amplitudes[outlier_indices]
        print(f"Detected {len(outlier_indices)} outliers with amplitudes: {outlier_amplitudes}")
    
    return np.array(outlier_indices)


def _remove_outlier_peak_valley_pairs(
    data: np.ndarray,
    peaks: np.ndarray,
    valleys: np.ndarray,
    peak_valley_pairs: List[Tuple],
    outlier_indices: np.ndarray,
    verbose: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据离群点索引删除相应的峰谷点
    """
    
    points_to_remove = set()
    
    for idx in outlier_indices:
        if idx < len(peak_valley_pairs):
            pair = peak_valley_pairs[idx]
            peak_idx, valley_idx = pair[0], pair[1]
            
            # 优先删除谷点，如果谷点被多个峰共享，则可能需要合并峰
            points_to_remove.add(('valley', valley_idx))
            
            if verbose:
                print(f"Marking for removal - Peak: {peak_idx}, Valley: {valley_idx}, "
                      f"Amplitude: {pair[2]:.3f}")
    
    # 删除标记的谷点
    valleys_to_keep = []
    for v in valleys:
        if ('valley', v) not in points_to_remove:
            valleys_to_keep.append(v)
    
    new_valleys = np.array(valleys_to_keep)
    
    # 处理峰点：如果相邻的峰之间的谷被删除了，需要合并峰点（保留更高的）
    new_peaks = _merge_adjacent_peaks(data, peaks, new_valleys, verbose)
    
    if verbose:
        print(f"After outlier removal: Peaks {len(peaks)} -> {len(new_peaks)}, "
              f"Valleys {len(valleys)} -> {len(new_valleys)}")
    
    return new_peaks, new_valleys


def _merge_adjacent_peaks(
    data: np.ndarray,
    peaks: np.ndarray,
    valleys: np.ndarray,
    verbose: bool
) -> np.ndarray:
    """
    合并相邻的峰点（当它们之间没有谷点时）
    """
    
    if len(peaks) <= 1:
        return peaks
    
    # 创建所有极值点的排序列表
    all_points = []
    for p in peaks:
        all_points.append((p, 'peak'))
    for v in valleys:
        all_points.append((v, 'valley'))
    
    all_points.sort(key=lambda x: x[0])
    
    # 找出相邻的峰点并合并
    merged_peaks = []
    i = 0
    
    while i < len(all_points):
        if all_points[i][1] == 'peak':
            # 收集连续的峰点
            consecutive_peaks = [all_points[i][0]]
            j = i + 1
            
            while j < len(all_points) and all_points[j][1] == 'peak':
                consecutive_peaks.append(all_points[j][0])
                j += 1
            
            # 在连续的峰点中保留最高的
            if len(consecutive_peaks) > 1:
                peak_values = [data[p] for p in consecutive_peaks]
                best_peak_idx = consecutive_peaks[np.argmax(peak_values)]
                merged_peaks.append(best_peak_idx)
                
                if verbose:
                    print(f"Merged {len(consecutive_peaks)} consecutive peaks, "
                          f"kept peak at index {best_peak_idx} with value {data[best_peak_idx]:.3f}")
            else:
                merged_peaks.append(consecutive_peaks[0])
            
            i = j
        else:
            i += 1
    
    return np.array(merged_peaks)


def _final_peak_valley_validation(
    data: np.ndarray,
    peaks: np.ndarray,
    valleys: np.ndarray,
    verbose: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    最终验证：确保峰谷相间且谷是峰间的最低值
    """
    
    if len(peaks) == 0 or len(valleys) == 0:
        return peaks, valleys
    
    # 创建交替的峰谷序列
    all_points = []
    for p in peaks:
        all_points.append((p, 'peak'))
    for v in valleys:
        all_points.append((v, 'valley'))
    
    all_points.sort(key=lambda x: x[0])
    
    # 确保峰谷交替
    validated_points = []
    if len(all_points) > 0:
        validated_points.append(all_points[0])
        
        for i in range(1, len(all_points)):
            current = all_points[i]
            prev = validated_points[-1]
            
            if current[1] != prev[1]:  # 类型不同，保留
                validated_points.append(current)
            else:  # 类型相同，保留更极端的
                if current[1] == 'peak':
                    if data[current[0]] > data[prev[0]]:
                        validated_points[-1] = current
                else:  # valley
                    if data[current[0]] < data[prev[0]]:
                        validated_points[-1] = current
    
    # 分离最终的峰谷
    final_peaks = []
    final_valleys = []
    
    for point in validated_points:
        if point[1] == 'peak':
            final_peaks.append(point[0])
        else:
            final_valleys.append(point[0])
    
    # 验证谷是峰间的最低值
    final_valleys = _validate_valleys_between_peaks(data, final_peaks, final_valleys, verbose)
    
    if verbose:
        print(f"Final validation: Peaks {len(peaks)} -> {len(final_peaks)}, "
              f"Valleys {len(valleys)} -> {len(final_valleys)}")
    
    return np.array(final_peaks), np.array(final_valleys)


def _validate_valleys_between_peaks(
    data: np.ndarray,
    peaks: np.ndarray,
    valleys: np.ndarray,
    verbose: bool
) -> np.ndarray:
    """
    验证每个谷点确实是相邻峰点之间的最低点
    """
    
    if len(peaks) < 2 or len(valleys) == 0:
        return valleys
    
    validated_valleys = []
    peaks_sorted = np.sort(peaks)
    
    for i in range(len(peaks_sorted) - 1):
        left_peak = peaks_sorted[i]
        right_peak = peaks_sorted[i + 1]
        
        # 找出在这两个峰之间的谷点
        between_valleys = [v for v in valleys if left_peak < v < right_peak]
        
        if len(between_valleys) == 0:
            # 如果没有谷点，在这个区间找最低点
            search_start = left_peak + 1
            search_end = right_peak
            if search_start < search_end:
                segment = data[search_start:search_end]
                min_idx = np.argmin(segment) + search_start
                validated_valleys.append(min_idx)
                if verbose:
                    print(f"Added missing valley at index {min_idx} between peaks {left_peak} and {right_peak}")
        
        elif len(between_valleys) == 1:
            # 验证这个谷点是否真的是最低的
            valley_idx = between_valleys[0]
            search_start = left_peak + 1
            search_end = right_peak
            
            if search_start < search_end:
                segment = data[search_start:search_end]
                true_min_idx = np.argmin(segment) + search_start
                
                if true_min_idx == valley_idx:
                    validated_valleys.append(valley_idx)
                else:
                    validated_valleys.append(true_min_idx)
                    if verbose:
                        print(f"Corrected valley position from {valley_idx} to {true_min_idx}")
            else:
                validated_valleys.append(valley_idx)
        
        else:
            # 多个谷点，保留最低的
            valley_values = [data[v] for v in between_valleys]
            best_valley = between_valleys[np.argmin(valley_values)]
            validated_valleys.append(best_valley)
            
            if verbose:
                print(f"Multiple valleys between peaks {left_peak} and {right_peak}, "
                      f"kept valley at {best_valley}")
    
    return np.array(validated_valleys)


# 需要同时导入的辅助函数（保持原有实现）
def _enforce_alternating_pattern(all_points: List, data: np.ndarray) -> List:
    """确保峰谷交替出现"""
    if len(all_points) < 2:
        return all_points
    
    optimized = [all_points[0]]
    
    for i in range(1, len(all_points)):
        current = all_points[i]
        prev = optimized[-1]
        
        # 如果类型相同，保留值更极端的点
        if current[1] == prev[1]:
            if current[1] == 'peak':
                # 保留更高的峰
                if current[2] > prev[2]:
                    optimized[-1] = current
            else:
                # 保留更低的谷
                if current[2] < prev[2]:
                    optimized[-1] = current
        else:
            optimized.append(current)
    
    return optimized


def _optimize_local_extrema(points: List, data: np.ndarray) -> List:
    """优化局部极值点位置"""
    if len(points) < 3:
        return points
    
    optimized = [points[0]]
    
    for i in range(1, len(points) - 1):
        current = points[i]
        prev_idx = optimized[-1][0]
        next_idx = points[i + 1][0]
        
        # 在相邻点之间寻找真正的极值
        search_start = max(prev_idx + 1, current[0] - 5)
        search_end = min(next_idx, current[0] + 6)
        
        if search_start >= search_end:
            optimized.append(current)
            continue
        
        search_range = range(search_start, search_end)
        search_data = data[search_start:search_end]
        
        if current[1] == 'peak':
            # 寻找最高点
            max_idx = np.argmax(search_data)
            actual_idx = search_start + max_idx
            optimized.append((actual_idx, 'peak', data[actual_idx]))
        else:
            # 寻找最低点
            min_idx = np.argmin(search_data)
            actual_idx = search_start + min_idx
            optimized.append((actual_idx, 'valley', data[actual_idx]))
    
    optimized.append(points[-1])
    


def _enforce_alternating_pattern(all_points: List, data: np.ndarray) -> List:
    """确保峰谷交替出现"""
    if len(all_points) < 2:
        return all_points
    
    optimized = [all_points[0]]
    
    for i in range(1, len(all_points)):
        current = all_points[i]
        prev = optimized[-1]
        
        # 如果类型相同，保留值更极端的点
        if current[1] == prev[1]:
            if current[1] == 'peak':
                # 保留更高的峰
                if current[2] > prev[2]:
                    optimized[-1] = current
            else:
                # 保留更低的谷
                if current[2] < prev[2]:
                    optimized[-1] = current
        else:
            optimized.append(current)
    
    return optimized


def _optimize_local_extrema(points: List, data: np.ndarray) -> List:
    """优化局部极值点位置"""
    if len(points) < 3:
        return points
    
    optimized = [points[0]]
    
    for i in range(1, len(points) - 1):
        current = points[i]
        prev_idx = optimized[-1][0]
        next_idx = points[i + 1][0]
        
        # 在相邻点之间寻找真正的极值
        search_start = max(prev_idx + 1, current[0] - 5)
        search_end = min(next_idx, current[0] + 6)
        
        if search_start >= search_end:
            optimized.append(current)
            continue
        
        search_range = range(search_start, search_end)
        search_data = data[search_start:search_end]
        
        if current[1] == 'peak':
            # 寻找最高点
            max_idx = np.argmax(search_data)
            actual_idx = search_start + max_idx
            optimized.append((actual_idx, 'peak', data[actual_idx]))
        else:
            # 寻找最低点
            min_idx = np.argmin(search_data)
            actual_idx = search_start + min_idx
            optimized.append((actual_idx, 'valley', data[actual_idx]))
    
    optimized.append(points[-1])
    
    return optimized


def _generate_segments(peaks: np.ndarray, valleys: np.ndarray, data_length: int) -> List[Tuple[int, int]]:
    """生成序列分段"""
    all_points = []
    for p in peaks:
        all_points.append(p)
    for v in valleys:
        all_points.append(v)
    
    all_points = sorted(all_points)
    
    segments = []
    start = 0
    
    for point in all_points:
        if point > start:
            segments.append((start, point))
            start = point
    
    # 添加最后一段
    if start < data_length - 1:
        segments.append((start, data_length - 1))
    
    return segments


def _calculate_statistics(data: np.ndarray, peaks: np.ndarray, valleys: np.ndarray) -> Dict:
    """计算统计信息"""
    stats = {
        'peak_count': len(peaks),
        'valley_count': len(valleys),
        'peak_values': data[peaks] if len(peaks) > 0 else np.array([]),
        'valley_values': data[valleys] if len(valleys) > 0 else np.array([]),
    }
    
    if len(peaks) > 0:
        stats['avg_peak_value'] = np.mean(data[peaks])
        stats['max_peak_value'] = np.max(data[peaks])
        stats['min_peak_value'] = np.min(data[peaks])
    
    if len(valleys) > 0:
        stats['avg_valley_value'] = np.mean(data[valleys])
        stats['max_valley_value'] = np.max(data[valleys])
        stats['min_valley_value'] = np.min(data[valleys])
    
    # 计算平均间距
    if len(peaks) > 1:
        stats['avg_peak_distance'] = np.mean(np.diff(peaks))
    if len(valleys) > 1:
        stats['avg_valley_distance'] = np.mean(np.diff(valleys))
    
    return stats


def plot_results(data: np.ndarray, result: Dict, title: str = "Peak Valley Detection", save_path: str = None):
    """Plot detection results"""
    plt.figure(figsize=(15, 8))
    
    # Main plot
    plt.subplot(2, 1, 1)
    plt.plot(data, 'b-', linewidth=1.5, label='Original Data', alpha=0.7)
    
    peaks = result['peaks']
    valleys = result['valleys']
    
    if len(peaks) > 0:
        # Ensure peaks are valid indices
        valid_peaks = peaks[peaks < len(data)]
        if len(valid_peaks) > 0:
            plt.plot(valid_peaks, [data[k] for k in valid_peaks], 'ro', 
                    markersize=8, label=f'Peaks ({len(valid_peaks)})')
    
    if len(valleys) > 0:
        # Ensure valleys are valid indices
        valid_valleys = valleys[valleys < len(data)]
        if len(valid_valleys) > 0:
            plt.plot(valid_valleys, [data[k] for k in valid_valleys], 'go', 
                    markersize=8, label=f'Valleys ({len(valid_valleys)})')
    
    plt.title(f'{title} (Window: {result["window_size"]}, Iterations: {result["iterations"]})')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Segmentation plot
    plt.subplot(2, 1, 2)
    segments = result['segments']
    colors = plt.cm.Set3(np.linspace(0, 1, max(len(segments), 1)))
    
    for i, (start, end) in enumerate(segments):
        # Ensure segment indices are valid
        start = max(0, min(start, len(data) - 1))
        end = max(start, min(end, len(data) - 1))
        
        if start < end:
            plt.plot(range(start, end + 1), data[start:end + 1], 
                    color=colors[i % len(colors)], linewidth=2, label=f'Segment {i+1}')
        elif start == end:
            plt.plot([start], [data[start]], 'o',
                    color=colors[i % len(colors)], markersize=6, label=f'Segment {i+1}')
    
    plt.title('Sequence Segmentation')
    plt.xlabel('Time')
    plt.ylabel('Value')
    if len(segments) <= 10:  # Only show legend if not too many segments
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def demo():
    """Demo function"""

    value_list=[0.0, 0.2, -0.02, -0.04, 0.141, -0.059, 0.301, 0.7, 1.137, 1.769, 2.418, 2.906, 3.333, 3.623, 4.567, 4.644, 6.15, 6.638, 7.777, 11.19, 12.895, 13.853, 13.853, 12.854, 11.599, 9.813, 8.298, 7.454, 6.159, 7.491, 6.714, 5.707, 5.707, 7.027, 3.624, 1.831, 2.283, 3.162, 0.799, 1.593, 1.593, 8.6, 12.31, 15.73, 19.388, 21.564, 18.333, 15.311, 11.856, 9.317, 0.811, -12.818, -13.81, -14.243, -14.563, -14.792, -14.562, -14.241, -13.762, -13.102, -13.532, -13.804, -14.032, -14.26, -14.031, -13.803, -13.576, -13.803, -13.621, -13.621, -13.393, -13.824, -14.553, -14.301, -14.004, -14.232, -13.82, -13.502, -8.576, -9.444, -9.444, -9.576, -9.598, -9.861, -9.751, -9.795, -9.114, -9.55, -9.769, -9.55, -10.054, -9.724, -9.987, -9.833, -10.053, -10.273, -10.559, -10.958, -11.179, -10.89, -10.669, -10.447, -10.204, -10.204, -9.984, -9.588, -8.843, -8.56, -7.561, -6.464, -6.464, -6.464, -5.846, -5.571, -5.149, -4.749, -4.267, -4.059, -3.851, -3.955, -3.726, -3.498, -3.291, -3.084, -2.837, -1.52, -1.317, -0.811, -0.226, 0.295, 1.312, 1.707, 2.788, 3.877, 11.278, 10.444, 10.444, 9.119, 9.119, 9.119, 12.045, 15.475, 18.079, 16.653, 13.92, 16.571, 14.335, 11.217, -0.414, -0.414, -0.856, -0.614, -0.412, 0.13, 0.829, 0.829, 0.829, 0.829, 0.829, 0.829, 0.829, 0.829, 0.829, 0.829, 1.682, 1.682, 1.682, 1.996, 2.408, 2.896, 3.129, 3.323, 3.516, 3.709, 3.921, 3.729, 3.921, 4.229, 4.803, 5.565, 6.415, 7.519, 8.352, 8.352, 8.352, 8.755, 8.755, 8.755, 8.755, 9.813, 10.787, 10.413, 10.95, 11.342, 11.324, 11.271, 11.129, 10.987, 10.88, 10.506, 10.148, 10.615, 11.187, 11.844, 12.373, 13.022, 13.613, 13.959, 14.303, 15.108, 20.66, 20.041, 20.041, 20.041, 20.041, 21.321, 23.713, 25.559, 28.015, 26.691, 25.166, 23.894, 25.066, 15.429, 15.598, 15.784, 15.616, 15.548, 15.227, 15.058, 14.888, 14.718, 14.547, 14.718, 14.889, 15.093, 14.923, 14.753, 14.923, 15.093, 15.263, 15.839, 15.671, 15.84, 15.671, 15.469, 15.638, 15.638, 16.195, 16.379, 16.195, 16.195, 16.38, 16.58, 17.031, 17.529, 18.106, 18.319, 18.531, 18.759, 18.971, 19.116, 19.246, 20.215, 20.215, 20.534, 20.979, 21.358, 22.79, 23.577, 22.813, 23.338, 23.645, 23.645, 24.316, 24.316, 25.83, 25.83, 25.83, 25.83, 28.708, 30.889, 32.161, 31.089, 29.642, 24.351, 25.319, 26.231, 22.837, 22.39, 22.033, 21.814, 22.002, 22.189, 22.454, 22.624, 22.795, 23.011, 22.564, 22.502, 22.719, 23.121, 23.213, 23.398, 23.612, 23.887, 24.496, 24.813, 25.565, 26.801, 32.159, 33.76, 33.839, 33.839, 33.839, 34.713, 35.796, 37.402, 36.513, 35.675, 36.009, 36.534, 36.762, 37.003, 37.683, 38.032, 38.032, 38.317, 38.662, 39.448, 36.614, 36.322, 38.907, 38.907, 38.907, 38.907, 38.907, 40.435, 42.496, 43.347, 42.35, 40.816, 31.737, 30.932, 18.983, 18.691, 18.529, 18.675, 18.919, 19.519, 19.68, 19.889, 20.049, 20.417, 20.035, 20.195, 20.419, 20.848, 21.133, 21.465, 22.124, 23.028, 23.643, 24.544, 23.397, 23.673, 24.498, 24.498, 23.622, 23.622, 23.622, 24.92, 26.302, 26.921, 26.029, 26.429, 26.576, 26.884, 27.25, 27.686, 27.325, 27.063, 28.449, 29.179, 30.085, 30.728, 31.31, 31.942, 31.942, 37.033, 38.683, 41.307, 40.391, 39.318, 38.505, 32.454, 22.12, 21.948, 21.792, 22.027, 22.183, 22.494, 22.727, 22.835, 22.989, 23.143, 23.297, 23.481, 23.221, 23.236, 23.236, 23.482, 23.91, 24.215, 24.381, 25.243, 27.202, 30.871, 30.871, 30.871, 30.871, 33.111, 35.465, 35.904, 36.058, 36.314, 36.569, 36.759, 37.062, 37.314, 37.527, 37.765, 38.648, 39.029, 39.565, 40.303, 41.186, 41.516, 42.007, 42.575, 45.63, 45.63, 45.63, 45.63, 46.304, 47.958, 48.864, 47.596, 46.631, 45.425, 35.842, 35.649, 35.495, 35.714, 35.984, 36.151, 36.546, 36.851, 37.104, 37.506, 37.719, 38.217, 38.353, 38.772, 39.886, 40.536, 40.785, 42.68, 45.007, 45.007, 46.041, 47.164, 47.597, 47.859, 48.099, 48.317, 48.544, 48.822, 49.016, 49.129, 49.332, 49.444, 49.626, 50.24, 50.688, 51.507, 51.72, 53.506, 53.218, 53.218, 53.218, 53.218, 55.791, 57.506, 58.305, 57.288, 56.545, 55.78, 49.501, 49.662, 49.562, 49.662, 49.481, 49.582, 49.683, 49.864, 49.995, 50.275, 50.662, 50.899, 50.998, 50.89, 51.528, 51.722, 51.877, 51.982, 52.117, 52.117, 52.222, 52.71, 53.059, 53.059, 53.407, 53.407, 53.407, 53.407, 55.317, 55.407, 55.041, 54.933, 54.735, 54.554, 54.772, 54.916, 55.025, 54.935, 55.034, 55.807, 55.63, 55.453, 55.453, 55.631, 55.8, 56.489, 55.985, 55.069, 55.438, 57.167, 57.167, 57.167, 57.886, 58.493, 59.854, 60.633, 59.924, 58.858, 51.535, 51.642, 51.477, 51.293, 51.117, 50.961, 50.863, 50.863, 50.529, 50.638, 50.391, 50.54, 50.441, 50.342, 50.332, 50.332, 50.332, 50.332, 50.332]
    data=value_list
    
    print("=" * 50)
    print("Adaptive Peak Valley Detection Demo")
    print("=" * 50)
    
    # Run detection
    result = adaptive_peak_valley_detection(
        data, 
        window_size=None,  # Adaptive window size
        verbose=True,
        min_distance=10,
        prominence_threshold=0.01,
    )
    
    print(f"\nDetection Results:")
    print(f"Peak count: {result['stats']['peak_count']}")
    print(f"Valley count: {result['stats']['valley_count']}")
    print(f"Segment count: {len(result['segments'])}")
    
    if result['stats']['peak_count'] > 0:
        print(f"Average peak value: {result['stats']['avg_peak_value']:.2f}")
        print(f"Average peak distance: {result['stats'].get('avg_peak_distance', 0):.1f}")
    
    if result['stats']['valley_count'] > 0:
        print(f"Average valley value: {result['stats']['avg_valley_value']:.2f}")
        print(f"Average valley distance: {result['stats'].get('avg_valley_distance', 0):.1f}")
    
    # Plot results
    plot_results(data, result, save_path='demo_result.png')  # Set save_path if needed
    
    return result


if __name__ == "__main__":
    demo()