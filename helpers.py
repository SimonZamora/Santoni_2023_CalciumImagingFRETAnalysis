# This contains all the functions called in the jupyter notebook 1b_ProcessingCalcium.ipynb
# It implements a variety of algorithms to process and extract information from a calcium imaging recording
# Feel free to reach out if you have any questions!


import pandas as pd
import numpy as np
from scipy.signal import find_peaks


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    from https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
    Call in a loop to create terminal progress bar
    @params:
        iteration  £ - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


"""
 ________  ________     ___ ________ ________
|\   ___ \|\  _____\   /  /|\  _____\\   __  \
\ \  \_|\ \ \  \__/   /  //\ \  \__/\ \  \|\  \
 \ \  \ \\ \ \   __\ /  //  \ \   __\\ \  \\\  \
  \ \  \_\\ \ \  \_|/  //    \ \  \_| \ \  \\\  \
   \ \_______\ \__\/_ //      \ \__\   \ \_______\
    \|_______|\|__|__|/        \|__|    \|_______|



"""

def smoothing_FDM(trace, trend_smooth, recording_intervals):
    """
    Apply Finite Difference Method (FDM) smoothing to a signal within specified recording intervals.

    Parameters:
    trace (numpy.ndarray): Input trace data.
    trend_smooth (int): Number of trend smoothing iterations.
    recording_intervals (list): List of recording intervals.

    Returns:
    numpy.ndarray: Smoothed trace data.
    """
    # Make a copy of the input trace
    final_smoothed_trace = trace.copy()

    # Iterate over each recording interval
    for k in range(len(recording_intervals) - 1):
        # Get the current recording interval
        interval_start, interval_end = recording_intervals[k], recording_intervals[k + 1]

        # Extract the trace within the current interval
        interval_trace = trace[interval_start:interval_end].copy()

        # Apply trend smoothing iterations
        for _ in range(trend_smooth * 4):
            # Apply endpoint smoothing
            interval_trace[0] = interval_trace[0] + (2 * interval_trace[1] - 2 * interval_trace[0]) / 4
            interval_trace[-1] = interval_trace[-1] + (2 * interval_trace[-2] - 2 * interval_trace[-1]) / 4

            # Apply internal smoothing
            for j in range(1, len(interval_trace) - 1):
                interval_trace[j] = interval_trace[j] + (
                            interval_trace[j - 1] + interval_trace[j + 1] - 2 * interval_trace[j]) / 4

        # Update the final smoothed trace with the smoothed interval trace
        final_smoothed_trace[interval_start:interval_end] = interval_trace

    return final_smoothed_trace




def diff_in_interval(trace, intervals, degree):
    """
    Compute the derivative of each signal interval by interval.

    Parameters:
    trace (numpy.ndarray): Input trace data.
    intervals (list): List of intervals for signal processing.
    degree (int): Degree of the derivative to compute.

    Returns:
    numpy.ndarray: Derivative of the input trace data.
    """
    final_trace = []  # List to store the final derivative trace
    trace_considered = []  # Temporary storage for the trace data within each interval

    for i in range(len(intervals) - 1):
        # Compute the derivative of the trace within each interval
        trace_considered = np.diff(trace[intervals[i]:intervals[i + 1]], degree)

        # Append the derivative of the current interval to the final trace
        final_trace = np.append(final_trace, trace_considered)

    return final_trace



def find_peaks_in_interval(y_diff, intervals, prominence):
    """
    Find peaks in each signal interval, considering the pattern of interest.

    Parameters:
    y_diff (numpy.ndarray): Derivative of the input signal.
    intervals (list): List of intervals for signal processing.
    prominence (float): Minimum prominence of peaks.

    Returns:
    list: List of peak indices.
    """
    final_peaks = []  # List to store the final peaks

    for i in range(len(intervals) - 1):
        # Extract the derivative data within the current interval
        y_diff_considered = y_diff[intervals[i]:intervals[i + 1]]

        # Find peaks within the interval with specified prominence
        peaks, _ = find_peaks(y_diff_considered, prominence=prominence)

        # Check each peak and add it to the final list if it's above the threshold
        for peak in peaks:
            if y_diff_considered[peak] > 0:
                final_peaks.append(peak + intervals[i])  # Adjust peak index relative to the whole signal

    return final_peaks


def return_triplet(Peaks_sorted, intervals):
    """
    Search for the pattern [2, 1, 2] in the sorted peaks.

    Parameters:
    Peaks_sorted (pandas.DataFrame): DataFrame containing sorted peaks with their differences.
    intervals (list): List of intervals for signal processing.

    Returns:
    list: List of triplets satisfying the pattern.
    """
    solns = []  # List to store potential solutions
    values = np.array(Peaks_sorted['Diff'])  # Array of peak differences
    values_index = np.array(Peaks_sorted['Indexs'])  # Array of peak indices

    searchval = [2, 1, 2]  # Desired pattern to search for
    N = len(searchval)
    possibles = np.where(values == searchval[0])[0]  # Find potential starting indices

    triplet = []  # List to store valid triplets

    # Iterate through potential starting indices
    for p in possibles:
        check = values[p:p+N]  # Extract subarray for pattern comparison
        if len(check) == N:
            if np.all(check == searchval):
                solns.append(p)  # Store potential solution index

    if len(solns) != 0:
        # Iterate through potential solutions
        for pot_triplet in solns:
            trip = values_index[pot_triplet:pot_triplet+N]  # Extract potential triplet
            real_trip = True  # Flag for valid triplet

            # Check if any of the peaks are the same
            if (trip[0] == trip[1] or trip[1] == trip[2]):
                real_trip = False

            # Check if the triplet indices are within valid recording intervals
            for interv in intervals:
                if (trip[0] - interv) * (trip[2] - interv) < 0 or (trip[2]-trip[0])>300: #windows that are too large are errors, we discard them
                    real_trip = False
                    break

            if real_trip:
                triplet.append(trip)  # Add valid triplet to the list

    return triplet



def flatten_signal(smoothed_trace_30, smoothed_trace_60, intervals, thresh_diff1=0.2, thresh_diff2=0.02):
    """
    Flatten the given pre-smoothed traces using first and second derivatives.

    Parameters:
    smoothed_trace_30 (numpy.ndarray): Pre-smoothed trace data with smoothing trend of 30
    smoothed_trace_60 (numpy.ndarray): Pre-smoothed trace data with smoothing trend of 60
    intervals (list): List of intervals for signal processing.
    thresh_diff1 (float): Threshold for identifying peaks in the first derivative.
    thresh_diff2 (float): Threshold for identifying peaks in the second derivative.

    Returns:
    numpy.ndarray: Flattened trace data.
    list: List of triplets indicating segments replaced by slopes.
    """
    # Compute the first derivative and associated peaks for smoothed_trace_30
    trace_diff_1 = diff_in_interval(smoothed_trace_30, intervals, 1)
    trace_peak_1 = find_peaks_in_interval(trace_diff_1, intervals, thresh_diff1)

    # Compute the second derivative and associated peaks for smoothed_trace_30
    trace_diff_2 = diff_in_interval(smoothed_trace_30, intervals, 2)
    trace_peak_2 = find_peaks_in_interval(trace_diff_2, intervals, thresh_diff2)

    # Initialize new trace with smoothed_trace_60
    new_trace = smoothed_trace_60.copy()
    triplet = []  # List to store triplets indicating segments replaced by slopes

    # Check if both types of peaks are found
    if len(trace_peak_1) != 0 and len(trace_peak_2) != 0:
        # Combine peaks and their associated derivatives into a DataFrame
        peaks_df = pd.DataFrame({'Indexs': [], 'Diff': []})
        for peak1 in trace_peak_1:
            peaks_df = peaks_df.append({'Indexs': peak1, 'Diff': 1}, ignore_index=True)
        for peak2 in trace_peak_2:
            peaks_df = peaks_df.append({'Indexs': peak2, 'Diff': 2}, ignore_index=True)
        
        # Sort peaks based on their indices
        peaks_sorted = peaks_df.sort_values(by='Indexs')

        # Get triplets of segments to be replaced by slopes
        triplet = return_triplet(peaks_sorted, intervals)

    # If triplets are found, replace segments with slopes
    if len(triplet) != 0:
        for trip in triplet:
            # Calculate the slope between two points and replace the segment with the slope
            interval_length = int(trip[2] - trip[0])
            a = smoothed_trace_60[int(trip[0])]
            b = smoothed_trace_60[int(trip[2])]
            slope = (b - a) / interval_length
            x = np.linspace(0, interval_length, interval_length + 1)
            new_segment = x * slope + a
            new_trace[int(trip[0]):int(trip[2]) + 1] = new_segment

    return new_trace,triplet



def compute_Fbaseline(cell, recording_intervals):
    """
    Compute the baseline fluorescence (F0) for a given cell.

    Parameters:
    cell (dict): Dictionary containing cell data including fluorescence signal.
    recording_intervals (list): List of intervals for signal processing.

    Returns:
    np.ndarray: Baseline fluorescence (F0) of the cell.
    """
    # Smooth the fluorescence signal with different trend smoothing parameters
    cell_smoothed30 = smoothing_FDM(cell["F"], 30, recording_intervals)
    cell_smoothed60 = smoothing_FDM(cell["F"], 60, recording_intervals)
    
    # Flatten the smoothed signals to compute baseline fluorescence (F0)
    cell_Fbaseline,triplet = flatten_signal(cell_smoothed30, cell_smoothed60, recording_intervals)
    cell["windows"]=triplet
    return cell_Fbaseline


"""
 ________  _______   ________  ___  __            ________  _______  _________  _______   ________ _________  ___  ________  ________
|\   __  \|\  ___ \ |\   __  \|\  \|\  \         |\   ___ \|\  ___ \|\___   ___\\  ___ \ |\   ____\\___   ___\\  \|\   __  \|\   ___  \
\ \  \|\  \ \   __/|\ \  \|\  \ \  \/  /|_       \ \  \_|\ \ \   __/\|___ \  \_\ \   __/|\ \  \___\|___ \  \_\ \  \ \  \|\  \ \  \\ \  \
 \ \   ____\ \  \_|/_\ \   __  \ \   ___  \       \ \  \ \\ \ \  \_|/__  \ \  \ \ \  \_|/_\ \  \       \ \  \ \ \  \ \  \\\  \ \  \\ \  \
  \ \  \___|\ \  \_|\ \ \  \ \  \ \  \\ \  \       \ \  \_\\ \ \  \_|\ \  \ \  \ \ \  \_|\ \ \  \____   \ \  \ \ \  \ \  \\\  \ \  \\ \  \
   \ \__\    \ \_______\ \__\ \__\ \__\\ \__\       \ \_______\ \_______\  \ \__\ \ \_______\ \_______\  \ \__\ \ \__\ \_______\ \__\\ \__\
    \|__|     \|_______|\|__|\|__|\|__| \|__|        \|_______|\|_______|   \|__|  \|_______|\|_______|   \|__|  \|__|\|_______|\|__| \|__|



"""

def amplitude_of_event(cell, prediction, lookback=20):
    """
    Computes the amplitude of each event in peak_list based on the raw trace F

    Parameters:
        cell (dict): A dictionary containing the raw trace data.
        prediction (array-like): An array containing the indices of detected peaks.
        lookback (int, optional): The number of data points to look back to find the minimum value before the peak.

    Returns:
        np.array: An array containing the computed amplitudes of the events.
    """
    amplitude_list = []

    # Iterate over each peak in the prediction
    for peak in prediction:
        # Find the minimum value before the peak within the specified lookback window
        min_before = np.min(cell["F"][peak - lookback:peak])
        # Compute the amplitude by subtracting the minimum value from the peak value
        amplitude_list.append(cell["F"][peak] - min_before)

    return np.array(amplitude_list)


def find_peak_from_prediction(cell, prediction, promin):
    """
    Finds peaks in the raw trace corresponding to peaks in the prediction trace.

    Parameters:
        cell (dict): A dictionary containing the raw trace 
        prediction (array): Array containing the prediction trace.
        promin (float): Minimum prominence required for peak detection in the prediction trace.

    Returns:
        tuple: A tuple containing two arrays - the indices of peaks in the prediction trace
               and the corresponding adjusted indices of peaks in the raw trace.
    """
    trace = cell["F"]

    # Run peak detection on the prediction trace, allowing a maximum distance of 4 between peaks
    prediction_peak, _ = find_peaks(prediction, distance=4, prominence=promin)

    # Initialize arrays to store the detected peaks
    prediction_peak_final = []
    peak_F = []

    # Iterate over each peak detected in the prediction trace
    for peak in prediction_peak:
        i = 0

        # Move the peak to the position where the raw trace starts to decrease
        while trace[peak + i] < trace[peak + i + 1]:
            i += 1

        # Check for duplicates and select the peak with the highest prediction value, this enforce that only one peak is attributed to each calcium event
        if (peak + i) in peak_F:
            if prediction[peak] > prediction[prediction_peak_final[-1]]:
                del prediction_peak_final[-1]
                del peak_F[-1]
                prediction_peak_final.append(peak)
                peak_F.append(peak + i)
        else:
            peak_F.append(peak + i)
            prediction_peak_final.append(peak)

    return np.array(prediction_peak_final), np.array(peak_F)

"""
 ________ ___  ___  ________  _______           ________  _______   ________  ___  __    ________
|\  _____\\  \|\  \|\   ____\|\  ___ \         |\   __  \|\  ___ \ |\   __  \|\  \|\  \ |\   ____\
\ \  \__/\ \  \\\  \ \  \___|\ \   __/|        \ \  \|\  \ \   __/|\ \  \|\  \ \  \/  /|\ \  \___|_
 \ \   __\\ \  \\\  \ \_____  \ \  \_|/__       \ \   ____\ \  \_|/_\ \   __  \ \   ___  \ \_____  \
  \ \  \_| \ \  \\\  \|____|\  \ \  \_|\ \       \ \  \___|\ \  \_|\ \ \  \ \  \ \  \\ \  \|____|\  \
   \ \__\   \ \_______\____\_\  \ \_______\       \ \__\    \ \_______\ \__\ \__\ \__\\ \__\____\_\  \
    \|__|    \|_______|\_________\|_______|        \|__|     \|_______|\|__|\|__|\|__| \|__|\_________\
                      \|_________|                                                         \|_________|


"""
def extract_peak_window(cell, detected_peaks,associated_probXamp):
    """
    Extract windows (computed in flatten signal) around peaks from the cell data based on peak information.

    Parameters:
        cell (dict): A dictionary containing information about the cell.
        detected_peaks (list): list of peak detected so far
        associated_probXamp (list): list of probXamp associated with each peak

    Returns:
        list: A list of windows around peaks.
    """
    windows_list = []

    # Check if there are any windows in the cell data
    if len(cell["windows"]) != 0:
        # Iterate over each window in the cell data
        for flat_line in cell["windows"]:
            peak_in_window = []
            prob_in_window = []

            # Iterate over each peak in the peak information
            for i in range(len(detected_peaks)):
                peak = detected_peaks[i]
                

                # Check if the peak falls within the window 
                if (flat_line[0] - peak) * (flat_line[2] - peak) < 0:
                    peak_in_window.append(peak)
                    prob_in_window.append(associated_probXamp[i])

            # If there are peaks within the window, add the window to the list
            if len(peak_in_window) != 0:
                nb_peak = np.ones(len(peak_in_window), dtype=bool)
                windows_list.append([peak_in_window, flat_line, prob_in_window, [],nb_peak]) #####

    return windows_list

def fuse_windows(windows, cell, factor):
    """
    Fuse peaks in a given window based on proximity of peaks.

    Parameters:
        windows (list): A list of windows around peaks.
        cell (dict): A dictionary containing information about the cell.
        factor (float): A factor to determine amount of fluctuation needed to avoid fusing.

    Returns:
        list: A list of fused windows.
    """
    for window in windows:
        peak_list = window[0]
        if len(peak_list) > 1:

            # Extract the start and the end of the window
            begin_w = int(window[1][0])
            end_w = int(window[1][2])

            # Get the raw signal in this window and the computed baseline F0 signal
            trace_window = cell["F0"][begin_w:end_w]
            trace_peaks = cell["F"][begin_w:end_w]

            fuses = []
            for i in range(len(peak_list) - 1):
                # Always consider one peak and the next one

                first_peak = int(peak_list[i] - begin_w)
                last_peak = int(peak_list[i + 1] - begin_w)

                # The area between the raw signal and the F0
                dist = trace_peaks - trace_window
                fuse = True

                # If both peak have enough amplitude -> that the area between F and F0 from the first peak and the second
                # is greater than factor > after one peak, the calcium signal has gone back to less than 50% of the spike amplitude
                # -> we consider those as two separate events and we don't fuse

                for area_under_the_curve in dist[first_peak:last_peak]:
                    if area_under_the_curve < factor * dist[first_peak] or area_under_the_curve < factor * dist[last_peak]:
                        fuse = False
                fuses.append(fuse)

            # Store if peaks need to be fused or not
            window[3] = fuses

    return windows


def keep_most_prob_in_window(windows):
    """
    Keep the peak with the highest probability within each fused window.

    Parameters:
        windows (list): A list of fused windows.

    Returns:
        list: A list of fused windows with only the peak with the highest probability kept.
    """
    if len(windows) != 0:
        for window in windows:
            prob_list = window[2]

            if len(prob_list) > 1:
                for i in range(len(window[3])):
                    #If a pair has been flagged as a fuse
                    if window[3][i]:
                        #Only keep the most probable
                        if (prob_list[i] - prob_list[i + 1]) < 0:
                            window[4][i] = False
                        else:
                            window[4][i + 1] = False

    return windows


def fuse_peaks(cell, detected_peaks,associated_probXamp, factor=0.3):
    """
    Fuse nearby peaks if several are detected in the same calcuim event.

    Parameters:
        cell (dict): A dictionary containing information about the cell.
        detected_peaks (list): list of peak detected so far
        associated_probXamp (list): list of probXamp associated with each peak
        factor (float): A factor to determine amount of fluctuation needed to avoid fusing

    Returns:
        dict: A dictionary containing information about the fused peaks.
    """
    # Check if there are detected peaks
    nb_event = len(detected_peaks)
    if nb_event > 0:
        # Initialize an array to keep track of which peaks to keep
        to_be_kept = np.ones(nb_event, dtype=bool)

        # Extract windows around peaks
        windows_list = extract_peak_window(cell, detected_peaks,associated_probXamp)

        # If windows are found, fuse nearby peaks and keep the most probable peak in each window
        if len(windows_list) != 0:
            windows_list = fuse_windows(windows_list, cell, factor)

            # When we need to fuse -> only keep the peak with the greater probXamp
            most_prob_in_window = keep_most_prob_in_window(windows_list)

            # Update the array of peaks to be kept based on fusion results
            for window in most_prob_in_window:
                if len(window[0]) > 1:
                    for j in range(len(window[0])):
                        for i in range(len(window[3])):
                            if window[3][i]:
                                if not window[4][j]:
                                    index = np.where((detected_peaks == window[0][j]))[0]
                                    #If a peak has been flag as needing fusing -> its index is not retained
                                    to_be_kept[index] = False

        # Select peaks to be kept based on the updated array
        keep_index = np.where(to_be_kept == True)[0]

        # Return information about the fused peaks
        return detected_peaks[keep_index],associated_probXamp[keep_index]
    else:
        # If no peaks are detected, return empty lists
        return [],[]



def remove_peaks_at_intervals(detected_peaks, associated_probXamp_list,recording_intervals):
    """
    Remove peaks that fall at specified recording intervals.

    Parameters:
        detected_peaks (list): A list of detected peak indices.
        recording_intervals (list): A list of recording intervals.

    Returns:
        list: A list of detected peak indices after removing peaks at recording intervals.
    """
    if len(detected_peaks) != 0:
        indx = []
        for i in range(len(detected_peaks)):
            cur_indx = detected_peaks[i]
            Keep = True
            for time in recording_intervals:
                #if the peak falls within 6 frames of a transition between recordings
                if (cur_indx + 3 - time) * (cur_indx - 3 - time) <= 0:
                    Keep = False
            
            if Keep:
                indx.append(i)
        return detected_peaks[indx],associated_probXamp_list[indx]
                
    else:
        return [],[]


def create_peak_index(cell,prediction,recording_intervals, treshold=1, promin=0.05):
    """
    Create a dictionary containing information about detected peaks, filtered by a threshold.

    Parameters:
        Cell (dict): A dictionary containing the trace informations
        treshold (float): Threshold value for selecting peaks based on ProbXAmp.
        interval_max (int): Maximum interval allowed between prediction peaks.
        promin (float): Minimum prominence required for peak detection in the prediction trace.

    Returns:
        list: A list containing the indices selected peaks.
    """
    # Find peaks in the raw trace corresponding to peaks in the prediction trace
    cascade_peak_index, Ftrace_peak_index = find_peak_from_prediction(cell,prediction, promin)


  # If no peaks are found, return an empty dictionary
    if len(cascade_peak_index) == 0:
        return {"index":[],"probXamp":[]}
    
    # If peaks are found, compute amplitudes and probabilities, else initialize empty lists
    if len(cascade_peak_index) > 0:
        cascade_peak_amp = prediction[cascade_peak_index]
        amplitudes = amplitude_of_event(cell, Ftrace_peak_index)

        # Compute a value based on the product of the confidence of the detection and the amplited of the event in the raw trace
        probXamp = amplitudes * cascade_peak_amp
    else:

        probXamp = []

    # Select peaks based on the threshold
    probXamp = np.array(probXamp)

    # Select peaks based on the threshold
    tresholded_peak = np.where((probXamp > treshold))[0]

    detected_peak_so_far = Ftrace_peak_index[tresholded_peak]
    probXamp_peak_so_far = probXamp[tresholded_peak]
    
    # Only one peak is detected by calcium event 
    peak_list,associated_probXamp_list = fuse_peaks(cell,detected_peak_so_far,probXamp_peak_so_far)

    # Remove peaks that might have been detected at the transition between recordings
    peak_list_final,associated_probXamp_list_final = remove_peaks_at_intervals(peak_list,associated_probXamp_list,recording_intervals)
    
    # Return list containing selected peak information
    return {"index":peak_list_final,"probXamp":associated_probXamp_list_final}


"""
 ________  ________  ________  ___  ___  ________  _________        ________  _______  _________  _______   ________ _________  ___  ________  ________
|\   __  \|\   __  \|\   __  \|\  \|\  \|\   ____\|\___   ___\     |\   ___ \|\  ___ \|\___   ___\\  ___ \ |\   ____\\___   ___\\  \|\   __  \|\   ___  \
\ \  \|\  \ \  \|\  \ \  \|\ /\ \  \\\  \ \  \___|\|___ \  \_|     \ \  \_|\ \ \   __/\|___ \  \_\ \   __/|\ \  \___\|___ \  \_\ \  \ \  \|\  \ \  \\ \  \
 \ \   _  _\ \  \\\  \ \   __  \ \  \\\  \ \_____  \   \ \  \       \ \  \ \\ \ \  \_|/__  \ \  \ \ \  \_|/_\ \  \       \ \  \ \ \  \ \  \\\  \ \  \\ \  \
  \ \  \\  \\ \  \\\  \ \  \|\  \ \  \\\  \|____|\  \   \ \  \       \ \  \_\\ \ \  \_|\ \  \ \  \ \ \  \_|\ \ \  \____   \ \  \ \ \  \ \  \\\  \ \  \\ \  \
   \ \__\\ _\\ \_______\ \_______\ \_______\____\_\  \   \ \__\       \ \_______\ \_______\  \ \__\ \ \_______\ \_______\  \ \__\ \ \__\ \_______\ \__\\ \__\
    \|__|\|__|\|_______|\|_______|\|_______|\_________\   \|__|        \|_______|\|_______|   \|__|  \|_______|\|_______|   \|__|  \|__|\|_______|\|__| \|__|
                                           \|_________|


"""
#Smoothing will increase the likelihood of fusing two peaks that are close together. To ensure that those peaks are still considered, we compare what has been fused before and after fusing
#If a two peaks have been fused in the smoothing trace but not in the original, we retrieve the index in the original case and add it to the smoothing trace



def resolve_peak_fusion(cell):
    """
    Identify peaks that are fused in the smoothed trace but not in the original trace.

    Parameters:
        cell (dict): A dictionary containing information about the cell.

    Returns:
        list: A list of pairs of peaks, where each pair contains peaks from the smoothed trace and
              corresponding peaks from the original trace.
    """
    # Extract windows around peaks from the original and smoothed traces
    windows_before_smoothing = extract_peak_window(cell, cell["DetectedEvent"]["index"], cell["DetectedEvent"]["probXamp"])
    windows_after_smoothing = extract_peak_window(cell, cell["DetectedEvent_smooth"]["index"], cell["DetectedEvent_smooth"]["probXamp"])

    # List to store peaks that are fused in the smoothed trace but not in the original
    rescued = []

    # Check if windows are found in both original and smoothed traces
    if len(windows_before_smoothing) > 0 and len(windows_after_smoothing) > 0:
        
        # Iterate over windows in the original trace
        for window_before in windows_before_smoothing:
            if len(window_before[0]) > 1:
                
                # Iterate over windows in the smoothed trace
                for window_after in windows_after_smoothing:
                    
                    # Check if a window is detected both before and after smoothing
                    if [window_before[1][0], window_before[1][2]] == [window_after[1][0], window_after[1][2]]:
                        
                        # Check if the number of peaks is different in both cases
                        if len(window_after[0]) != len(window_before[0]):
                            
                            # Add peaks from the smoothed window to the rescued list
                            # along with corresponding peaks from the original window
                            rescued.append([window_after[0], window_before[0]])
                            break  # Break the loop as the window pair is found
                        break  # Break the loop as the corresponding window is found

    return rescued


def rescue_fused_events(cell):
    """
    Rescues fused events from the original trace based on peak information.

    Parameters:
        cell (dict): A dictionary containing information about the cell and its peaks.

    Returns:
        dict: Dictionary containing the rescued peak information.
    """

    # Resolve fused peaks between two sets of peak information
    rescued_list = resolve_peak_fusion(cell)
    
    # If no peaks need to be rescued from the original trace, return the smoothed trace detected peak list information
    if len(rescued_list) == 0:
        return cell["DetectedEvent_smooth"]
    
    # Create an empty DataFrame to store resolved peak information
    df = pd.DataFrame(columns=["index", "probXamp"])
    
    # Collect indices of peaks to be changed
    to_be_changed = [peak_indx for window_after in [window[0] for window in rescued_list] for peak_indx in window_after]
    
    # Filter out unchanged peaks from the smoothed trace and add them to the DataFrame
    for i in range(len(cell["DetectedEvent_smooth"]["index"])):
        if cell["DetectedEvent_smooth"]["index"][i] not in to_be_changed:
            df.loc[len(df)] = [cell["DetectedEvent_smooth"]["index"][i],
                               cell["DetectedEvent_smooth"]["probXamp"][i]]
    
    # Collect peak indices that need to be rescued from the original trace
    index_from_original = [peak_indx for window_before in [window[1] for window in rescued_list] for peak_indx in window_before]
    
    # Filter out changed peaks from the original trace and add them to the DataFrame
    for i in range(len(cell["DetectedEvent"]["index"])):
        if cell["DetectedEvent"]["index"][i] in index_from_original:
            df.loc[len(df)] = [cell["DetectedEvent"]["index"][i],
                               cell["DetectedEvent"]["probXamp"][i]]
    
    # Sort the DataFrame by the "index" column
    df.sort_values(by='index', inplace=True)
    
    # Convert DataFrame back to dictionary format
    rescued_peaks = {
        "index": np.array(df["index"], dtype=np.int64),
        "probXamp": np.array(df["probXamp"])
    }
    
    return rescued_peaks


def return_info_in_interval(cell_detected_event, interval_of_interest):
    """
    Extracts information from a cell's detected events within a given time interval.

    Parameters:
        cell_detected_event (dict): A dictionary containing information about detected events in a cell.
        interval_of_interest (list): A list containing two elements representing the start and end points
            of the interval of interest.

    Returns:
        dict: A dictionary containing filtered indices and their corresponding probability-amplitude values
            within the interval of interest. Keys are "index" and "probXamp".
    """
    # Extract start and end points of the interval of interest
    start_exp = interval_of_interest[0]
    end_exp = interval_of_interest[1]

    # Check if there are any detected events in the cell
    if len(cell_detected_event["index"]) != 0:
        # Filter detected events within the specified interval
        index_in_experimental_window = np.where((cell_detected_event["index"] > start_exp) & 
                                                (cell_detected_event["index"] < end_exp))[0]

        # Return filtered indices and their corresponding probability-amplitude values
        return {
            "index": cell_detected_event["index"][index_in_experimental_window],
            "probXamp": cell_detected_event["probXamp"][index_in_experimental_window]
        }
    else:
        # Return empty lists if no detected events are present
        return {
            "index": [],
            "probXamp": []
        }

    

def detect_robust_cells(cells, interval_of_interest=[1909,6000],proportion_needed=0.8):
    """
    Detects robust cells based on the proportion of detected events remaining after rescuing fused events.

    Parameters:
        cells (list): A list of dictionaries containing information about cells and their detected events.
        interval_of_interest (list): A list of dictionaries containing the frame corresponding to the beginning and the end of the experiment
        proportion_needed (float): The proportion of detected events needed to consider a cell as robust. Default is 0.8.

    Returns:
        list: A list of robust cells.
    """
    robust_cells = []

    # Iterate over each cell
    for cell in cells:
        

        # Rescue fused events
        cell["DetectedEvent_rescued"] = rescue_fused_events(cell)

        # We want to ensure that they are robustly detected in the timeframe of the experiment so we focus on the event detected during this period

        cell["Experimental_Interval"]= return_info_in_interval(cell["DetectedEvent"], interval_of_interest)
        cell["Experimental_Interval_rescued"]= return_info_in_interval(cell["DetectedEvent_rescued"], interval_of_interest)

        
        # Check if events are detected in the cell
        if len(cell["Experimental_Interval"]["index"]) != 0:
            nb_total_event_detected = len(cell["Experimental_Interval"]["index"])
            nb_total_event_detected_after_smoothing = len(cell["Experimental_Interval_rescued"]["index"])
            proportion_still_there = nb_total_event_detected_after_smoothing / nb_total_event_detected
            
            # Check if the proportion of events remaining meets the specified threshold
            if proportion_still_there >= proportion_needed:
                robust_cells.append(cell)
    
    return robust_cells