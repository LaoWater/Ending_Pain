#
# Calculating compressions functions
#
def calculate_compressions(stations_metrics_instance, tracks_metrics_instance):
    # Define the relationships between tracks and their corresponding stations.
    track_to_station_mapping = {
        "Train Track 1 (Nose to Shoulder)": ["Shoulders"],
        "Train Track 2 (Shoulder to Hip)": ["Shoulders", "Hips"],
        "Train Track 3 (Hip to Knees)": ["Knees", "Hips"],
        "Train Track 4 (Knees to Foot)": ["Knees", "Feet"],
        "Train Track 5 (Shoulder to Elbow)": ["Shoulders"],
    }

    def get_offsets_and_sort(metrics_dict):
        # Prepare a list of (name, offset) tuples
        name_offset_pairs = [
            (key, float(value['Offset'].replace('%', '')) if isinstance(value['Offset'], str)
            else float(value['Offset'])) for key, value in metrics_dict.items()
        ]
        # Sort based on the absolute value of offsets, but keep the tuples intact
        sorted_pairs = sorted(name_offset_pairs, key=lambda pair: abs(pair[1]), reverse=True)
        # Format the sorted pairs for printing and retain numeric data for calculations
        sorted_info_for_print = [f"{pair[0]}: {pair[1]}" for pair in sorted_pairs]
        return sorted_pairs, sorted_info_for_print

    # Assuming tracks_metrics_instance.tracks and stations_metrics_instance.metrics are defined elsewhere
    tracks_sorted_pairs, tracks_sorted_info_for_print = get_offsets_and_sort(tracks_metrics_instance.tracks)
    stations_sorted_pairs, stations_sorted_info_for_print = get_offsets_and_sort(stations_metrics_instance.metrics)

    # 70% Tracks weigh-in, 30% Stations
    def calculate_weighted_values(tracks_pairs, stations_pairs, mapping):
        weighted_results = []
        for w_track, track_value in tracks_pairs:
            if w_track in mapping:
                station_names = mapping[w_track]
                station_values = [value for name, value in stations_pairs if name in station_names]

                # Calculate the average station value if multiple stations are related to a single track
                avg_station_value = sum(station_values) / len(station_values) if station_values else 0

                # Calculate weighted value and round it
                weighted_value = round(0.7 * track_value + 0.3 * avg_station_value, 2)
                weighted_results.append((w_track, weighted_value, abs(weighted_value)))

        # Sort by the absolute values stored as the third element in each tuple
        weighted_results.sort(key=lambda x: x[2], reverse=True)

        # Remove the absolute values from the results, keeping the original weighted values
        final_results = [(weighted_track, value) for weighted_track, value, _ in weighted_results]

        return final_results

    # Calculate and sort the AI calculated values
    ai_sorted_results = calculate_weighted_values(tracks_sorted_pairs, stations_sorted_pairs, track_to_station_mapping)

    # Explicitly handling numerical operations for compression sums using the pairs
    right_tracks_compression = sum([pair[1] for pair in tracks_sorted_pairs if pair[1] > 0])
    left_tracks_compression = sum([abs(pair[1]) for pair in tracks_sorted_pairs if pair[1] < 0])
    right_stations_compression = sum([pair[1] for pair in stations_sorted_pairs if pair[1] > 0])
    left_stations_compression = sum([abs(pair[1]) for pair in stations_sorted_pairs if pair[1] < 0])

    compressions = {
        'tracks_offsets_sorted_for_print': tracks_sorted_info_for_print,
        'stations_offsets_sorted_for_print': stations_sorted_info_for_print,
        'compressions': {'right_tracks_compression': round(right_tracks_compression, 2),
                         'left_tracks_compression': round(left_tracks_compression, 2),
                         'right_stations_compression': round(right_stations_compression, 2),
                         'left_stations_compression': round(left_stations_compression, 2),
                         },
        'Left': round((0.7 * left_tracks_compression + 0.3 * left_stations_compression), 2),
        'Right': round((0.7 * right_tracks_compression + 0.3 * right_stations_compression), 2),
        'AI_Sorted_Results': ai_sorted_results
    }

    return compressions


# To print the results in a readable format:
def print_calculated_compressions(compressions):
    print("\n---------------------------------------------------------\n"
          "*Dictionary\nSorted Tracks and Stations will make up ai_sorted_results with 70%-30% weigh\n"
          "\nTracks Offsets (sorted):")
    for track_info in compressions['tracks_offsets_sorted_for_print']:
        print(track_info)
    print("\nStations Offsets (sorted):")
    for station_info in compressions['stations_offsets_sorted_for_print']:
        print(station_info)
    print("\nCompressions Summary:")
    print("Right Tracks Compression:", compressions['compressions']['right_tracks_compression'])
    print("Left Tracks Compression:", compressions['compressions']['left_tracks_compression'])
    print("Right Stations Compression:", compressions['compressions']['right_stations_compression'])
    print("Left Stations Compression:", compressions['compressions']['left_stations_compression'])

    # Printing the calculated Left and Right values
    print("\n---------------------------------------------------------\n\nWeighted Compressions (tracks and "
          "stations combined):\naka Ending_Pain Lateral Lines Compression Scale\n")
    print("Left Line Compression:", compressions['Left'])
    print("Right Line Compression:", compressions['Right'])
    print("\n---------------------------------------------------------\nAI Sorted Results Analysis:\n"
          "*Treatment Plan, Tracks & Stations Priority*\n", compressions['AI_Sorted_Results'])