from lib_location_study import *

if __name__ == "__main__":
    ## Extract all Google location data
    df = extract_location_json("location-history.json")
    plot_geo_map(df)
    
    ## Data Analysis
    # Frequency per location
    loc_map = create_location_map(df)
    
    # Duration per location
    duration_map = create_duration_map(df)    
    
    # Most Frequented Locations
    sorted_locations = sorted(loc_map.items(), key=lambda x: x[1], reverse=True)

    print("Most Visited Locations:")
    for (lat, lon), count in sorted_locations:
        print(f"Latitude: {lat}, Longitude: {lon} → Visits: {count}")
    
    # Longest Duration Locations
    top_10_durations = sorted(duration_map.items(), key=lambda x: x[1], reverse=True)

    print("Longest Visited Locations:")
    for key, duration in top_10_durations:
        print(f"Geolocation {key}: Time (m) {duration:.2f}")
    
    ## Call Details API and store results (only one-time needed) 
    # https://developers.google.com/maps/documentation/places/web-service/details
    # write_google_details_of_type_visit(df)
    
    ## Create Clusters
    
    duration_array = create_duration_array(df)
    weights = duration_array[:, 2]

    # Custom weighted distance function
    def weighted_distance(a, b):
        spatial_dist = np.linalg.norm(a[:2] - b[:2])    # Geographic euclidean distance
        time_weight = 1 / (1 + abs(a[2] - b[2]))        # Inverse time spent
        return spatial_dist + (time_weight * 1/100)     # Distance + Time * arbitrary constant

    # Compute pairwise distances
    distance_matrix = squareform(pdist(duration_array, metric=weighted_distance))

    # Apply weighted DBSCAN
    # https://github.com/lucboruta/wdbscan
    labels = wdbscan(distance_matrix, epsilon=0.0005, mu=100, weights=weights)       # mu = 100, e = 0.002, 0.0005 - epsilon: neighborhood must atleast weigh this
    
    # Plot Cluster
    plot_clusters(duration_array, labels)
    
    ## Map labels to cluster record
    # Label records in df
    if len(labels) != len(df):
        df["cluster"] = -1                                  # noise
        df.loc[df.index[:len(labels)], "cluster"] = labels
    
    duration_tuples = [tuple(loc[:2]) for loc in duration_array]
    label_mapping = dict(zip(duration_tuples, labels))

    df["cluster"] = df.apply(lambda row: label_mapping.get((row["latitude"], row["longitude"]), -1), axis=1)
    
    
    # Find max duration loc per cluster
    max_duration_per_cluster = find_max_duration_per_cluster(df, duration_array)
    plot_geo_map(df, "cluster", max_duration_per_cluster)
    
    # Write non-visit-overlapping guesses to file (only one-time needed) 
    # write_non_overlapping_guesses_to_file(df, max_duration_per_cluster)   # One-time call
    
    ## Process API metadata
    # Parse visit places data
    api_data_df = parse_places("places_data/visit_places_info.txt")
    
    # Add original df info
    api_data_df = api_data_df.merge(df, on="placeID", how="left")
    api_data_df = api_data_df.drop_duplicates(subset=["placeID"], keep="first")
    
    # Visual df analysis
    api_data_df.to_csv("visit_data.csv", index=False)
    
    # Parse nearby places
    for i in range(1, 17):
        new_df = parse_places(f"places_data/nearby_places_info_{i}.txt", "nearby_places")
        api_data_df = pd.concat([api_data_df, new_df], ignore_index=True)
    
    ## Find Overlapping geolocations
    api_data_df["overlapping guess"] = 0

    # overlapping guess = 1 if guess loc = visit loc
    api_data_df.loc[api_data_df["sourceType"] == "visit", "overlapping guess"] = api_data_df.apply(
        lambda row: check_overlap(row, max_duration_per_cluster) if row["sourceType"] == "visit" else 0,
        axis=1
    )

    # Visual df analysis
    api_data_df.to_csv("visit_and_nearby_places_data.csv", index=False)

    # Plot visits, guesses, and overlapping locations
    plot_geo_map(api_data_df)
    
    ## Save API request Data
    # api_data_df, word_map = find_label(api_data_df)      # Minimal Use
    # save_pickle(api_data_df, "api_data_df.pkl")          # Minimal Use
    # save_pickle(word_map, "word_map.pkl")                # Minimal Use

    # Load relavant data
    api_data_df = load_pickle("api_data_df.pkl")
    word_map = load_pickle("word_map.pkl")
    
    # Visual df analysis
    api_data_df.to_csv("visit_and_nearby_places_data_with_label.csv", index=False)
    
    ## Word Analysis    
    # Print top words
    print_top_words(word_map, 10)
    
    # Plot Word Map
    plot_word_map(word_map)
    
    # Update word Map with types count
    word_map = update_word_map_from_types(word_map, api_data_df)
    
    # Update label with relavant Google Type field
    api_data_df["Label"] = api_data_df.apply(lambda row: extract_label(row["Types"], row.get("Label", None)), axis=1)
    
    # Visual df analysis
    api_data_df.to_csv("visit_and_nearby_places_data_with_final_label.csv", index=False)
    
    # Plot Geo Map with labels
    plot_geo_map(api_data_df)
    
    ## Plot word map without excluded words
    excluded_words = {"point of interest", "establishment"}

    for word in excluded_words:
        word_map[word] = 0
    
    plot_word_map(word_map)