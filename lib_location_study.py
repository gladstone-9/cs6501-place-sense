import json
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx
from collections import defaultdict
import Constants
import requests
from pprint import pprint
import numpy as np
import pandas as pd
import json
from datetime import datetime
from scipy.spatial.distance import pdist, squareform
import re
import unicodedata
from adjustText import adjust_text
import pickle

# Store objects
def save_pickle(word_map, filename="word_map.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(word_map, f)

# Load objects
def load_pickle(filename="word_map.pkl"):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}

'''
Extract location-history.json data
- visit records
- activity records
- timelinePath records
'''
def extract_location_json(filename):    
    with open(filename, "r") as file:
        data = json.load(file)

    locations = []
    placeIDs = []
    sourceTypes = []
    durations = []

    for entry in data:
        try:
            # Parse startTime and endTime
            start_time = entry.get("startTime", None)
            end_time = entry.get("endTime", None)
            
            # Convert to datetime and compute duration (in minutes)
            duration = None
            if start_time and end_time:
                start_dt = datetime.fromisoformat(start_time.rstrip("Z"))  # Remove Z for ISO format
                end_dt = datetime.fromisoformat(end_time.rstrip("Z"))
                duration = (end_dt - start_dt).total_seconds() / 60  # Convert to minutes

            # visit record
            if "visit" in entry:
                visit = entry["visit"].get("topCandidate", {})
                place_location = visit.get("placeLocation", "N/A")
                placeID = visit.get("placeID", "N/A")

                locations.append(place_location)
                placeIDs.append(placeID)
                sourceTypes.append("visit")
                durations.append(duration)

            # activity record
            if "activity" in entry:
                activity = entry["activity"]

                for key in ["start", "end"]:
                    if key in activity:
                        locations.append(activity[key])
                        placeIDs.append("N/A")
                        sourceTypes.append("activity")
                        durations.append(duration)

            # timelinePath record
            if "timelinePath" in entry:
                previous_offset = None                  # offset tracking
                for point in entry["timelinePath"]:
                    locations.append(point["point"])
                    placeIDs.append("N/A")
                    sourceTypes.append("timelinePath")
                    
                    # Extract offset
                    current_offset = int(point.get("durationMinutesOffsetFromStartTime", 0))
                    
                    # Compute duration (current offset - last offset)
                    if previous_offset is None:
                        duration = current_offset
                    else:
                        duration = current_offset - previous_offset
                    
                    durations.append(duration)
                    previous_offset = current_offset

        except Exception as e:
            print(f"Error processing entry: {e}")

    # Create DataFrame
    df = pd.DataFrame({
        "placeLocation": locations,
        "placeID": placeIDs,
        "sourceType": sourceTypes,
        "duration": durations
    })

    # Remove N/A locations
    df = df[df["placeLocation"] != "N/A"]

    # Extract Latitude, Longitude
    df[["latitude", "longitude"]] = df["placeLocation"].str.replace("geo:", "", regex=False).str.split(",", expand=True)
    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)

    return df

# Redraw basemap to current axis limits (used in geo-plotting)
def update_basemap(ax):
    ax.set_autoscale_on(False)
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
    plt.draw()

# Zoom events (used in geo-plotting)
def on_zoom(ax, event):
    if event.button == 'up' or event.button == 'down':  # Mouse scroll zoom
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())
        update_basemap(ax)

# Plot geomap data
def plot_geo_map(df, color_overlay="type", location_guess=None):    
    # GeoDataFrame
    gdf = gpd.GeoDataFrame(df, 
                           geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), 
                           crs="EPSG:4326")
    
    # Web mercator projection
    gdf = gdf.to_crs(epsg=3857)
    
    fig, ax = plt.subplots(figsize=(35, 15))
    
    if color_overlay == "type":
        # Color mapping for sourceType
        color_map = {
            "visit": "red",
            "activity": "blue",
            "timelinePath": "green"
        }

        # Assign missing source types
        gdf["color"] = gdf["sourceType"].map(color_map).fillna("purple")

        if "overlapping guess" in gdf.columns:
            gdf.loc[gdf["overlapping guess"] == 1, "color"] = "deepskyblue"

        label_exists = "Label" in gdf.columns

        # Plot each sourceType
        for source_type, color in color_map.items():
            subset = gdf[gdf["sourceType"] == source_type]
            if not subset.empty:
                subset.plot(ax=ax, marker='o', color=color, markersize=50, alpha=0.6, label=source_type)

                # Add labels
                if label_exists:
                    for x, y, label in zip(subset.geometry.x, subset.geometry.y, subset["Label"]):
                        ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9, color="yellow")

        # Plot overlapping guess
        if "overlapping guess" in gdf.columns:
            subset_overlap = gdf[gdf["overlapping guess"] == 1]
            if not subset_overlap.empty:
                subset_overlap.plot(ax=ax, marker='2', color="deepskyblue", markersize=60, alpha=0.8, edgecolor="black", label="Overlapping Guess")

                # # Add labels if the "Label" column exists
                # if label_exists:
                #     for x, y, label in zip(subset_overlap.geometry.x, subset_overlap.geometry.y, subset_overlap["Label"]):
                #         ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9, color="black")

        # Plot missing sourceType values
        subset_missing = gdf[gdf["sourceType"].isna()]
        if not subset_missing.empty:
            subset_missing.plot(ax=ax, marker='^', color="purple", markersize=50, alpha=0.6, edgecolor="black", label="Nearby Sig. Locations")

            # # Add labels if the "Label" column exists
            # if label_exists:
            #     for x, y, label in zip(subset_missing.geometry.x, subset_missing.geometry.y, subset_missing["Label"]):
            #         ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9, color="black")
    
    elif color_overlay == "cluster":
        gdf.plot(ax=ax, column="cluster", cmap="tab10", markersize=50, alpha=0.6, legend=False)

        # Noise points handling
        noise_points = gdf[gdf["cluster"] == -1]
        if not noise_points.empty:
            noise_points.plot(ax=ax, color="black", marker="x", markersize=50, alpha=0.2, label="Noise")
        
        # Plot max duration per cluster
        max_duration_df = pd.DataFrame(
            location_guess.items(),
            columns=["cluster", "location"]
        )
        max_duration_df[["latitude", "longitude"]] = pd.DataFrame(
            max_duration_df["location"].tolist(), index=max_duration_df.index
        )

        max_gdf = gpd.GeoDataFrame(max_duration_df, 
                                    geometry=gpd.points_from_xy(max_duration_df["longitude"], max_duration_df["latitude"]), 
                                    crs="EPSG:4326").to_crs(epsg=3857)

        max_gdf.plot(ax=ax, marker="P", color="gold", markersize=45, edgecolor="black", alpha=0.8, label="≈Sig. Location")
        
        # Plot visit records
        visit_points = gdf[gdf["sourceType"] == "visit"]
        if not visit_points.empty:
            visit_points.plot(ax=ax, marker='*', color="red", edgecolor="black", markersize=60, alpha=0.95, label="Visit")

        ax.legend()

    # Set Plotting Options
    ax.set_title("Geolocation Plot by Source Type")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xticks(ax.get_xticks())  
    ax.set_yticks(ax.get_yticks())    
    ax.legend()

    # Add basemap
    # ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)   # Very detailed
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)      # High-level
    
    fig.canvas.mpl_connect("scroll_event", lambda event: on_zoom(ax, event))

    plt.show()

# Plot clusters from DBSCAN labels
def plot_clusters(locations, labels):
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    # Plot clusters
    plt.figure(figsize=(10, 6))
    for label, color in zip(unique_labels, colors):
        cluster_points = locations[labels == label]
        
        if label == -1:  # Noise
            plt.scatter(cluster_points[:, 1], cluster_points[:, 0], c="black", alpha=0.2 ,marker="x", label="Noise")
        else:
            plt.scatter(cluster_points[:, 1], cluster_points[:, 0], c=[color], marker="o", edgecolors="k", s=100)

    # Plotting Options
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Weighted DBSCAN Clustering")
    plt.legend()
    plt.grid(True)

    plt.show()
    
def get_google_details(PLACE_ID, API_KEY, FIELDS):
    # API Endpoint
    # https://developers.google.com/maps/documentation/places/web-service/details
    url = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={PLACE_ID}&key={API_KEY}&fields={FIELDS}"

    # Make the request
    response = requests.get(url)
    data = response.json()
    return data

def write_to_file(output_file, content):
    """Appends content to a file."""
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(content + "\n")

# Process place_ids of all visit records and write to file
def write_google_details_of_type_visit(df):
    API_KEY = Constants.API_KEY_GOOGLE
    FIELDS = "name,place_id,formatted_address,business_status,types,opening_hours,editorial_summary"
    unique_PLACE_IDS = df[['placeID']].drop_duplicates()
    for _, row in unique_PLACE_IDS.iterrows():
        PLACE_ID = row['placeID']
        data = get_google_details(PLACE_ID, API_KEY, FIELDS)
        print(data)

        if data.get("status") == "OK":
            result = data.get("result", {})

            name = result.get("name", "N/A")
            place_id = result.get("place_id", "N/A")
            address = result.get("formatted_address", "N/A")
            types = result.get("types", [])
            business_status = result.get("business_status", "N/A")
            editorial_summary = result.get("editorial_summary", {}).get("overview", "N/A")

            # Extract opening hours
            opening_hours = result.get("opening_hours", {})
            open_now = opening_hours.get("open_now", "N/A")
            weekday_text = opening_hours.get("weekday_text", [])

            # Collect output
            output = (
                f"Name: {name}\n"
                f"place_id: {place_id}\n"
                f"Address: {address}\n"
                f"Types: {', '.join(types)}\n"
                f"Business Status: {business_status}\n"
                f"Open Now: {open_now}\n"
            )

            if weekday_text:
                output += "Opening Hours:\n" + "\n".join(f"  {day}" for day in weekday_text) + "\n"
            else:
                output += "Opening Hours: N/A\n"

            output += f"Editorial Summary: {str(editorial_summary)}\n"
            output += "=" * 50 + "\n"

            # Write to file
            write_to_file("places_info_2.txt", output)
            print(output)
        else:
            print("Error:", data.get("status"), data.get("error_message", "No error message"))

def write_non_overlapping_guesses_to_file(df, max_duration_per_cluster):
    non_visit_significant_locations = find_max_duration_without_placeID(df, max_duration_per_cluster)

    for idx, (lat, lon) in enumerate(non_visit_significant_locations):
        location_str = f"{lat},{lon}"
        result_data = nearby_search(location_str)
    
    if result_data.get("status") == "OK":
        for result in result_data.get("results", []):
            name = result.get("name", "N/A")
            place_id = result.get("place_id", "N/A")
            address = result.get("vicinity", "N/A")
            types = result.get("types", [])
            business_status = result.get("business_status", "N/A")
            editorial_summary = result.get("editorial_summary", {}).get("overview", "N/A")
            actual_location = result.get("geometry", {}).get("location", {})
            actual_lat = actual_location.get("lat", "N/A")
            actual_lng = actual_location.get("lng", "N/A")
            
            opening_hours = result.get("opening_hours", {})
            open_now = opening_hours.get("open_now", "N/A")
            weekday_text = opening_hours.get("weekday_text", [])
            
            output = (
                f"Name: {name}\n"
                f"Guessed GeoLocation: {location_str}\n"
                f"Actual GeoLocation: {actual_lat},{actual_lng}\n"
                f"Place ID: {place_id}\n"
                f"Address: {address}\n"
                f"Types: {', '.join(types)}\n"
                f"Business Status: {business_status}\n"
                f"Open Now: {open_now}\n"
            )
            
            if weekday_text:
                output += "Opening Hours:\n" + "\n".join(f"  {day}" for day in weekday_text) + "\n"
            else:
                output += "Opening Hours: N/A\n"
            
            output += f"Editorial Summary: {str(editorial_summary)}\n"
            output += "=" * 50 + "\n"
            
            filename = f"nearby_places_info_{idx + 1}.txt"
            print(output)
            write_to_file(filename, output)
    else:
        print("Error:", result_data.get("status"), result_data.get("error_message", "No error message"))

# Dict: (latitude, longitude) --> Visit Count
def create_location_map(df):
    loc_map = defaultdict(int)
    for lat, lon in zip(df["latitude"], df["longitude"]):
        loc_map[(lat, lon)] += 1  # Count occurrences of each location
    
    return loc_map

# Dict: (latitude, longitude) --> Duration (m)
def create_duration_map(df):
    duration_map = defaultdict(int)
    
    for lat, lon, duration in zip(df["latitude"], df["longitude"], df["duration"]):
        duration_map[(lat, lon)] += duration
    
    return duration_map

# Array: [latitude, longitude, duration]
def create_duration_array(df):
    duration_map = defaultdict(int)
    
    # Durations for each (latitude, longitude)
    for lat, lon, duration in zip(df["latitude"], df["longitude"], df["duration"]):
        duration_map[(lat, lon)] += duration  
    
    processed_array = np.array([[lat, lon, duration] for (lat, lon), duration in duration_map.items()])
    
    return processed_array
    

# Weighted DBSCAN implementation
# Adapted from https://github.com/lucboruta/wdbscan
def wdbscan(dmatrix, epsilon, mu, weights=None, noise=True):
    """
    Generates a density-based clustering of arbitrary shape. Returns a numpy
    array coding cluster membership with noise observations coded as 0.
    
    Positional arguments:
    dmatrix -- square dissimilarity mat rix (cf. numpy.matrix, numpy.squareform,
            and numpy.pdist).
    epsilon -- maximum reachability distance.
    mu      -- minimum reachability weight (cf. minimum number of points in the
            classical DBSCAN).
    
    Keyword arguments:
    weights -- weight array (if None, weights default to 1).
    noise   -- Boolean indicating whether objects that do not belong to any
            cluster should be considered as noise (if True) or assigned to
            clusters of their own (if False).
    """
    n = len(dmatrix)
    ematrix = dmatrix <= epsilon  # Epsilon-reachability matrix
    epsilon_neighborhood = lambda i: [j for j in range(n) if ematrix[i, j]]
    
    if weights is None:
        weights = np.ones(n, dtype=int)  # Default to classic DBSCAN

    status = np.zeros(n, dtype=int)  # Unclassified = 0
    cluster_id = 1  # First cluster index

    for i in range(n):
        if status[i] == 0:  # Unclassified point
            seeds = epsilon_neighborhood(i)
            if weights[seeds].sum() < mu:
                status[i] = -1  # Mark as noise
            else:
                status[seeds] = cluster_id
                seeds.remove(i)
                while seeds:
                    j = seeds[0]
                    eneighborhood = epsilon_neighborhood(j)
                    if weights[eneighborhood].sum() >= mu:
                        for k in eneighborhood:
                            if status[k] <= 0:  # Unclassified or noise
                                if status[k] == 0:
                                    seeds.append(k)
                                status[k] = cluster_id
                    seeds.remove(j)
                cluster_id += 1

    if not noise:  # Assign cluster IDs to noise
        noisy = (status == -1)
        status[noisy] = range(cluster_id, cluster_id + noisy.sum())

    return status

# Geolocation with max duration per cluster
# Dict: Cluster ID --> (latitude, longitude) 
def find_max_duration_per_cluster(df, duration_array):
    cluster_max_duration = {}

    duration_dict = {(lat, lon): duration for lat, lon, duration in duration_array}

    # Group records by cluster
    grouped = df.groupby("cluster")

    for cluster, records in grouped:
        max_duration = -1
        best_location = None

        for _, row in records.iterrows():
            lat, lon = row["latitude"], row["longitude"]
            duration = duration_dict.get((lat, lon), 0)
            
            if duration > max_duration:
                max_duration = duration
                best_location = (lat, lon)

        if best_location:
            cluster_max_duration[cluster] = best_location

    return cluster_max_duration

# Filter out overlapping max duration locations
# List of (lat, lon)
def find_max_duration_without_placeID(df, max_duration_per_cluster):
    filtered_locations = []

    for cluster, (lat, lon) in max_duration_per_cluster.items():
        matching_records = df[(df["latitude"] == lat) & (df["longitude"] == lon)]
        
        # ALL records for given location must have no placeID
        if (matching_records["placeID"] == "N/A").all():
            filtered_locations.append((lat, lon))

    return filtered_locations

# radius in m
# https://developers.google.com/maps/documentation/places/web-service/search-nearby#maps_http_places_nearbysearch-txt
def nearby_search(location, radius=65):
    API_KEY = Constants.API_KEY_GOOGLE
    base_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": location,       # lat, lng
        "radius": radius,
        "key": API_KEY
    }
    
    response = requests.get(base_url, params=params)
    return response.json()

# Reformat opening hours field 
def clean_opening_hours(hours):
    if hours == "N/A":
        return hours
    cleaned_hours = unicodedata.normalize("NFKC", hours)
    cleaned_hours = re.sub(r"\s+", " ", cleaned_hours).strip()
    return cleaned_hours

# Extract df of API metadata from .txt file
def parse_places(file_path, file_type="visits"):
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.read()
    
    entries = data.split("==================================================")
    parsed_data = []
    
    for i, entry in enumerate(entries):
        # Skip first entry if nearby places
        if file_type == "nearby_places" and (i == 0 or not entry.strip()):
            continue
        
        if entry.strip():
            place = {}
            
            name_match = re.search(r"Name: (.+)", entry)
            if name_match:
                place["Name"] = name_match.group(1)
            
            guessed_geo_match = re.search(r"Guessed GeoLocation: (.+)", entry)
            if guessed_geo_match:
                place["Guessed GeoLocation"] = guessed_geo_match.group(1)
            
            actual_geo_match = re.search(r"Actual GeoLocation: (.+)", entry)
            if actual_geo_match:
                actual_geo = actual_geo_match.group(1).split(",")
                place["latitude"] = float(actual_geo[0])
                place["longitude"] = float(actual_geo[1])
            
            if(file_type == "visits"):
                place_id_match = re.search(r"place_id: (.+)", entry)
            elif(file_type == "nearby_places"):
                place_id_match = re.search(r"Place ID: (.+)", entry)
            if place_id_match:
                place["placeID"] = place_id_match.group(1)
            
            address_match = re.search(r"Address: (.+)", entry)
            if address_match:
                place["Address"] = address_match.group(1)
            
            types_match = re.search(r"Types: (.+)", entry)
            if types_match:
                place["Types"] = types_match.group(1)
            
            business_status_match = re.search(r"Business Status: (.+)", entry)
            if business_status_match:
                place["Business Status"] = business_status_match.group(1)
            
            open_now_match = re.search(r"Open Now: (.+)", entry)
            if open_now_match:
                place["Open Now"] = open_now_match.group(1)
            
            opening_hours_match = re.search(r"Opening Hours:(.+?)(?=Editorial Summary:|$)", entry, re.DOTALL)
            if opening_hours_match:
                place["Opening Hours"] = clean_opening_hours(opening_hours_match.group(1).strip())
            else:
                place["Opening Hours"] = "N/A"
            
            editorial_summary_match = re.search(r"Editorial Summary: (.+)", entry)
            if editorial_summary_match:
                place["Editorial Summary"] = editorial_summary_match.group(1)
            
            parsed_data.append(place)
    
    return pd.DataFrame(parsed_data)

# Generate datamuse string
def format_record(row):
    parts = []
    
    for col in ["Name", "Types", "Editorial Summary"]:
        if pd.notna(row[col]) and row[col] != "N/A":
            parts.append(row[col])
    
    return ": ".join(parts) if parts else None

# Label record with first word from datamuse request
def find_label(df):    
    # Generate query string for each record
    df["Formatted String"] = df.apply(format_record, axis=1)
    df = df.dropna(subset=["Formatted String"])
    
    word_map = {}
    labels = []

    # Call one_look_thesaurus_request
    for input_string in df["Formatted String"]:
        label = one_look_thesaurus_request(df, word_map, input_string)
        labels.append(label)

    # Add labels to DataFrame
    df["Label"] = labels  

    return df, word_map

# Update word map and return first word from Datamuse API
def one_look_thesaurus_request(df, word_map, input_string):
    
    url = "https://api.datamuse.com/words"
    params = {"ml": input_string}               # Find words with a similar meaning to string

    response = requests.get(url, params=params)

    if response.status_code == 200:
        words = response.json()
        
        if words:
            first_word = words[0]["word"]   # Get the first word
        else:
            first_word = "Unknown"          # Default label

        for word in words:
            word_text = word["word"]
            word_map[word_text] = word_map.get(word_text, 0) + 1 

        return first_word

    else:
        print("Error:", response.status_code)
        return "Error"

# Generate word cloud
def plot_word_map(word_map):
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="viridis"
    ).generate_from_frequencies(word_map)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Relevant Words", fontsize=14)
    plt.show()
    
# Check latitude and longitude match entry in max_duration_per_cluster
def check_overlap(row, cluster_dict):
    return 1 if (row["latitude"], row["longitude"]) in cluster_dict.values() else 0

# Print top N words with highest count in word map
def print_top_words(word_map, top_n=20):
    sorted_words = sorted(word_map.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Top {top_n} words:")
    for word, count in sorted_words[:top_n]:
        print(f"{word}: {count}")


# Update word map with types.
def update_word_map_from_types(word_map, df):
    """Updates word_map with counts from the 'Types' column in df."""
    
    for types_str in df["Types"].dropna():
        types_list = types_str.split(", ")
        
        for word in types_list:
            formatted_word = word.replace("_", " ")
            word_map[formatted_word] = word_map.get(formatted_word, 0) + 15

    return word_map

# Extract first type from Type field if not excluded word
def extract_label(types_str, current_label):
    # Excluded words
    excluded_words = {"point of interest", "establishment"}
    
    if pd.notna(types_str):
        types = [t.replace("_", " ") for t in types_str.split(", ")]
        for t in types:
            if t not in excluded_words:
                return t  
    return current_label