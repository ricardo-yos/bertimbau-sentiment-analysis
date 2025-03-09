#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import random
import numpy as np
import pandas as pd
import googlemaps
from dotenv import load_dotenv

# Load environment variables
BASE_DIR = os.path.abspath("..")
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

# Output data path
OUTPUT_PATH = os.path.join(BASE_DIR, "data/raw/places_raw.csv")

# List of place types to search
PLACE_TYPES = [
    "pet_store",          # Pet store
    "veterinary_care"     # Veterinary care
]

def get_google_maps_client():
    """
    Initializes and returns the Google Maps API client using the API key.
    
    Returns:
        googlemaps.Client: An instance of the Google Maps API client.
        
    Raises:
        ValueError: If the API key is not found.
    """
    API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
    if not API_KEY:
        raise ValueError("API key not found. Check your .env file.")
    return googlemaps.Client(key=API_KEY)

def exponential_backoff(retry_count):
    """
    Returns the sleep time based on the exponential backoff algorithm.
    
    Args:
        retry_count (int): The number of retry attempts made.

    Returns:
        float: The calculated sleep time (in seconds).
    """
    return min(2 ** retry_count + random.uniform(0, 1), 30)

def get_city_bounds(city):
    """
    Fetches the bounding box (northeast and southwest coordinates) of a city 
    using the Google Maps API.
    
    Args:
        city (str): The name of the city to get the bounds for.

    Returns:
        tuple: A tuple containing two dictionaries (northeast, southwest), 
               representing the bounding coordinates of the city.
               Returns (None, None) if an error occurs.
    """
    gmaps = get_google_maps_client()
    try:
        geocode_result = gmaps.geocode(city)
        if geocode_result:
            viewport = geocode_result[0]["geometry"]["bounds"]
            return viewport["northeast"], viewport["southwest"]
    except Exception as e:
        print(f"Error fetching bounding box for {city}: {e}")
    return None, None

def calculate_grid_spacing(search_radius_km):
    """
    Calculates grid spacing based on the desired search radius.
    
    Args:
        search_radius_km (float): The radius (in kilometers) for the grid spacing.

    Returns:
        float: The calculated grid spacing (in kilometers).
    """
    return (2 * search_radius_km) / (1 + np.sqrt(3))

def calculate_num_points(northeast, southwest, grid_spacing_km):
    """
    Calculates the number of grid points along latitude and longitude based on the city's bounding box.
    
    Args:
        northeast (dict): Dictionary containing the "lat" and "lng" of the northeast corner of the bounding box.
        southwest (dict): Dictionary containing the "lat" and "lng" of the southwest corner of the bounding box.
        grid_spacing_km (float): The grid spacing (in kilometers).

    Returns:
        tuple: A tuple containing the number of grid points along latitude and longitude.
    """
    lat_diff = northeast["lat"] - southwest["lat"]
    num_points_lat = int(np.ceil(lat_diff * 111 / grid_spacing_km)) + 1

    avg_lat = (northeast["lat"] + southwest["lat"]) / 2.0
    lng_diff = northeast["lng"] - southwest["lng"]
    num_points_lng = int(np.ceil(lng_diff * 111 * np.cos(np.deg2rad(avg_lat)) / grid_spacing_km)) + 1

    return num_points_lat, num_points_lng

def generate_grid(northeast, southwest, num_points_lat, num_points_lng):
    """
    Generates a grid of latitude and longitude points within the city's bounding box.
    
    Args:
        northeast (dict): Dictionary containing the "lat" and "lng" of the northeast corner of the bounding box.
        southwest (dict): Dictionary containing the "lat" and "lng" of the southwest corner of the bounding box.
        num_points_lat (int): Number of points along the latitude axis.
        num_points_lng (int): Number of points along the longitude axis.

    Returns:
        list: A list of tuples containing latitude and longitude points.
    """
    lats = np.linspace(southwest["lat"], northeast["lat"], num_points_lat)
    lngs = np.linspace(southwest["lng"], northeast["lng"], num_points_lng)
    return [(lat, lng) for lat in lats for lng in lngs]

def fetch_places(lat, lng, place_type, radius, gmaps):
    """
    Searches for places of a specific type around a given latitude and longitude.
    
    Args:
        lat (float): Latitude of the center point for the search.
        lng (float): Longitude of the center point for the search.
        place_type (str): Type of place to search for (e.g., "pet_store").
        radius (float): Search radius in kilometers.
        gmaps (googlemaps.Client): The Google Maps API client instance.

    Returns:
        list: A list of places found within the specified radius and place type.
    """
    results = []
    next_page_token = None
    retry_count = 0

    while True:
        try:
            response = gmaps.places_nearby(
                location=(lat, lng),
                radius=int(radius * 1000),
                type=place_type,
                page_token=next_page_token
            )

            results.extend(response.get("results", []))
            next_page_token = response.get("next_page_token", None)

            if not next_page_token:
                break
            time.sleep(2)
        except Exception as e:
            print(f"Error occurred: {e}")
            retry_count += 1
            if retry_count > 3:
                print("Max retries reached. Breaking the loop.")
                break
            else:
                sleep_time = exponential_backoff(retry_count)
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
    return results

def fetch_all_places(city, search_radius_km=2):
    """
    Searches for all place types across a grid in a city.

    Args:
        city (str): The name of the city to search.
        search_radius_km (float): The radius for the search grid in kilometers (default is 2 km).

    Returns:
        list: A list of places found within the city grid.
    """
    gmaps = get_google_maps_client()
    northeast, southwest = get_city_bounds(city)
    if not northeast or not southwest:
        print(f"Could not retrieve city bounds for {city}.")
        return []

    # Calculate grid spacing based on search radius
    grid_spacing_km = calculate_grid_spacing(search_radius_km)
    print(f"Calculated grid spacing: {grid_spacing_km:.2f} km")

    # Calculate the number of grid points needed
    num_points_lat, num_points_lng = calculate_num_points(northeast, southwest, grid_spacing_km)
    print(f"Number of grid points: {num_points_lat} (lat) x {num_points_lng} (lng)")

    # Generate the grid
    grid_points = generate_grid(northeast, southwest, num_points_lat, num_points_lng)

    all_places = []
    for lat, lng in grid_points:
        for place_type in PLACE_TYPES:
            print(f"Searching for {place_type} at ({lat}, {lng}) with radius {search_radius_km:.2f} km...")
            places = fetch_places(lat, lng, place_type, search_radius_km, gmaps)
            if places:
                for place in places:
                    all_places.append({
                        "Name": place["name"],
                        "Address": place.get("vicinity", ""),
                        "Rating": place.get("rating", "N/A"),
                        "Number of Reviews": place.get("user_ratings_total", "N/A"),
                        "Place ID": place["place_id"],
                        "Type": place_type,
                        "Latitude": place["geometry"]["location"]["lat"],
                        "Longitude": place["geometry"]["location"]["lng"]
                    })
    return all_places

def remove_duplicates(places_data):
    """
    Removes duplicate places based on Place ID.
    
    Args:
        places_data (list): A list of places (dictionaries).

    Returns:
        list: A list of unique places based on Place ID.
    """
    unique_places = {place["Place ID"]: place for place in places_data}
    return list(unique_places.values())

def save_places_to_csv(places_data, output_file=OUTPUT_PATH):
    """
    Saves place data to a CSV file.
    
    Args:
        places_data (list): A list of places to save.
        output_file (str): The file path to save the data (default is "places_raw.csv").

    Returns:
        None
    """
    if not places_data:
        print("No data to save.")
        return

    unique_places = remove_duplicates(places_data)
    df_places = pd.DataFrame(unique_places)
    df_places.to_csv(output_file, sep=';', index=False, encoding="utf-8")
    print(f"Data saved to {output_file}")

# Main execution block
if __name__ == "__main__":
    city = "Santo André, SP, Brazil"
    search_radius_km = 2  # Define the search radius in km

    places_data = fetch_all_places(city, search_radius_km)
    save_places_to_csv(places_data)
