from io import StringIO
from datetime import datetime, timedelta, date, time
from zoneinfo import ZoneInfo
import json
from time import sleep
from dateutil.relativedelta import relativedelta

import requests
import pandas as pd
import numpy as np
from geopy import distance

import utils.constants as constants
from utils.meteo_france_client import Client
from dotenv import load_dotenv


class Weather:
    def __init__(self):
        load_dotenv()

    def _reverse_search_city(self, lat_lon: list) -> requests.Response:
        """search a city from its coordinates.

        Args:
            lat_lon (list): city coordinates.

        Returns:
            requests.Response: response from the API or an empty list if no result
            found.
        """
        lat = lat_lon[0]
        lon = lat_lon[1]
        x = len(str(lat))

        while x > 0:
            url = constants.REVERSE_ADRESS_URL
            payload = {
                'lon': lon,
                'lat': str(lat)[:x],
                'type': 'street',
                'limit': 1
            }
            r = requests.get(url, params=payload)

            if r.json().get('features'):
                    return r.json().get('features')[0]
            else:
                x -= 1

        return r.json().get('features')

    def _get_nearest_station_information(self, lat_lon: list) -> dict:
        """Get information for the nearest observation station calculated with
        geodesic distance.

        Args:
            lat_lon (list): coordinates to calculate distance from. 

        Returns:
            dict: nearest station information.
        """
        df = pd.read_csv(
            constants.WEATHER_STATION_LIST_PATH,
            sep=';',
            dtype={'Id_station': object},
            parse_dates=['Date_ouverture']
        )

        df.columns = df.columns.str.lower()
        df['nom_usuel'] = df['nom_usuel'].str.title()

        if ('latitude' and 'longitude' in df.columns) and lat_lon:
            df['distance'] = df.apply(
                lambda x: distance.distance(
                    [x['latitude'], x['longitude']],
                    lat_lon).km,
                    axis='columns'
            )

            return df.nsmallest(1, 'distance').to_dict('records')[0]

    def filter_nearest_station_information(self, lat_lon: list) -> dict:
        """Get and filter information for the nearest station to only provide
        what is useful for the STreamlit app.

        Args:
            lat_lon (list): city coordinates.

        Returns:
            dict: nearest station information.
        """
        station_info = self._get_nearest_station_information(lat_lon)

        reverse_info = self._reverse_search_city([station_info['latitude'],
                                             station_info['longitude']])

        if not reverse_info:
            reverse_info = {'properties': {'city': '-', 'context': '-'}}

        return {
            'id_station': station_info.get('id_station'),
            'nom_usuel': station_info.get('nom_usuel'),
            'date_ouverture': station_info.get('date_ouverture'),
            'altitude': station_info.get('altitude'),
            'distance': station_info.get('distance'),
            'city': reverse_info.get('properties').get('city'),
            'context': reverse_info.get('properties').get('context')
        }

    def get_station_info(self, coordinates: list[float]) -> dict:
        """Call function with cache decorator to retrieve nearest observation
        station information.

        Args:
            coordinates (list[float]): nearest station latitude and longitude.

        Returns:
            dict: station information
        """

        return self.filter_nearest_station_information(coordinates)

    def get_climatological_data(self, id_station: str) -> pd.DataFrame:
        """Call function with cache decorator to get climatological data for a full
        year.

        Args:
            id_station (str): id of nearest observation station ;
            year (int): year of requested data.

        Returns:
            pd.DataFrame: climatological data.
        """

        nearest_station_info = self.get_station_info(['43.0145', '3.0525'])

        end_date = datetime.now()-relativedelta(days=2)

        start_date = end_date - relativedelta(months=2)

        if start_date < nearest_station_info.get('date_ouverture'):
            print('start date to early')
            start_date = f'''{nearest_station_info.get('date_ouverture')}T00:00:00Z'''
        else:
            start_date = start_date.strftime('%Y-%m-%dT00:00:00Z')

        # Define the end datetime (if selected year is the current year we probably
        # can't end the period at end of December)
        end_date = end_date.strftime('%Y-%m-%dT00:00:00Z')

        print('begin = ', start_date, ', end = ', end_date)

        # Get data from the api
        visualization_order_response = Client().order_daily_climatological_data(
            id_station, start_date, end_date)

        print(visualization_order_response.text)

        if visualization_order_response.status_code == 202:
            try:
                visualization_order_id = (
                    visualization_order_response
                    .json() 
                    .get('elaboreProduitAvecDemandeResponse')
                    .get('return')
                )
            except json.JSONDecodeError:
                raise Exception('Erreur de décodage de la réponse JSON.')
        else:
            raise Exception(
                f'''Echec de la récupération des données.  
                {visualization_order_response.status_code} : {visualization_order_response.reason}
                ''')

        # Get data from order id and retry if data not yet ready (code 204)
        response_code = 204
        n_tries = 0
        while response_code == 204 and n_tries < 5:
            visualization_climatological_response = Client().order_recovery(
                    visualization_order_id)
            response_code = visualization_climatological_response.status_code
            n_tries =+ n_tries
            sleep(10)

        if visualization_climatological_response.status_code == 201:
            try:
                visualization_climatological_data = visualization_climatological_response.text

                # Import data in DataFrame
                df = pd.read_csv(StringIO(visualization_climatological_data), sep=';',
                                parse_dates=['DATE'])
                # Convert 'object' to 'float'
                string_col = df.select_dtypes(include=['object']).columns
                for col in string_col:
                    df[col] = df[col].str.replace(',', '.')
                    df[col] = df[col].astype('float')
                # Remove all variables with only NaN
                df =  df.dropna(axis='columns')
            except Exception as e:
                f'Echec lors de la lecture de la réponse : {e}.'
        else:
            raise Exception(
                f'''Echec de la récupération des données.  
                {visualization_climatological_data.status_code} : {visualization_climatological_data.reason}
                ''')

        df = df[["DATE", "RR", "TM", "DG", "ETPGRILLE"]]
        df["DATE"] = df["DATE"].dt.strftime('%d/%m/%Y')
        # Read the CSV file that contains the mapping
        mapping_df = pd.read_csv(constants.WEATHER_PARAMETERS_LIST_PATH, sep=';')

# Create a dictionary that maps parameter to label
# If label is empty or missing, you can decide to either not include it in the mapping or keep the original name
        mapping = {row['parameter']: row['label'] for _, row in mapping_df.iterrows() if pd.notnull(row['label'])}

        df.rename(columns=mapping, inplace=True)

        return df
    
    def process_weather(self, station_id):
        return self.get_climatological_data(station_id)