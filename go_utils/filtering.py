import numpy as np

__doc__ = """
# Overview
This submodule contains code to facilitate the general filtering of data. 
The following sections discuss some of the logic and context behind these methods.

# Methods

## Filter Invalid Coords
Certain entries in the GLOBE Database have latitudes and longitudes that don't exist.

## Filter Duplicates
Due to reasons like GLOBE Observer trainings among other things, there are oftentimes multiple observations of the same exact entry. This can lead to a decrease in data quality and so this utility can be used to reduce this. Groups of entries that share the same MGRS Latitude, NGRS Longitude, measured date, and other dataset specific attributes (e.g. water source) could likely be duplicate entries. In (Low, et. al)[https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021GH000436], Mosquito Habitat Mapper duplicates are removed by groups of size greater than 10 sharing MGRS Latitude, MGRS Longitude, Water source, and Sitename values.

## Filter Poor Geolocational Data
Geolocational data may not be the most accurate. As a result, this runs a relatively naive check to remove poor geolocational data. More specifically, if the MGRS coordinates match up with the GPS coordinates or the GPS coordinates are whole numbers, then the entry is considered poor quality.
"""


def filter_invalid_coords(
    df, latitude_col, longitude_col, inclusive=False, inplace=False
):
    """
    Filters latitude and longitude of a DataFrame to lie within the latitude range of [-90, 90] or (-90, 90) and longitude range of [-180, 180] or (-180, 180)

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to filter
    latitude_col : str
        The name of the column that contains latitude values
    longitude_col : str
        The name of the column that contains longitude values
    inclusive : bool, default=False
        True if you would like the bounds of the latitude and longitude to be inclusive e.g. [-90, 90]. Do note that these bounds may not work with certain GIS software and projections.
    inplace : bool, default=False
        Whether to return a new DataFrame. If True then no DataFrame copy is not returned and the operation is performed in place.

    Returns
    -------
    pd.DataFrame or None
        A DataFrame with invalid latitude and longitude entries removed. If `inplace=True` it returns None.
    """
    if not inplace:
        df = df.copy()

    if inclusive:
        mask = (
            (df[latitude_col] >= -90)
            & (df[latitude_col] <= 90)
            & (df[longitude_col] <= 180)
            & (df[longitude_col] >= -180)
        )
    else:
        mask = (
            (df[latitude_col] > -90)
            & (df[latitude_col] < 90)
            & (df[longitude_col] < 180)
            & (df[longitude_col] > -180)
        )

    if not inplace:
        return df[mask]
    else:
        df.mask(~mask, inplace=True)
        df.dropna(inplace=True)


def filter_duplicates(df, columns, group_size, inplace=False):
    """
    Filters possible duplicate data by grouping together suspiciously similar entries.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to filter
    columns : list of str
        The name of the columns that duplicate data would share. This can include things such as MGRS Latitude, MGRS Longitude, measure date, and other fields (e.g. mosquito water source for mosquito habitat mapper).
    group_size : int
        The number of duplicate entries in a group needed to classify the group as duplicate data.
    inplace : bool, default=False
        Whether to return a new DataFrame. If True then no DataFrame copy is not returned and the operation is performed in place.

    Returns
    -------
    pd.DataFrame or None
        A DataFrame with duplicate data removed. If `inplace=True` it returns None.
    """

    if not inplace:
        df = df.copy()

    # groups / filters suspected events
    suspect_df = df.groupby(by=columns).filter(lambda x: len(x) >= group_size)
    suspect_mask = df.isin(suspect_df)

    if not inplace:
        return df[~suspect_mask].dropna(how="all")
    else:
        df.mask(suspect_mask, inplace=True)
        df.dropna(how="all", inplace=True)


def filter_poor_geolocational_data(
    df,
    latitude_col,
    longitude_col,
    mgrs_latitude_col,
    mgrs_longitude_col,
    inplace=False,
):
    """
    Filters latitude and longitude of a DataFrame that contain poor geolocational quality.

    latitude_col : str
        The name of the column that contains latitude values
    longitude_col : str
        The name of the column that contains longitude values
    mgrs_latitude_col : str
        The name of the column that contains MGRS latitude values
    mgrs_longitude_col : str
        The name of the column that contains MGRS longitude values
    inplace : bool, default=False
        Whether to return a new DataFrame. If True then no DataFrame copy is not returned and the operation is performed in place.

    Returns
    -------
    pd.DataFrame or None
        A DataFrame with bad latitude and longitude entries removed. If `inplace=True` it returns None.
    """

    def geolocational_filter(gps_lat, gps_lon, recorded_lat, recorded_lon):
        return (
            (recorded_lat == gps_lat and recorded_lon == gps_lon)
            or gps_lat == int(gps_lat)
            or gps_lon == int(gps_lon)
        )

    if not inplace:
        df = df.copy()

    vectorized_filter = np.vectorize(geolocational_filter)
    bad_data = vectorized_filter(
        df[latitude_col].to_numpy(),
        df[longitude_col].to_numpy(),
        df[mgrs_latitude_col].to_numpy(),
        df[mgrs_longitude_col].to_numpy(),
    )

    filtered_df = df[~bad_data]

    if not inplace:
        return filtered_df
    else:
        df.mask(~df.isin(filtered_df), inplace=True)
        df.dropna(how="all", inplace=True)
