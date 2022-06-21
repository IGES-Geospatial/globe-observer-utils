import numpy as np
import pandas as pd
import pytest

from go_utils.filtering import (
    filter_duplicates,
    filter_invalid_coords,
    filter_poor_geolocational_data,
)


@pytest.mark.util
@pytest.mark.filtering
def test_duplicate_filter():
    df = pd.DataFrame.from_dict(
        {
            "Latitude": [5, 5, 7, 8],
            "Longitude": [6, 6, 10, 2],
            "attribute1": ["foo", "foo", "foo", "bar"],
            "attribute2": ["baz", "baz", "baz", "baz"],
        }
    )

    filtered_df = filter_duplicates(df, ["Latitude", "Longitude", "attribute1"], 2)

    assert not np.any(
        (filtered_df["Latitude"] == 5)
        & (filtered_df["Longitude"] == 6)
        & (filtered_df["attribute1"] == "foo")
    )

    filtered_df = filter_duplicates(df, ["attribute1", "attribute2"], 3)
    assert not np.any(
        (filtered_df["attribute1"] == "foo") & (filtered_df["attribute2"] == "baz")
    )

    assert not filtered_df.equals(df)
    filter_duplicates(df, ["attribute1", "attribute2"], 3, True)
    assert filtered_df.equals(df)


@pytest.mark.util
@pytest.mark.filtering
def test_poor_geolocational_data_filter():
    df = pd.DataFrame.from_dict(
        {
            "Latitude": [36.5, 37.8, 39.2, 30, 19.2],
            "Longitude": [95.2, 28.6, 15, 13.5, 30.8],
            "MGRSLatitude": [36.5, 37.9, 39.3, 30.2, 19.3],
            "MGRSLongitude": [95.2, 28.6, 15.5, 14, 30.2],
        }
    )
    filtered_df = filter_poor_geolocational_data(
        df, "Latitude", "Longitude", "MGRSLatitude", "MGRSLongitude"
    )

    assert not np.any(
        (filtered_df["Latitude"] == filtered_df["MGRSLatitude"])
        & (filtered_df["Longitude"] == filtered_df["MGRSLongitude"])
    )
    assert not np.any(filtered_df["Latitude"] == filtered_df["Latitude"].astype(int))
    assert not np.any(filtered_df["Longitude"] == filtered_df["Longitude"].astype(int))

    assert not filtered_df.equals(df)
    filter_poor_geolocational_data(
        df, "Latitude", "Longitude", "MGRSLatitude", "MGRSLongitude", True
    )
    assert filtered_df.equals(df)


latlon_data = [
    (
        {
            "lat": [-90, 90, 50, -9999, 0, 2, -10, 36.5, 89.999],
            "lon": [-180, 180, 179.99, -179.99, -9999, 90, -90, 35.6, -17.8],
        }
    ),
    (
        {
            "latitude": [-90, 90, -89.999, -23.26, -9999, 12.75, -10, 36.5, 89.999],
            "longitude": [-180, 22.2, -37.85, -179.99, 180, 90, -90, -9999, 179.99],
        }
    ),
]


@pytest.mark.util
@pytest.mark.filtering
@pytest.mark.parametrize("df_dict", latlon_data)
def test_latlon_filter(df_dict):
    df = pd.DataFrame.from_dict(df_dict)
    latitude, longitude = df.columns

    # Test exclusive filtering
    filtered_df = filter_invalid_coords(df, latitude, longitude)
    assert np.all(filtered_df[latitude] > -90)
    assert np.all(filtered_df[latitude] < 90)
    assert np.all(filtered_df[longitude] > -180)
    assert np.all(filtered_df[longitude] < 180)

    # Test inclusive filtering
    filtered_df = filter_invalid_coords(df, latitude, longitude, inclusive=True)
    assert np.all(filtered_df[latitude] >= -90)
    assert np.all(filtered_df[latitude] <= 90)
    assert np.all(filtered_df[longitude] >= -180)
    assert np.all(filtered_df[longitude] <= 180)

    # Test inplace
    assert not filtered_df.equals(df)
    filter_invalid_coords(df, latitude, longitude, inclusive=True, inplace=True)
    assert filtered_df.equals(df)
