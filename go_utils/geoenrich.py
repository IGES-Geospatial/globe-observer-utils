from datetime import datetime
import numpy as np

from arcgis.features import GeoAccessor
from arcgis.gis import GIS

from go_utils.download import convert_dates_to_datetime, default_data_clean
from go_utils.info import start_date, end_date, region_dict, abbreviation_dict


def get_country_api_data(
    protocol,
    start_date=start_date,
    end_date=end_date,
    is_clean=True,
    countries=[],
    regions=[],
):
    """
    Gets country enriched API Data. Due note that this data comes from layers in ArcGIS that are updated daily. Therefore, there will be some delay between when an entry is uploaded onto the GLOBE data base and being on the ArcGIS dataset.

    Parameters
    ----------
    protocol : str, {"mosquito_habitat_mapper", "land_covers"}
        The desired GLOBE Observer Protocol. Currently only mosquito habitat mapper and land cover is supported.
    start_date : str, default= 2017-05-31
        The desired start date of the dataset in the format of (YYYY-MM-DD).
    end_date : str, default= today's date in YYYY-MM-DD form.
        The desired end date of the dataset in the format of (YYYY-MM-DD).
    countries : list of str, default=[]
        The list of desired countries. Look at go_utils.info.region_dict to see supported country names. If the list is empty, all data will be included.
    regions : list of str, default=[]
        The list of desired regions. Look at go_utils.info.region_dict to see supported region names and the countries they enclose. If the list is empty, all data will be included.
    latlon_box : dict of {str, double}, optional
        The longitudes and latitudes of a bounding box for the dataset. The minimum/maximum latitudes and longitudes must be specified with the following keys: "min_lat", "min_lon", "max_lat", "max_lon". The default value specifies all latitude and longitude coordinates.
    """

    item_id_dict = {
        "mosquito_habitat_mapper": "4e8bdb70b3d6424b8831e9cc621cf3b6",
        "land_covers": "c68acbfc68db4409b495fd4636646aa6",
    }

    if protocol not in item_id_dict:
        raise ValueError(
            "Invalid protocol, currently only 'mosquito_habitat_mapper' and 'land_covers' are supported."
        )

    gis = GIS()
    item = gis.content.get(itemid=item_id_dict[protocol])
    df = GeoAccessor.from_layer(item.layers[0])
    df.drop(["SHAPE"], axis=1, inplace=True)

    # Due to the size of the mhm column names, ArcGIS truncates the names so it must be renamed in this step.
    if protocol == "mosquito_habitat_mapper":

        mhm_rename_dict = {
            "mosquitohabitatmapperAbdomenClo": "mosquitohabitatmapperAbdomenCloseupPhotoUrls",
            "mosquitohabitatmapperBreedingGr": "mosquitohabitatmapperBreedingGroundEliminated",
            "mosquitohabitatmapperLarvaFullB": "mosquitohabitatmapperLarvaFullBodyPhotoUrls",
            "mosquitohabitatmapperLarvaeCoun": "mosquitohabitatmapperLarvaeCount",
            "mosquitohabitatmapperLastIdenti": "mosquitohabitatmapperLastIdentifyStage",
            "mosquitohabitatmapperMeasurem_1": "mosquitohabitatmapperMeasurementLatitude",
            "mosquitohabitatmapperMeasurem_2": "mosquitohabitatmapperMeasurementLongitude",
            "mosquitohabitatmapperMeasuremen": "mosquitohabitatmapperMeasurementElevation",
            "mosquitohabitatmapperMosquitoAd": "mosquitohabitatmapperMosquitoAdults",
            "mosquitohabitatmapperMosquitoEg": "mosquitohabitatmapperMosquitoEggs",
            "mosquitohabitatmapperMosquitoHa": "mosquitohabitatmapperMosquitoHabitatMapperId",
            "mosquitohabitatmapperMosquitoPu": " mosquitohabitatmapperMosquitoPupae",
            "mosquitohabitatmapperMosquito_1": "mosquitohabitatmapperMosquitoEggCount",
            "mosquitohabitatmapperWaterSou_1": "mosquitohabitatmapperWaterSourcePhotoUrls",
            "mosquitohabitatmapperWaterSou_2": "mosquitohabitatmapperWaterSourceType",
            "mosquitohabitatmapperWaterSourc": "mosquitohabitatmapperWaterSource",
        }
        df.rename(
            mhm_rename_dict,
            axis=1,
            inplace=True,
        )

    # Filter the dates
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    measured_at = protocol.replace("_", "") + "MeasuredAt"

    convert_dates_to_datetime(df)

    df = df[(df[measured_at] >= start) & (df[measured_at] <= end)]

    if is_clean:
        df = default_data_clean(df, protocol)

    if countries:
        df = _get_valid_countries(df, protocol, countries)

    if regions:
        for region in regions:
            df = _get_valid_countries(df, protocol, region_dict[region])
    return df


def _get_valid_countries(df, protocol, country_list):
    country_filter = np.vectorize(lambda country_col: country_col in country_list)
    mask = country_filter(df[f"{abbreviation_dict[protocol]}_COUNTRY"].to_numpy())
    return df[mask]
