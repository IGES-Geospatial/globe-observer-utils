import pytest
import json
from go_utils.download import get_api_data, parse_api_data
from test_data import sample_lc_json, sample_mhm_json, globe_down_json

globe_protocols = ["mosquito_habitat_mapper", "land_covers"]
globe_test_data = [sample_lc_json, sample_mhm_json]


@pytest.mark.parametrize("protocol", globe_protocols)
def test_default_download(protocol):
    try:
        df = get_api_data(protocol)
        assert not df.empty
    except RuntimeError:
        pass


def test_bad_api_call():
    with pytest.raises(RuntimeError, match="settings"):
        get_api_data("asdf")


def test_globe_down():
    with pytest.raises(RuntimeError, match="down"):
        parse_api_data(json.loads(globe_down_json))


@pytest.mark.parametrize("test_data", globe_test_data)
def test_globe_normal(test_data):
    df = parse_api_data(json.loads(sample_lc_json))
    assert not df.empty
