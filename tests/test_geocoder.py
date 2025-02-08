def test_geocoder() -> None:
    from glam import Geocoder

    gc = Geocoder("test_deps")
    res = gc.parse_addresses(["16 Western Springs Road, Morningside, Auckland 1021"])[0]

    assert res.street_name.upper() == "WESTERN SPRINGS ROAD"
    assert res.first_number == "16"
    assert res.postcode == "1021"
    assert "MORNINGSIDE" in res.suburb_town_city.upper()
    assert "AUCKLAND" in res.suburb_town_city.upper()
