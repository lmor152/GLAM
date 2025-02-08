from collections.abc import Sequence
from pathlib import Path
from typing import Literal, final

import pandas as pd

from glam import logs

Address = str
AddressID = str
AddressIDs = Sequence[AddressID]
Confidence = float | int | Literal[0]
Confidences = Sequence[Confidence]
Addresses = Sequence[Address]
AddressComponent = str | int | None

logger = logs.get_logger()


@final
class LINZAddress:
    __slots__ = [
        "address_id",
        "unit_value",
        "address_number",
        "address_number_suffix",
        "address_number_high",
        "full_road_name",
        "suburb_locality",
        "town_city",
        "full_address_ascii",
        "shape_X",
        "shape_Y",
        "postcode",
    ]

    def __init__(
        self,
        address_id: str | int,
        unit_value: str,
        address_number: str,
        address_number_suffix: str,
        address_number_high: str,
        full_road_name: str,
        suburb_locality: str,
        town_city: str,
        full_address_ascii: str,
        shape_X: float,
        shape_Y: float,
        postcode: str | None = None,
        **kwargs: object,
    ) -> None:
        self.address_id: int = int(address_id)
        self.unit_value: str = unit_value
        self.address_number: str = address_number
        self.address_number_suffix: str = address_number_suffix
        self.address_number_high: str = address_number_high
        self.full_road_name: str = full_road_name
        self.suburb_locality: str = suburb_locality
        self.town_city: str = town_city
        self.full_address_ascii: str = full_address_ascii
        self.shape_X: float = shape_X
        self.shape_Y: float = shape_Y
        self.postcode = postcode

    def __repr__(self) -> str:
        if self.address_id < 0:
            return "No Match"
        return self.full_address_ascii

    def to_dict(self) -> dict[str, str | None]:
        return {slot: getattr(self, slot) for slot in self.__slots__}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LINZAddress):
            return False
        return self.to_dict == other.to_dict


LINZAddresses = Sequence[LINZAddress]


@final
class NZSA:
    df: pd.DataFrame | None = None
    url: str = "https://r2.lmor152.com/nzsa.csv.zip"
    postcodes = False

    @classmethod
    def load(cls, path: Path | str) -> None:
        if cls.df is not None:
            logger.debug("Skipping NZSA load as data is already loaded")
            return

        df = pd.read_csv(path, dtype=str)  # type: ignore

        df["address_id"] = df["address_id"].astype(int)

        # check postcodes
        if "postcode" in df.columns:
            cls.postcodes = True
            df["postcode_int"] = pd.to_numeric(df["postcode"])

        # add some useful columns
        df["address_number_int"] = pd.to_numeric(df["address_number"])
        df["suburb_town_city"] = (
            df["suburb_locality_ascii"].fillna("")
            + ", "
            + df["town_city_ascii"].fillna("")
        )
        df["suburb_town_city"] = df["suburb_town_city"].str.replace(
            "^, |, $", "", regex=True
        )  # remove leading and trailing commas and spaces

        df = df.replace({pd.NA: None, "": None})

        cls.df = df

    @classmethod
    def integrate_pnf(cls, PNF: str, save_path: Path) -> None:
        """Integrate postcodes for geocoding via a NZ Post PNF file"""

        import geopandas as gpd

        pnf = gpd.read_file(PNF)
        gdf = gpd.GeoDataFrame(
            cls.df[["address_id", "shape_X", "shape_Y"]],
            geometry=gpd.points_from_xy(cls.df.shape_X, cls.df.shape_Y),
            crs=4326,
        )
        gdf = gdf.sjoin(pnf, how="left", predicate="within")

        cls.df["postcode"] = gdf["POSTCODE"]

        cls.df.to_csv(save_path, index=False)

        cls.postcodes = True

    @classmethod
    def integrate_paf(cls, PAF: str, save_path: Path) -> None:
        """Integrate postcodes for geocoding via a NZ Post PAF file"""
        msg = "Integrating PAF data is not yet supported"
        raise NotImplementedError(msg)
        cls.postcodes = True

    @classmethod
    def get_addresses(cls, address_ids: Sequence[AddressID]) -> LINZAddresses:
        if cls.df is None:
            raise ValueError("NZSA data not loaded")

        temp = pd.DataFrame(address_ids, columns=["address_id"])

        temp = temp.merge(cls.df, on="address_id", how="left")
        return temp.apply(lambda x: LINZAddress(**x), axis=1).to_list()
