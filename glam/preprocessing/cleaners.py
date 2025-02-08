import re

from unidecode import unidecode

from glam.types import Address


def clean_postcodes(addresses: list[Address]) -> list[Address]:
    """Fixes postcodes like xxx.0 and xxxx.0"""

    # replacing postcodes like xxxx.0 with xxxx
    addresses = [re.sub(r"\b([0-9]{4})\.0", r"\1", addy) for addy in addresses]

    # replacing postcodes like xxx.0 with 0xxx
    return [re.sub(r"\b([0-9]{3})\.0", r"0\1", addy) for addy in addresses]


def clean_rural_delivery(addresses: list[Address]) -> list[Address]:
    """Removes RD xx from addresses"""
    return [
        re.sub(r"\bRD.{0,2}[0-9]{1,2}", "", addy, flags=re.IGNORECASE)
        for addy in addresses
    ]


def clean_PObox(addresses: list[Address]) -> list[Address]:
    """Replaces PO Box addresses with None"""
    return [
        addy if not bool(re.search(r"PO.{0,2}Box", addy, flags=re.IGNORECASE)) else " "
        for addy in addresses
    ]


def clean_NZ(addresses: list[Address]) -> list[Address]:
    """Removes NZ from the end of addresses"""
    return [
        re.sub(r"\bNew Zealand\s{0,}$|\bNZ\s{0,}$", r"", addy, flags=re.IGNORECASE)
        for addy in addresses
    ]


def clean_whitespace(addresses: list[Address]) -> list[Address]:
    addresses = [re.sub(r"\s{1,}", r" ", addy) for addy in addresses]
    return [re.sub(r"^\s{1,}\b|\b\s{1,}$", r"", addy) for addy in addresses]


def clean_ascii(addresses: list[Address]) -> list[Address]:
    return [unidecode(addy).lower() for addy in addresses]


all_cleaners = [
    clean_postcodes,
    clean_rural_delivery,
    clean_PObox,
    clean_NZ,
    clean_whitespace,
    clean_ascii,
]
