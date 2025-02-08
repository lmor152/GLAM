
==========================================
Geocoding via LINZ Address Matching (GLAM) 
==========================================

Work in Progress!!!
-------------------


This package implements methods for performing entity matching on unstructured NZ addresses to the LINZ street address database for free offline geocoding. This package does not support PO boxes or international addresses. 

Usage::

  from glam.interface import Geocoder

  data_dir = 'data' # Location of glam data dependencies. should include data/nz-street-address.csv
  gc = Geocoder(data_dir)

  addresses = [
    'level 10, 10 customhouse quay, wellington central',
    'third floor 18 viaduct harbour ave auckland',
  ]

  gc.geocode_addresses(addresses)


Output::

  [{'address_number': '10',
    'full_road_name_ascii': 'Customhouse Quay',
    'suburb_locality_ascii': 'Wellington Central',
    'town_city_ascii': 'Wellington',
    'postcode': '6011',
    'shape_X': '174.7784009',
    'shape_Y': '-41.28219255',
    'match_score': 100.0},
  {'address_number': '18',
    'full_road_name_ascii': 'Viaduct Harbour Avenue',
    'suburb_locality_ascii': 'Auckland Central',
    'town_city_ascii': 'Auckland',
    'postcode': '1010',
    'shape_X': '174.7577686',
    'shape_Y': '-36.8453608833',
    'match_score': 100.0}]


Dependencies

- Default models are packaged with the Python wheel.
- Tensorflow is required but not automatically installed to remain OS agnostic
- Data dependencies will be built from the raw LINZ and PNF files the first time using the geocoder (may be slow).
  
  - Raw LINZ dataset can be downloaded from nz-street-address_
  - PNF file is available for purchase from NZ Post PNF_ (optional, but postcodes will not be available without the PNF file. And it may also improve geocoding efficiency)

.. _nz-street-address: https://data.linz.govt.nz/layer/53353-nz-street-address/
.. _PNF: https://www.nzpost.co.nz/business/sending-within-nz/quality-addressing/postcode-network-file

:Author: Liam Morris
:Email: liam.morris04@gmail.com
