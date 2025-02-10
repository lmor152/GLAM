# Geocoding via LINZ Address Matching (GLAM) 

## Overview
This package implements methods for performing entity matching on unstructured NZ address to the New Zealand Street Address dataset maintained by Land Information New Zealand. This package does not support PO boxes or international addresses.

## Installation
Get the latest version of glam by installing from PyPI

```bash
pip install glam
```

## Usage

To setup glam, the dependencies have to be downloaded first. Alternatively, if this is not done, glam will download dependencies the first time it's used for geocoding. 
```python
from glam import Geocoder, download_dependencies

# directory to store glam's dependencies
download_dependencies("path/to/glamdeps") 
```

```python

addresses = [
    "16 western springs rd morningside",
    "4 ryelands dr, lincoln",
]

gc = Geocoder(
    "path/to/glamdeps", 
    matcher = "tfidf", 
    parser = "rnn"
)

matched_addresses = gc.geocode_addresses(addresses)
print(matched_addresses)
```

Outputs:
```
[
    Search address 16 western springs rd morningside -> matched to 16 Western Springs Road, Morningside, Auckland with 0.899795426050709 confidence,

    Search address 4 ryelands dr lincoln -> matched to 4 Ryelands Drive, Lincoln with 0.8462000263063917 confidence
]
```

Each matched address contains coordinates and other fields from the NZSA dataset:
```python
matched_addresses[0].matched_address.to_dict()
```

Outputs: 
```python
{
    'address_id': 1076278,
    'unit_value': None,
    'address_number': '16',
    'address_number_suffix': None,
    'address_number_high': None,
    'full_road_name': 'Western Springs Road',
    'suburb_locality': 'Morningside',
    'town_city': 'Auckland',
    'full_address_ascii': '16 Western Springs Road, Morningside, Auckland',
    'shape_X': '174.7345083',
    'shape_Y': '-36.8739952167',
    'postcode': '1021'
 }
```

Glam can also be used to parse addresses:

```python
gc.parse_addresses(addresses)
```

Outputs:
```
[
    ParsedAddress{'unit': None, 'building': None, 'level': None, 'first_number': '16', 'first_number_suffix': None, 'second_number': None, 'street_name': 'WESTERN SPRINGS ROAD', 'suburb_town_city': 'MORNINGSIDE', 'postcode': None},

    ParsedAddress{'unit': None, 'building': None, 'level': None, 'first_number': '4', 'first_number_suffix': None, 'second_number': None, 'street_name': 'RYELANDS DRIVE', 'suburb_town_city': 'LINCOLN', 'postcode': None}
]
```

## Available Models
Models are divided into parsers and matchers. Some matchers work directly on unstructured address strings, others require a parser to add structure to the addresses before matching.

### Matchers
- TFIDF (default): uses term frequency inverse document frequency to match unstructured addresses directly
- Fuzzy: uses component-wise fuzzy matching on parsed addresses 
- Vector: uses custom address vectorisation logic to find nearest match
- Embedding: uses deep learned embeddings to vectorise addresses and find the nearest match

### Parsers
- RNN: uses an LSTM to parse unstructured address strings
- libpostal: a wrapper around the postal Python package (requires libpostal to be installed)

TFIDF is recommended for most use cases as it's the most accurate method. Vector is the fastest method but has reduced accuracy and robustness compared to TFIDF. Note that by default, dependencies are only downloaded for TFIDF matching. If another method is selected, the dependencies will be built the first time it is used and saved for future use.

## Custom Address Datasets
Glam can be used to match to other address datasets by placing your dataset in the dependencies directory under `nz-street-address.csv`. Dependencies should be rebuilt by deleting the matching directory under the dependency directory to ensure they are referencing the new address database.

## Author
Liam Morris - lmor152