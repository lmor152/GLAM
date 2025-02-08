street_types = [
    'ACCESS','ACCESSWAY','ALLEY','ANCHORAGE','APPROACH','ARCADE','ARCH','AVENUE','BANK','BAY','BEACH','BELT',
    'BEND','BLUFF','BOULEVARD','BRAE','BRIARS','BRIDGE','BYPASS','CENTRE','CHASE','CIRCLE','CIRCUIT','CIRCUS','CLAIM','CLOSE',
    'CORNER','COMMON','COURT','COURTS','COVE','CREEK','CRESCENT','CREST','CUL','DALE','DELL','DEVIATION','DOWNS','DRIVE','DUNE',
    'ELM','END','ENTRANCE','ESPLANADE','ESTATE','EXPRESSWAY','FAIRWAY','FALL','FARE','FARMS','FEN','FERN','FREEWAY','FLAT',
    'FLATS','GARDEN','GARDENS','GATE','GLADE','GLEN','GRANGE','GREEN','GROVE','GULLY','HAVEN','HEAD','HEIGHTS','HIGHWAY','HILL',
    'ISLAND','JUNCTION','KEY','KNOB','LADDER','LANDING','LANE','LEA','LEADER','LEIGH','LINE','LINK','LOOKOUT','LOOP','MALL',
    'MEAD','MEADOWS','MEWS','MILE','MOTORWAY','MOTU','MOUNT','NEAVES','OAKS','PADDOCK','PAKU','PARADE','PARK','PARKWAY','PASS',
    'PASSAGE','PATH','PLACE','PLAZA','POINT','PRIORS','PROMENADE','QUADRANT','QUAY','REEF','RESERVE','REST','RETREAT','RIDGE',
    'RISE','ROAD','ROADS','ROADWAY','ROUTE','ROW','SERVICE LANE','SLOPE','SPA','SPUR','SQUARE','STATE HIGHWAY','STEEP','STEPS',
    'STRAIGHT','STRAND','STREET','TERRACE','TOWERS','TRACK','TRAIL','TRAMWAY','TREES','VALE','VALLEY','VENUS','VIEW','VIEWS',
    'VILLAGE','VILLAS','VISTA','VUE','WALK','WATERS','WAY','WHARF','WYND'
]

street_abbreviations = {
    'ACCS': 'ACCESS','ACCSWY': 'ACCESSWAY','ALY': 'ALLEY','ANCG': 'ANCHORAGE','APP': 'APPROACH','ARC': 'ARCADE',
    'AV' : 'AVENUE','AVE': 'AVENUE','BNK': 'BANK','BCH': 'BEACH','BND': 'BEND','BLF': 'BLUFF','BLVD': 'BOULEVARD','BR': 'BRAE',
    'BRG': 'BRIDGE','BYP': 'BYPASS','CTR': 'CENTRE','CH': 'CHASE','CIR': 'CIRCLE','CCT': 'CIRCUIT','CRCS': 'CIRCUS','CLM': 'CLAIM',
    'CL': 'CLOSE','CNR': 'CORNER','CMN': 'COMMON','CRT': 'COURT','CRTS': 'COURTS','CV': 'COVE','CRK': 'CREEK','CR':'CRESCENT','CRES': 'CRESCENT',
    'CRST': 'CREST','DLE': 'DALE','DEL': 'DELL','DVN': 'DEVIATION','DOWNS': 'DOWNS','DR': 'DRIVE','DUNE': 'DUNE','ENT': 'ENTRANCE',
    'ESP': 'ESPLANADE','EST': 'ESTATE','EXP': 'EXPRESSWAY','FAWY': 'FAIRWAY','FRMS': 'FARMS','FEN': 'FEN','FWY': 'FREEWAY','FLT': 'FLAT',
    'FLTS': 'FLATS','GDN': 'GARDEN','GDNS': 'GARDENS','GTE': 'GATE','GLD': 'GLADE','GLN': 'GLEN','GRG': 'GRANGE','GRN': 'GREEN',
    'GRV': 'GROVE','GLY': 'GULLY','HVN': 'HAVEN','HTS': 'HEIGHTS','HWY': 'HIGHWAY','HL': 'HILL','IS': 'ISLAND','JCT': 'JUNCTION',
    'JNC': 'JUNCTION','LADR': 'LADDER','LNDG': 'LANDING','LEDR': 'LEADER','LGH': 'LEIGH','LKT': 'LOOKOUT','MDWS': 'MEADOWS','MWY': 'MOTORWAY',
    'MOTU': 'MOTU','MT': 'MOUNT','NVS': 'NEAVES','OAKS': 'OAKS','PADK': 'PADDOCK','PDE': 'PARADE','PK': 'PARK','PKWY': 'PARKWAY',
    'PASS': 'PASS','PSGE': 'PASSAGE','PTH': 'PATH','PL': 'PLACE','PLZ': 'PLAZA','PT': 'POINT','PRIORS': 'PRIORS','PROM': 'PROMENADE',
    'QDRT': 'QUADRANT','QY': 'QUAY','RES': 'RESERVE','REST': 'REST','RTR': 'RETREAT','RDGE': 'RIDGE','RISE': 'RISE','RD': 'ROAD',
    'RDS': 'ROADS','RDWY': 'ROADWAY','RTE': 'ROUTE','SVLN': 'SERVICE LANE','SLP': 'SLOPE','SPUR': 'SPUR','SQ': 'SQUARE','SH': 'STATE HIGHWAY',
    'STEEP': 'STEEP','STPS': 'STEPS','STGT': 'STRAIGHT','STRD': 'STRAND','ST': 'STREET','TCE': 'TERRACE','TWRS': 'TOWERS','TRK': 'TRACK',
    'TRL': 'TRAIL','TMWY': 'TRAMWAY','TRS': 'TREES','VALE': 'VALE','VLY': 'VALLEY','VNUS': 'VENUS','VW': 'VIEW','VWS': 'VIEWS',
    'VLG': 'VILLAGE','VLLS': 'VILLAS','VIS': 'VISTA','WLK': 'WALK','WATERS': 'WATERS','WHRF': 'WHARF',
}

street_abbreviations_reversed = {v: k for k, v in street_abbreviations.items()}

ordinal_words = [
    'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth', 'eleventh',
    'twelfth', 'thirteenth', 'fourteenth', 'fifteenth', 'sixteenth', 'seventeenth', 'eighteenth', 'nineteenth',
    'twentieth', 'twenty-first', 'twenty-second', 'twenty-third', 'twenty-fourth', 'twenty-fifth', 'twenty-sixth',
    'twenty-seventh', 'twenty-eighth', 'twenty-ninth', 'thirtieth', 'thirty-first', 'thirty-second', 'thirty-third',
    'thirty-fourth', 'thirty-fifth', 'thirty-sixth', 'thirty-seventh', 'thirty-eighth', 'thirty-ninth', 'fortieth',
    'forty-first', 'forty-second', 'forty-third', 'forty-fourth', 'forty-fifth', 'forty-sixth', 'forty-seventh',
    'forty-eighth', 'forty-ninth', 'fiftieth', 'fifty-first', 'fifty-second', 'fifty-third', 'fifty-fourth',
    'fifty-fifth', 'fifty-sixth', 'fifty-seventh', 'fifty-eighth', 'fifty-ninth', 'sixtieth', 'sixty-first',
    'sixty-second', 'sixty-third', 'sixty-fourth', 'sixty-fifth', 'sixty-sixth', 'sixty-seventh', 'sixty-eighth',
    'sixty-ninth', 'seventieth', 'seventy-first', 'seventy-second', 'seventy-third', 'seventy-fourth', 'seventy-fifth',
    'seventy-sixth', 'seventy-seventh', 'seventy-eighth', 'seventy-ninth', 'eightieth', 'eighty-first', 'eighty-second',
    'eighty-third', 'eighty-fourth', 'eighty-fifth', 'eighty-sixth', 'eighty-seventh', 'eighty-eighth', 'eighty-ninth',
    'ninetieth', 'ninety-first', 'ninety-second', 'ninety-third', 'ninety-fourth', 'ninety-fifth', 'ninety-sixth',
    'ninety-seventh', 'ninety-eighth', 'ninety-ninth', 'one-hundredth'
]

cardinal_words = [
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen',
    'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty', 'twenty-one', 'twenty-two',
    'twenty-three', 'twenty-four', 'twenty-five', 'twenty-six', 'twenty-seven', 'twenty-eight', 'twenty-nine', 'thirty',
    'thirty-one', 'thirty-two', 'thirty-three', 'thirty-four', 'thirty-five', 'thirty-six', 'thirty-seven',
    'thirty-eight', 'thirty-nine', 'forty', 'forty-one', 'forty-two', 'forty-three', 'forty-four', 'forty-five',
    'forty-six', 'forty-seven', 'forty-eight', 'forty-nine', 'fifty', 'fifty-one', 'fifty-two', 'fifty-three',
    'fifty-four', 'fifty-five', 'fifty-six', 'fifty-seven', 'fifty-eight', 'fifty-nine', 'sixty', 'sixty-one',
    'sixty-two', 'sixty-three', 'sixty-four', 'sixty-five', 'sixty-six', 'sixty-seven', 'sixty-eight', 'sixty-nine',
    'seventy', 'seventy-one', 'seventy-two', 'seventy-three', 'seventy-four', 'seventy-five', 'seventy-six',
    'seventy-seven', 'seventy-eight', 'seventy-nine', 'eighty', 'eighty-one', 'eighty-two', 'eighty-three',
    'eighty-four', 'eighty-five', 'eighty-six', 'eighty-seven', 'eighty-eight', 'eighty-nine', 'ninety', 'ninety-one',
    'ninety-two', 'ninety-three', 'ninety-four', 'ninety-five', 'ninety-six', 'ninety-seven', 'ninety-eight',
    'ninety-nine', 'one-hundred'
]

level_types = [
    'BASEMENT', 'GROUND FLOOR','GROUND LVL', 'GROUND', 'GROUND LEVEL', 'LOBBY', 'LOWER GROUND FLOOR', 'MEZZANINE', 'OBSERVATION DECK',
    'PARKING', 'PENTHOUSE', 'PLATFORM', 'PODIUM', 'ROOFTOP', 'SUB-BASEMENT', 'UPPER GROUND FLOOR'
]

dwelling_types = [
    'ANTENNA', 'APARTMENT', 'BLOCK', 'BOATSHED', 'BUILDING',
    'BUNGALOW', 'CAGE', 'CARPARK', 'CARSPACE', 'CLUB', 'COOLROOM', 'COTTAGE', 'DUPLEX', 'FACTORY', 'FLAT',
    'GARAGE', 'HALL', 'HOUSE', 'KIOSK', 'LEASE', 'LOBBY', 'LOFT', 'LOT', 'MAISONETTE', 'MARINE BERTH',
    'OFFICE', 'PENTHOUSE', 'REAR', 'RESERVE', 'ROOM', 'SECTION', 'SHED', 'SHOP', 'SHOWROOM', 'SIGN', 'SITE',
    'STALL', 'STORE', 'STRATA UNIT', 'STUDIO', 'SUBSTATION', 'SUITE', 'TENANCY', 'TOWER', 'TOWNHOUSE',
    'UNIT', 'VAULT', 'VILLA', 'WARD', 'WAREHOUSE', 'WORKSHOP'
]

def num2word(value, output='ordinal_words'):
    """
    Converts zero or a *positive* integer (or their string
    representations) to an ordinal/cardinal value.
    :param value: the number to convert
    :param output: one of 'ordinal_words', 'ordinal', 'cardinal'
    """
    try:
        value = int(value)
    except ValueError:
        return value

    assert output in (
    'ordinal_words', 'ordinal', 'cardinal'), "`output` must be one of 'ordinal_words', 'ordinal' or 'cardinal'"

    if output == 'ordinal_words' and (0 < value < 100):
        val = ordinal_words[value - 1]
    elif output == 'ordinal_words':
        raise ValueError("'ordinal_words' only supported between 1 and 100")
    elif output == 'ordinal':
        if value % 100 // 10 != 1:
            if value % 10 == 1:
                val = u"%d%s" % (value, "st")
            elif value % 10 == 2:
                val = u"%d%s" % (value, "nd")
            elif value % 10 == 3:
                val = u"%d%s" % (value, "rd")
            else:
                val = u"%d%s" % (value, "th")
        else:
            val = u"%d%s" % (value, "th")
    else:
        val = cardinal_words[value - 1]

    return val.upper()
