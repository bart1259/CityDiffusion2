CONFIG = {
    "backgroundColor": (0,0,0),
    "defaultTag": "rural",

    "features": [
        # -------------------------------------------------------------
        # 1. Broad land-cover layers
        # -------------------------------------------------------------

        {
            "name": "agriculture",
            "color": (205, 190, 95),
            "boundaryColor": None,
            "extractFunction": lambda x: x.get_landuse(
                custom_filter={
                    "landuse": [
                        "farmland",
                        "farmyard",
                        "orchard",
                        "vineyard",
                        "allotments",
                        "greenhouse_horticulture",
                    ]
                }
            ),
            "geometryFunction": lambda x: x,
            "tag": "rural"
        },
        {
            # Forest mapped as landuse=forest.
            # This is separate from natural=wood below, because both
            # tagging styles are common in OSM.
            "name": "managed_forest",
            "color": (45,95,45),
            "boundaryColor": None,
            "extractFunction": lambda x: x.get_landuse(
                custom_filter={
                    "landuse": [
                        "forest",
                    ]
                }
            ),
            "geometryFunction": lambda x: x,
            "tag": "rural"
        },
        {
            # Common Dutch managed green-space / pasture tags.
            "name": "grass_and_meadow",
            "color": (105, 150, 80),
            "boundaryColor": None,
            "extractFunction": lambda x: x.get_landuse(
                custom_filter={
                    "landuse": [
                        "grass",
                        "meadow",
                        "village_green",
                    ]
                }
            ),
            "geometryFunction": lambda x: x,
            "tag": "rural"
        },
        {
            # These areas are frequently absent when only buildings and
            # roads are rendered.
            "name": "urban_landuse",
            "color": (75, 75, 75),
            "boundaryColor": None,
            "extractFunction": lambda x: x.get_landuse(
                custom_filter={
                    "landuse": [
                        "residential",
                        "commercial",
                        "retail",
                        "industrial",
                        "construction",
                        "brownfield",
                        "garages",
                        "cemetery",
                    ]
                }
            ),
            "geometryFunction": lambda x: x,
            "tag": "urban"
        },
        {
            "name": "infrastructure_landuse",
            "color": (95, 95, 95),
            "boundaryColor": None,
            "extractFunction": lambda x: x.get_landuse(
                custom_filter={
                    "landuse": [
                        "railway",
                        "quarry",
                        "landfill",
                    ]
                }
            ),
            "geometryFunction": lambda x: x,
            "tag": "urban"
        },

        # -------------------------------------------------------------
        # 2. Natural land cover
        # Drawn above general landuse so natural features remain visible.
        # -------------------------------------------------------------

        {
            "name": "forests_and_woods",
            "color": (45, 95, 45),
            "boundaryColor": None,
            "extractFunction": lambda x: x.get_natural(
                custom_filter={
                    "natural": [
                        "wood",
                        "scrub",
                    ]
                }
            ),
            "geometryFunction": lambda x: x,
            "tag": "rural"
        },
        {
            "name": "heath_and_natural_grassland",
            "color": (175, 125, 70),
            "boundaryColor": None,
            "extractFunction": lambda x: x.get_natural(
                custom_filter={
                    "natural": [
                        "heath",
                        "grassland",
                    ]
                }
            ),
            "geometryFunction": lambda x: x,
            "tag": "rural"
        },
        {
            "name": "wetlands",
            "color": (105, 145, 125),
            "boundaryColor": None,
            "extractFunction": lambda x: x.get_natural(
                custom_filter={
                    "natural": [
                        "wetland",
                        "marsh",
                        "reedbed",
                        "bog",
                    ]
                }
            ),
            "geometryFunction": lambda x: x,
            "tag": "rural"
        },
        {
            "name": "sand_and_bare_ground",
            "color": (210, 190, 145),
            "boundaryColor": None,
            "extractFunction": lambda x: x.get_natural(
                custom_filter={
                    "natural": [
                        "sand",
                        "beach",
                        "bare_rock",
                    ]
                }
            ),
            "geometryFunction": lambda x: x,
            "tag": "rural"
        },

        # -------------------------------------------------------------
        # 3. Water
        # Water should sit above land-cover layers, but below roads and
        # buildings so bridges/roads/buildings remain visible.
        # -------------------------------------------------------------

        {
            "name": "water_reservoirs_and_basins",
            "color": (0, 95, 230),
            "boundaryColor": None,
            "extractFunction": lambda x: x.get_landuse(
                custom_filter={
                    "landuse": [
                        "reservoir",
                        "basin",
                    ]
                }
            ),
            "geometryFunction": lambda x: x,
            "tag": "water"
        },
        {
            "name": "water",
            "color": (0, 95, 230),
            "boundaryColor": None,
            "extractFunction": lambda x: x.get_natural(
                custom_filter={
                    "natural": [
                        "water",
                    ]
                }
            ),
            "geometryFunction": lambda x: x,
            "tag": "water"
        },

        # -------------------------------------------------------------
        # 4. Transport
        # Draw after area fills, otherwise landuse/water polygons can
        # paint over roads.
        # -------------------------------------------------------------

        {
            "name": "roads",
            "color": (40, 210, 40),
            "boundaryColor": None,
            "extractFunction": lambda x: x.get_network(
                network_type="all",
                nodes=False,
            ),
            # Assumes your data has a projected CRS with metre units.
            "geometryFunction": lambda x: x.geometry.buffer(
                4, # 4 meter wide roads by default
                cap_style=2,
                join_style=2,
            ),
            "tag": "urban"
        },

        # -------------------------------------------------------------
        # 5. Buildings
        # Buildings are last so they appear above all landuse and road
        # fills.
        # -------------------------------------------------------------

        {
            "name": "buildings",
            "color": (255, 0, 0),
            "boundaryColor": (150, 0, 0),
            "extractFunction": lambda x: x.get_buildings(),
            "geometryFunction": lambda x: x,
            "tag": "urban"
        },
    ],
}