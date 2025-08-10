import googlemaps
import json2xml
import os

class Geocoder:

    def __init__(self, api_key):
        self.api_key = api_key

        if os.path.isfile(self.api_key):
            with open(self.api_key) as f:
                self.api_key = f.read().strip()

        self.gmaps = googlemaps.Client(key=self.api_key)

    def reverse_geocode(
        self,
        lat,
        lon,
        return_as_xml=False,
        extra_attrs=None,  # a dict
        exclude_attributes=["street_number", "route", "postal_code"],
    ):

        r = self.gmaps.reverse_geocode([lat, lon])[0]
        r = {
            i["types"][0]: i["long_name"]
            for i in r["address_components"]
            if i["types"][0] not in exclude_attributes
        }
        r["coords"] = {"lon": f"{lon:.4f}", "lat": f"{lat:.4f}"}
        if extra_attrs is not None:
            r.update(extra_attrs)
        if return_as_xml:
            r = "\n".join(
                json2xml.Json2xml(
                    r,
                    item_wrap=True,
                    wrapper="location",
                    pretty=True,
                    attr_type=False,
                    root=True,
                )
                .to_xml()
                .split("\n")[1:]
            )

        return r