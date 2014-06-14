#!/usr/bin/env python


import json
import urllib
import os
from get_city_geoloc_google import get_city_geoloc_google


def main(filename):
    d = dict()

    with open(filename) as f:
        for line in f:
            pieces = line.rstrip().split('\t')

            state_abbrev = pieces[2]
            city = pieces[3]
            latitude = float(pieces[4])
            longitude = float(pieces[5])

            k = state_abbrev + '/' + city
            v = (
                    state_abbrev,
                    city,
                    latitude,
                    longitude,
                    line.rstrip(),
                )
            if k in d:
                d[k].append(v)
            else:
                d[k] = [v]

    for (k, v) in d.items():
        if len(v) == 1:
            print v[0][4]
        else:
            state_abbrev = v[0][0]
            city = v[0][1]
            (glat, glng) = get_city_geoloc_google(state_abbrev, city)[0]
            ds = [(line, (glat - lat)**2 + (glng - lng)**2)
                for (lat, lng, line)
                in (entry[2:] for entry in v)]
            print min(ds, key=lambda x: x[1])[0]


if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
