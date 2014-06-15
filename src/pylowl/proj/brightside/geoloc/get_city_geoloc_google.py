#!/usr/bin/env python


import json
import urllib
import os


GEOCODE_URL_BASE = 'https://maps.googleapis.com/maps/api/geocode/json'
STATE_ABBREV_MAP_PATH = 'data/states_abbrev_map.txt'


def load_state_abbrev_map():
    d = dict()
    with open(STATE_ABBREV_MAP_PATH) as f:
        for line in f:
            pieces = line.split()
            d[pieces[0]] = ' '.join(pieces[1:])
    return d


def get_city_geoloc_google(state_abbrev, city_name):
    api_key = os.getenv('GOOGLE_API_KEY')

    state_abbrev_map = load_state_abbrev_map()

    address = '%s, %s' % (city_name, state_abbrev_map[state_abbrev])
    params = dict(address=address.replace(' ', '+'), sensor='false')

    if api_key is not None:
        params['key'] = api_key

    url = GEOCODE_URL_BASE + '?' + urllib.urlencode(params)

    coords = []

    response = json.loads(urllib.urlopen(url).read())
    if 'results' in response:
        for result in response['results']:
            loc = result['geometry']['location']
            latitude = loc['lat']
            longitude = loc['lng']
            coords.append((float(latitude), float(longitude)))

    return coords


def main(state_abbrev, city_name):
    for (latitude, longitude) in get_city_geoloc_google(state_abbrev, city_name):
        print '\t'.join((
            state_abbrev,
            city_name,
            str(latitude),
            str(longitude)
        ))


if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
