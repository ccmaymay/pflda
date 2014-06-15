#!/usr/bin/env python


import json
import urllib
import os


MQLREAD_URL_BASE = 'https://www.googleapis.com/freebase/v1/mqlread'
STATE_ABBREV_MAP_PATH = 'data/states_abbrev_map.txt'


def load_state_abbrev_map():
    d = dict()
    with open(STATE_ABBREV_MAP_PATH) as f:
        for line in f:
            pieces = line.split()
            d[pieces[0]] = ' '.join(pieces[1:])
    return d


def main(state_abbrev, city_name, fb_id):
    api_key = os.getenv('GOOGLE_API_KEY')

    state_abbrev_map = load_state_abbrev_map()

    query = [{
        'id': fb_id,
        'key': {
            'namespace': '/wikipedia/en_id',
            'value': None
        },
        '/location/location/geolocation': [{
            'latitude': None,
            'longitude': None,
        }]
    }]
    params = dict(query=json.dumps(query))

    if api_key is not None:
        params['key'] = api_key

    url = MQLREAD_URL_BASE + '?' + urllib.urlencode(params)

    response = json.loads(urllib.urlopen(url).read())

    for result in response['result']:
        for geolocation in result['/location/location/geolocation']:
            latitude = geolocation['latitude']
            longitude = geolocation['longitude']
            print '\t'.join((
                fb_id,
                result['key']['value'],
                state_abbrev,
                city_name,
                str(latitude),
                str(longitude)
            ))


if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
