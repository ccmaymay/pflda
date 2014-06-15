#!/usr/bin/env python


import json
import urllib
import os


MQLREAD_URL_BASE = 'https://www.googleapis.com/freebase/v1/search'
STATE_ABBREV_MAP_PATH = 'data/states_abbrev_map.txt'


def load_state_abbrev_map():
    d = dict()
    with open(STATE_ABBREV_MAP_PATH) as f:
        for line in f:
            pieces = line.split()
            d[pieces[0]] = ' '.join(pieces[1:])
    return d


def main(state_abbrev, city_name, location_type=None, container_in_filter=False):
    if location_type is None:
        location_type = '/location/citytown'

    api_key = os.getenv('GOOGLE_API_KEY')

    state_abbrev_map = load_state_abbrev_map()

    if container_in_filter:
        query = city_name
        filt = '(all type:%s /location/location/containedby:"%s")' % (location_type, state_abbrev_map[state_abbrev])
    else:
        query = '%s, %s' % (city_name, state_abbrev)
        filt = '(all type:%s)' % location_type
    params = dict(query=query, filter=filt)

    if api_key is not None:
        params['key'] = api_key

    url = MQLREAD_URL_BASE + '?' + urllib.urlencode(params)

    response = json.loads(urllib.urlopen(url).read())
    results = response.get('result', [])
    if results:
        result = results[0]
        print '\t'.join((result['id'], state_abbrev, city_name))


if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
