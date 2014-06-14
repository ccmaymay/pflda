#!/usr/bin/env python


import json
import urllib
import os


MQLREAD_URL_BASE = 'https://www.googleapis.com/freebase/v1/mqlread'


def main(state_abbrev):
    api_key = os.getenv('GOOGLE_API_KEY')

    query = {
        'id': None,
        'key': {
            'namespace': '/wikipedia/en_id',
            'value': None
        },
        'name': None,
        'type': '/location/us_state',
        '/location/administrative_division/iso_3166_2_code': 'US-' + state_abbrev,
    }
    params = dict(query=json.dumps(query))

    if api_key is not None:
        params['key'] = api_key

    url = MQLREAD_URL_BASE + '?' + urllib.urlencode(params)

    response = json.loads(urllib.urlopen(url).read())
    result = response['result']

    print '\t'.join((
        result['id'],
        result['key']['value'],
        state_abbrev,
        result['name']
    ))


if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
