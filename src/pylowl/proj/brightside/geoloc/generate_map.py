#!/usr/bin/env python


import os


def main(input_filename, output_filename):
    api_key = os.getenv('GOOGLE_API_KEY')

    with open(output_filename, 'w') as out_f:
        out_f.write('''
            <!DOCTYPE html>
            <html>
              <head>
                <meta name="viewport" content="initial-scale=1.0, user-scalable=no" />
                <style type="text/css">
                  html { height: 100%% }
                  body { height: 100%%; margin: 0; padding: 0 }
                  #map-canvas { height: 100%% }
                </style>
                <script type="text/javascript"
                  src="https://maps.googleapis.com/maps/api/js?key=%s&sensor=false">
                </script>
                <script type="text/javascript">
                  function initialize() {
                    var mapOptions = {
                      center: new google.maps.LatLng(37.1756138,-97.1681773),
                      zoom: 5
                    };
                    var map = new google.maps.Map(document.getElementById("map-canvas"),
                        mapOptions);
        ''' % api_key)
        with open(input_filename) as in_f:
            for line in in_f:
                fields = line.rstrip().split('\t')

                state_abbrev = fields[2]
                city = fields[3]
                latitude = fields[4]
                longitude = fields[5]

                out_f.write('''
                    var pos = new google.maps.LatLng(%s, %s);
                    var marker = new google.maps.Marker({
                        position: pos,
                        title: "%s, %s"
                    });
                    marker.setMap(map);
                ''' % (latitude, longitude, city, state_abbrev))
        out_f.write('''
                  }
                  google.maps.event.addDomListener(window, 'load', initialize);
                </script>
              </head>
              <body>
                <div id="map-canvas"/>
              </body>
            </html>
        ''')


if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
