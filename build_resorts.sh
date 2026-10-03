#!/bin/bash
# Build many real courses for the terrain bank. Failures are logged and skipped.
cd "$(dirname "$0")"
while IFS='|' read -r resort run; do
  [ -z "$resort" ] && continue
  echo "=== $resort ${run:+(run: $run)}"
  if [ -n "$run" ]; then
    timeout 240 .venv/bin/python resort.py "$resort" --run "$run" --no-render 2>&1 | grep -vi "warn" | tail -3
  else
    timeout 240 .venv/bin/python resort.py "$resort" --no-render 2>&1 | grep -vi "warn" | tail -3
  fi
  sleep 2
done <<'LIST'
Kitzbühel|streif
Wengen|lauberhorn
Adelboden|chuenis
Schladming|planai
Garmisch-Partenkirchen|kandahar
Val d'Isère|bellevarde
Zermatt|
St. Moritz|
Chamonix|
Sölden|
Courchevel|
Åre|
Levi, Finland|
Whistler Blackcomb|
Aspen Mountain|
Jackson Hole Mountain Resort|
Palisades Tahoe|
Mammoth Mountain|
Big Sky Resort|
Lake Louise Ski Resort|
Park City Mountain|
Copper Mountain|
Niseko|
Hakuba Happo-one|
Portillo, Chile|
Cerro Catedral|
Coronet Peak|
Thredbo|
Gulmarg|
Lech|
LIST
echo "=== all done"
