
To check parameter sampling is working or not
```
python3 -W ignore hparam_search.py --env boxing_v1,double_dunk_v2 --local --frames 1 --num-eval-episodes 1 --device cuda --num-trials=2 --trainer-type shared_rainbow
```