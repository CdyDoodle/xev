#!/usr/bin/env bash
# Step 7 check: start the service, call it with a few reviews, time 30 requests, stop it.
source /mnt/g/dev/repo/xev/sentiment/env.sh
mkdir -p runs/serve
uv run uvicorn serve:app --host 127.0.0.1 --port 8010 > runs/serve/server.log 2>&1 &
pid=$!
for i in $(seq 1 120); do curl -s localhost:8010/health > /dev/null 2>&1 && break; sleep 1; done
echo "== health"; curl -s localhost:8010/health; echo
ask() { curl -s localhost:8010/v1/sentiment -H 'content-type: application/json' -d "{\"text\": \"$1\"}"; echo; }
echo "== examples"
ask "房间很干净，前台服务也很热情，下次还会再来。"
ask "电脑用了三天就蓝屏，客服还一直推脱，非常失望。"
ask "书的内容还行，就是印刷有点模糊。"
ask "送餐很快，味道也不错。"
ask "住这个酒店实在是太享受了，还可以免费听隔壁 KTV 的高音演奏。"
echo "== latency (30 requests, ms as reported by the server)"
for i in $(seq 1 30); do ask "房间很干净，前台服务也很热情，下次还会再来。"; done \
  | python3 -c "import sys,json; v=sorted(json.loads(l)['latency_ms'] for l in sys.stdin if l.strip()); print('median', v[len(v)//2], 'p90', v[int(len(v)*.9)], 'max', v[-1])"
kill $pid; wait $pid 2>/dev/null
echo "== stopped"
