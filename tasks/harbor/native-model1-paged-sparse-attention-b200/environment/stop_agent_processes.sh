#!/usr/bin/env bash
# Stop every process the agent left running so nothing changes the workspace
# while Harbor collects it. Harbor runs this from the [[verifier.collect]] hook
# in task.toml, after the agent phase and before artifact collection. PID 1,
# Harbor's `sh -c "sleep infinity"` keepalive, and this script and its
# ancestors are spared. Each signal sent is recorded in
# /logs/artifacts/stopped-agent-processes.txt.
record=/logs/artifacts/stopped-agent-processes.txt

inspect() {
  stat=""
  read -r stat 2>/dev/null < "/proc/$1/stat"
  set -- ${stat##*) }
  state=$1
  parent=$2
}

arguments() {
  args=()
  while IFS= read -r -d '' arg || [ -n "$arg" ]; do
    args+=("$arg")
  done 2>/dev/null < "/proc/$1/cmdline"
}

spared=" 1 "
pid=$$
while [ -n "$pid" ] && [ "$pid" != 0 ] && [ "$pid" != 1 ]; do
  spared="$spared$pid "
  inspect "$pid"
  pid=$parent
done

keepalive_parents=" "
for entry in /proc/[0-9]*; do
  p=${entry#/proc/}
  inspect "$p"
  [ "$p" = 1 ] || [ "$parent" = 1 ] || continue
  arguments "$p"
  if [ "${#args[@]}" = 3 ] && [ "${args[0]}" = sh ] && [ "${args[1]}" = -c ] && [ "${args[2]}" = "sleep infinity" ]; then
    spared="$spared$p "
    keepalive_parents="$keepalive_parents$p "
  fi
done
for entry in /proc/[0-9]*; do
  p=${entry#/proc/}
  inspect "$p"
  case "$keepalive_parents" in *" $parent "*) ;; *) continue ;; esac
  arguments "$p"
  if [ "${#args[@]}" = 2 ] && [ "${args[0]}" = sleep ] && [ "${args[1]}" = infinity ]; then
    spared="$spared$p "
  fi
done

collect_targets() {
  targets=""
  for entry in /proc/[0-9]*; do
    p=${entry#/proc/}
    case "$spared" in *" $p "*) continue ;; esac
    inspect "$p"
    [ -n "$parent" ] || continue
    if [ "$state" = Z ]; then
      set -- "$entry"/task/*
      [ "$#" -le 1 ] && continue
    fi
    targets="$targets $p"
  done
}

signal_targets() {
  now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  for p in $targets; do
    arguments "$p"
    line="$now SIG$1 $p ${args[*]}"
    echo "$line"
    echo "$line" >> "$record" 2>/dev/null
    kill -s "$1" "$p" 2>/dev/null
  done
}

collect_targets
[ -z "$targets" ] && exit 0
signal_targets TERM
for attempt in 1 2 3 4 5 6 7 8 9 10; do
  sleep 0.5
  collect_targets
  [ -z "$targets" ] && exit 0
done
for attempt in 1 2 3 4 5; do
  signal_targets KILL
  sleep 0.5
  collect_targets
  [ -z "$targets" ] && exit 0
done
line="still running:$targets"
echo "$line"
echo "$line" >> "$record" 2>/dev/null
exit 1
