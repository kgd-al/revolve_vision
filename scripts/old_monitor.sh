#!/bin/bash

if [ "$1" == '--gnuplot' ]
then
  process(){
    clear
    gnuplot -e "
      set terminal dumb;
      set key above;
      set xlabel 'Samples';
      set ylabel 'CPU';
      set y2label 'MEM';
      set yrange [0:*];
      set y2range [0:*];
      set y2tics;
      plot '$log' using 0:2 with lines title 'cpu',
              '' using 0:3 with lines axes x1y2 title 'mem';
      "
    tail -n1 $log
    sleep 1
  }
else
  process(){
    tail -n1 $log
    sleep 1
  }
fi

export LC_ALL=C
log=memory.log
while :
do
  (
  date -Iseconds | tr '\n' ' '
  ps -u $USER -o "%cpu=,%mem=" \
  | awk '{cpu += $1; mem += $2; }
     END {printf "%.2f%% %.2f%%", cpu, mem; }'
  free | awk '/Mem/{printf " %.2f%%\n", 100*($2-$7)/$2}'
  ) >> $log
  process
done
