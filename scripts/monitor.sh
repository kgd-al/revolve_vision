#!/bin/bash

export LC_ALL=C

sum(){
    scale=$1
    [ -z $scale ] && scale=1
    awk '{sum += $NF}
         /python/{psum += $NF}
         /'$USER'/{usum += $NF}
         END{print psum, usum, sum}'
}

scale(){
    awk -vscale=$1 '{printf "%g", $1/scale;
                     for (i=2; i<=NF; i++) printf " %g", $i/scale;
                     printf "\n";}'
}

compute(){
    read -a columns <<< "Python $USER System"
    read -a cpu <<< $(ps -e --format=cmd,user,%cpu | sum)
    read -a memory <<< $(ps -e --format=cmd,user,rss | sum | scale 1048576)
    gpu_usage=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,nounits,noheader)
    read -a gpu_memory <<< $(
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,nounits,noheader \
        | tr , ' ' | scale 1024)
}

display(){
    echo "Consumption (ps) ==============="
    printf "%10s %9s %12s\n" " " "CPU (%)" "Memory (GiB)"
    printf "_%.0s" {0..30}; printf "\n"
    for i in "${!columns[@]}"
    do
        printf "%10s %9.2f %12.2f\n" ${columns[i]} ${cpu[i]} ${memory[i]}
    done
    echo
    echo "free ==========================="
    free -h
    echo
    echo "GPU ============================"
    awk -F, '{printf "%10s: %.2f%%\n", "Usage", $1}' <<< $gpu_usage
    awk '{printf "%10s: %.2f / %.2f (GiB)\n", "Memory", $1, $2}' <<< "${gpu_memory[@]}"
    echo "================================"
    echo
}

csv(){
    awk '{printf "%s", $1; for (i=2; i<=NF; i++) printf ",%s", $i; printf "\n";}' <<< "$@"
}

log(){
    file=.monitor.log
    if [ ! -f "$file" ]
    then
        echo "Date,Time,python_cpu_%,${USER}_cpu_%,system_cpu_%,python_memory_GiB,${USER}_memory_GiB,system_memory_GiB,RAM_used_GiB,GPU_used_%,GPU_memory_GiB" > $file
    fi

    if [ "$(echo "${cpu[0]} >= 1.0" | bc)" = 1 ]
    then
        csv "$(date +"%Y-%m-%d %H:%M:%S")" ${cpu[@]} ${memory[@]} $(free -h | awk '/Mem/{printf "%g", $3}') $gpu_usage ${gpu_memory[0]} >> $file
    fi
}

work(){
    compute
    display
    log
}

if [ "$1" == "watch" ]
then
    watch -d -n 1 $0
else
    work
fi
