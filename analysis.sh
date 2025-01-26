#!/bin/bash

get_model_name() {
    local smooth=$1
    local corpus=$(basename "$2")
    
    case "$smooth:$corpus" in
        "l:pride.txt") echo "LM1" ;;
        "g:pride.txt") echo "LM2" ;;
        "i:pride.txt") echo "LM3" ;;
        "l:ulysses.txt") echo "LM4" ;;
        "g:ulysses.txt") echo "LM5" ;;
        "i:ulysses.txt") echo "LM6" ;;
        *) echo "Unknown model" ;;
    esac
}

mkdir -p results

total=$((3 * 3 * 2 * 2)) # n=3, smooth=3, corpus=2, set=2
current=0

show_progress() {
    local task="$1"
    local percentage=$((current * 100 / total))
    local filled=$((percentage / 2))
    local empty=$((50 - filled))
    
    # create the bar
    printf -v bar "%${filled}s" ""
    printf -v empty_bar "%${empty}s" ""
    bar=${bar// /#}
    empty_bar=${empty_bar// /-}
    
    # print progress bar with carriage return
    printf "\r[%s%s] %3d%% | %s" "$bar" "$empty_bar" "$percentage" "$task"
}

for n in 1 3 5; do
    for smooth in "l" "g" "i"; do
        for corpus in "data/pride.txt" "data/ulysses.txt"; do
            for set in "train" "test"; do
                ((current++))
                model_name=$(get_model_name "$smooth" "$corpus")
                task="Processing n=$n, ${model_name}, ${set} set"
                show_progress "$task"
                
                output_file="results/2023114009_${model_name}_${n}_${set}-perplexity.txt"
                python3 src/language_model.py -a "$set" -m "$smooth" -n "$n" "$corpus" > "$output_file"
            done
        done
    done
done

echo \n