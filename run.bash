## QUERY MODEL
for pole in "dimension" "relation" "orientation"; do
    for model in "DeepSeek-R1-Distill-Llama-8B-Q8_0.gguf"; do
        echo "========$pole========"
        ctx="2048"
        if [[ "$model" = "DeepSeek-R1-Distill-Llama-8B-Q8_0.gguf" ]]; then
            ctx="16384"
        fi
        
        for ((i = 0; i < 5; i++)); do
            echo "rethorics $model"
            python prompt_generation.py data/$pole.csv few_shot.json $pole --rhetorics --model $model --seed $i --ctx $ctx
            echo "cot $model"
            python prompt_generation.py data/$pole.csv few_shot.json $pole --model $model --seed $i --ctx $ctx
        done
    done
done

echo "## RUN EVALUATION"
for pole in "dimension" "relation" "orientation"; do
    for model in "DeepSeek-R1-Distill-Llama-8B-Q8_0.gguf"; do
        echo "========$pole========"
        echo "rethorics $model"
        for ((i = 0; i < 5; i++)); do
            python compute_score.py --data data/$pole.csv --seed $i --rhetorics --path output --task $pole --model $model
        done
        echo "cot $model"
        for ((i = 0; i < 5; i++)); do
            python compute_score.py --data data/$pole.csv --seed $i --path output --task $pole --model $model
        done
    done
done
