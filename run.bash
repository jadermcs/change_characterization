## QUERY MODEL
for pole in "dimension" "relation" "orientation"; do
    for model in "Phi-3-mini-4k-instruct-fp16.gguf" "Meta-Llama-3-8B-Instruct-Q8_0.gguf" "Meta-Llama-3-70B-Instruct-Q2_K.gguf"; do
        echo "========$pole========"
        ctx="2048"
        if [[ "$model" = "Meta-Llama-3-70B-Instruct-Q2_K.gguf" ]]; then
            ctx="1024"
        fi
        
        for ((i = 0; i < 5; i++)); do
            echo "rethorics $model"
            python prompt_generation.py data/$pole.csv few_shot.json $pole --reason --style 0 --model $model --seed $i --ctx $ctx
            echo "cot $model"
            python prompt_generation.py data/$pole.csv few_shot.json $pole --reason --style 1 --model $model --seed $i --ctx $ctx
            echo "few-shot $model"
            python prompt_generation.py data/$pole.csv few_shot.json $pole --style 1 --model $model --seed $i
        done
    done
done

echo "## RUN EVALUATION"
for pole in "dimension" "relation" "orientation"; do
    for model in "Phi-3-mini-4k-instruct-fp16.gguf" "Meta-Llama-3-8B-Instruct-Q8_0.gguf" "Meta-Llama-3-70B-Instruct-Q2_K.gguf"; do
        echo "========$pole========"
        echo "rethorics $model"
        for ((i = 0; i < 5; i++)); do
            python compute_score.py data/$pole.csv $i output/0/ $pole $model
        done
        echo "cot $model"
        for ((i = 0; i < 5; i++)); do
            python compute_score.py data/$pole.csv $i output/1/ $pole $model
        done
        echo "few-shot $model"
        for ((i = 0; i < 5; i++)); do
            python compute_score.py data/$pole.csv $i output/1_noreason/ $pole $model
        done
    done
done
