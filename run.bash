## QUERY MODEL
for pole in "dimension" "relation" "orientation"; do
    for model in "microsoft_Phi-4-mini-instruct-Q8_0.gguf" "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf" "gemma-3-12b-it.Q8_0.gguf"; do
        echo "========$pole========"
        ctx="4096"
        
        echo "rethorics $model"
        python prompt_generation.py data/$pole.csv few_shot.json $pole --rhetorics --model $model --seed $i --ctx $ctx
        echo "cot $model"
        python prompt_generation.py data/$pole.csv few_shot.json $pole --model $model --seed $i --ctx $ctx
    done
done

echo "## RUN EVALUATION"
for pole in "dimension" "relation" "orientation"; do
    for model in "microsoft_Phi-4-mini-instruct-Q8_0.gguf" "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf" "gemma-3-12b-it.Q8_0.gguf"; do
        echo "========$pole========"
        echo "rethorics $model"
        python compute_score.py --data data/$pole.csv --seed $i --rhetorics --path output --task $pole --model $model
        echo "cot $model"
        python compute_score.py --data data/$pole.csv --seed $i --path output --task $pole --model $model
    done
done
