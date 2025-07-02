#!/bin/bash

# Setup and test vLLM installation
echo "## SETUP VLLM"
echo "Testing vLLM setup..."
python test_vllm.py

if [ $? -ne 0 ]; then
    echo "vLLM setup failed. Running setup script..."
    python setup_vllm.py
    exit 1
fi

echo "## VLLM WIC EVALUATION"
echo "Running WiC evaluation with vLLM + Llama3..."

# Test with different prompt types and models
for prompt_type in "zeugma" "simple"; do
    for model in "meta-llama/Meta-Llama-3-8B-Instruct" "meta-llama/Meta-Llama-3-70B-Instruct"; do
        echo "======== $prompt_type with $model ========"

        # Run with different batch sizes based on model size
        if [[ $model == *"70B"* ]]; then
            batch_size=2
        else
            batch_size=8
        fi

        echo "Running with batch_size=$batch_size"
        python vllm_zeugma.py \
            --file data/wic.test.json \
            --start_index 0 \
            --type $prompt_type \
            --model $model \
            --batch_size $batch_size

        echo "Completed $prompt_type with $model"
        echo "Results saved to wic_vllm_${prompt_type}_results.jsonl"
        echo ""
    done
done

echo "## ANALYSIS"
echo "Analyzing results..."
echo "Check the generated .jsonl files for detailed results:"
ls -la wic_vllm_*_results.jsonl 2>/dev/null || echo "No vLLM results found"
