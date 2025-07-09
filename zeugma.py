from sglang import assistant_begin, assistant_end
from sglang import assistant, function, gen, system, user
from sglang import image
from sglang import RuntimeEndpoint, set_default_backend
from sglang.srt.utils import load_image
from sglang.test.test_utils import is_in_ci
from sglang.utils import print_highlight, terminate_process, wait_for_server

if is_in_ci():
    from patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd


server_process, port = launch_server_cmd(
    "python -m sglang.launch_server --model-path Qwen/Qwen3-8B --host 0.0.0.0"
)

wait_for_server(f"http://localhost:{port}")
print(f"Server started on http://localhost:{port}")


@function
def text_qa(s, question):
    s += user(question)
    s += assistant(gen("answer", stop="\n"))


states = text_qa.run_batch(
    [
        {"question": "What is the capital of the United Kingdom?"},
        {"question": "What is the capital of France?"},
        {"question": "What is the capital of Japan?"},
    ],
    progress_bar=True,
)

for i, state in enumerate(states):
    print_highlight(f"Answer {i + 1}: {states[i]['answer']}")
