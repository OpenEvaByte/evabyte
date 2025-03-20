from pprint import pprint

from . import (
    ds1000,
    gsm,
    math_task,
    humaneval,
    humaneval_mbpp_plus,
    bigcodebench,
    cute
)

TASK_REGISTRY = {
    **ds1000.create_all_tasks(),
    **humaneval.create_all_tasks(),
    **gsm.create_all_tasks(),
    **math_task.create_all_tasks(),
    **humaneval_mbpp_plus.create_all_tasks(),
    **bigcodebench.create_all_tasks(),
    **cute.create_all_tasks()
}

ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name):
    try:
        return TASK_REGISTRY[task_name]()
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")
