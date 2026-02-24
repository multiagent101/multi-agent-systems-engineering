import threading
import queue
import time
from typing import Any, Dict, List


class Task:
    def __init__(self, task_id: int, payload: Any):
        self.task_id = task_id
        self.payload = payload


class Result:
    def __init__(self, task_id: int, output: Any):
        self.task_id = task_id
        self.output = output


class Worker(threading.Thread):
    def __init__(self, worker_id: int, task_queue: queue.Queue, result_queue: queue.Queue):
        super().__init__()
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.local_state: Dict[str, Any] = {}
        self.daemon = True

    def run(self):
        while True:
            task: Task = self.task_queue.get()
            if task is None:
                self.task_queue.task_done()
                break
            result = self.process(task)
            self.result_queue.put(result)
            self.task_queue.task_done()

    def process(self, task: Task) -> Result:
        time.sleep(0.1)
        output = task.payload * 2
        return Result(task.task_id, output)


class Coordinator:
    def __init__(self, num_workers: int):
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.workers: List[Worker] = [
            Worker(i, self.task_queue, self.result_queue)
            for i in range(num_workers)
        ]
        self.global_state: Dict[str, Any] = {
            "results": {}
        }

    def start(self):
        for worker in self.workers:
            worker.start()

    def submit_tasks(self, tasks: List[Task]):
        for task in tasks:
            self.task_queue.put(task)

    def collect_results(self, expected_results: int):
        collected = 0
        while collected < expected_results:
            result: Result = self.result_queue.get()
            self.global_state["results"][result.task_id] = result.output
            collected += 1
            self.result_queue.task_done()

    def shutdown(self):
        for _ in self.workers:
            self.task_queue.put(None)
        for worker in self.workers:
            worker.join()


if __name__ == "__main__":
    coordinator = Coordinator(num_workers=4)
    coordinator.start()

    tasks = [Task(i, i) for i in range(10)]
    coordinator.submit_tasks(tasks)
    coordinator.collect_results(expected_results=10)
    coordinator.shutdown()

    print(coordinator.global_state)
