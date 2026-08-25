from typing import Any, Callable, List, Tuple


def call_in_batches(
    func: Callable[..., Any],
    args_list: List[Tuple[Any, ...]],
    batch_size: int = 10,
    start_index: int = 0,
    **kwargs: Any
) -> List[Any]:
    """
    Calls a function in batches with the provided arguments.

    Args:
        func (Callable[..., Any]): The function to call.
        args_list (List[Tuple[Any, ...]]): A list of tuples containing the arguments for each call.
        batch_size (int, optional): The number of calls to make in each batch. Defaults to 10.
        **kwargs (Any): Additional keyword arguments to pass to the function.

    Returns:
        List[Any]: A list of results from each function call.
    """
    results = []
    for i in range(start_index, len(args_list), batch_size):
        batch_args = args_list[i:i + batch_size]
        batch_results = [func(*args, **kwargs) for args in batch_args]
        results.extend(batch_results)
    return results 