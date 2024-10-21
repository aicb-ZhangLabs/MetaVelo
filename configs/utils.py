import os
import datetime
import warnings

from functools import cache


def get_current_time():
    # Get the current time
    current_time = datetime.datetime.now()

    # Format the time
    formatted_time = current_time.strftime("%m%d%Y_%H%M%S")

    return formatted_time


@cache
def get_output_path(append_date: bool = True) -> str:
    output_path = "./"
    if "OUTPUT_PATH" in os.environ:
        output_path = os.environ["OUTPUT_PATH"]
    else:
        warnings.warn(
            f"OUTPUT_PATH not found in environment variables. Using default path: {output_path}"
        )

    if append_date:
        output_path = os.path.join(output_path, get_current_time())

    return output_path
