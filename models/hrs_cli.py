import sys
from typing import Optional, Tuple, List

__version__ = "1.0.0"
__author__ = "wcrr51"


def print_info():
    print(f"Hybrid Recommender System V{__version__} by '{__author__}'")
    print(f"Dataset: Yelp Dataset [https://www.yelp.com/dataset/]'")


def print_help():
    print("help\t- show available commands")
    print("info\t- show system information")
    print("exit\t- exit the program")


def input_str(message: Optional[str] = None) -> str:
    message = f"{message}\n > " if message is not None else " > "
    string = input(message)
    while string == "" or string.isspace():
        string = input(message)
    return string


def input_command() -> Tuple[str, List[str]]:
    command_parts = input_str().split()
    return command_parts[0], command_parts[1:]


def main() -> int:
    print_info()
    username: str = input_str("\nEnter username:")
    print(f"Set active username to '{username}'")

    print("Type 'help' for a list of available commands.")

    while True:
        command, args = input_command()

        if command == "help":
            print_help()
        elif command == "info":
            print_info()
        elif command == "exit":
            return 0
        else:
            print("Unrecognised command. Type 'help' for a list of commands.")


if __name__ == "__main__":
    sys.exit(main())
