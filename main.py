import os
import optuna


def main() -> None:
    repository = os.getenv("INPUT_REPOSITORY")
    print(f"Hello world from github actions in {repository}!")


if __name__ == "__main__":
    main()
