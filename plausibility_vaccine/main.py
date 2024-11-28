def hello_torch() -> str:
    import torch

    _ = torch.zeros(1337)
    return 'Hello PyTorch'


def main() -> None:
    print('Checking packages...', end='')
    hello_torch()
    print('Ok.')


if __name__ == '__main__':
    main()
