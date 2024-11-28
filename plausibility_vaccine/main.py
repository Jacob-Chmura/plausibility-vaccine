from plausibility_vaccine.util import seed_everything, setup_basic_logging

seed = 0


def main() -> None:
    setup_basic_logging()
    seed_everything(seed)


if __name__ == '__main__':
    main()
