from config import parse_input_parameters, get_config


def train(config):
    pass


if __name__ == "__main__":
    args = parse_input_parameters()

    config = get_config(args)
    print(f'Config for this training: {config}')

    print('Training model:')
    train(config)
    print('Model training ended.')
