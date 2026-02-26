from src.core.config import ConfigLoader
from src.core.logger import StructuredLogger
from src.data.alpaca import AlpacaClient


def main():
    logger = StructuredLogger("test")
    config_loader = ConfigLoader(logger=logger)
    client = AlpacaClient(config_loader=config_loader, logger=logger)
    try:
        actives = client.get_most_actives(top=50)
        print(f"Actives: {actives}")
    except Exception as e:
        print(f"Error actives: {e}")

    try:
        movers = client.get_movers(top=50)
        print(f"Movers: {movers}")
    except Exception as e:
        print(f"Error movers: {e}")


if __name__ == "__main__":
    main()
