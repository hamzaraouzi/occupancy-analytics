import click
from occupancy import Occupancy


@click.command()
@click.option("--source", type=str, help="rtsp link or video path")
@click.option("--model", type=str, help="model artifact")
def main(source, model):
    occupancy = Occupancy(source, model)
    occupancy.run()


if __name__ == "__main__":
    main()